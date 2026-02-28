"""
Offline, idempotent script to normalize topic fields in the 6 source JSON
files for the AI Interview System.

Usage:
    # Preview changes without writing anything
    python scripts/normalize_topics.py --dry-run

    # Apply changes in-place (originals are backed up to *.bak)
    python scripts/normalize_topics.py

    # Apply to a specific file only
    python scripts/normalize_topics.py --file data/datasets/.../foo.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter
from typing import Optional

from src.utils.topic_taxonomy import TopicNormalizer, CANONICAL_TOPICS

QUESTION_FILES = [
    # "data/datasets/processed/interview_questions/filtered_github_kaggle_iqs.json",
    # "data/datasets/processed/interview_questions/interview_questions_chip.json",
    # "data/datasets/processed/interview_questions/interview_questions.json",
    # "data/datasets/processed/interview_questions/leetcode_questions.json",
    # "data/datasets/processed/interview_questions/llm_generated_iqs.json",
    # "data/datasets/processed/interview_questions/system_design_iqs.json",
    "data/datasets/processed/interview_questions/final_interview_questions.json",    
]

_normalizer = TopicNormalizer()


def _normalize_file(file_path: Path, dry_run: bool) -> dict:
    """
    Normalize topics in a single JSON file.

    Returns a summary dict:
      {
        "total": int,
        "changed": int,
        "changes": Counter({(raw, canonical): count}),
        "unknown_fallbacks": list[str],
      }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = {
        "total": len(data),
        "changed": 0,
        "changes": Counter(),
        "unknown_fallbacks": [],
    }

    for item in data:
        raw = item.get("topic", "")
        canonical = _normalizer.normalize(raw)
        if canonical != raw:
            summary["changes"][(raw, canonical)] += 1
            summary["changed"] += 1
            if canonical == "general" and raw not in ("general", "", None):
                summary["unknown_fallbacks"].append(raw)
            if not dry_run:
                item["topic"] = canonical

    if not dry_run and summary["changed"] > 0:
        # Back up original before overwriting
        backup = file_path.with_suffix(".json.bak")
        shutil.copy2(file_path, backup)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return summary


def _print_file_summary(file_path: Path, summary: dict) -> None:
    print(f"\n{'─' * 70}")
    print(f"  File : {file_path.name}")
    print(f"  Items: {summary['total']}   Changed: {summary['changed']}")
    if summary["changes"]:
        print("  Topic changes:")
        for (raw, canonical), count in sorted(summary["changes"].items(), key=lambda x: x[0][0]):
            marker = "⚠ " if canonical == "general" else "  "
            print(f"    {marker}{raw!r:50s} → {canonical!r}  (×{count})")
    if summary["unknown_fallbacks"]:
        print(f"\n  ⚠  {len(summary['unknown_fallbacks'])} topic(s) fell back to 'general':")
        for t in sorted(set(summary["unknown_fallbacks"])):
            print(f"       {t!r}")


def run(files: list[Path], dry_run: bool) -> None:
    mode_label = "DRY RUN (no files written)" if dry_run else "APPLYING changes (originals backed up to *.bak)"
    print(f"\n{'=' * 70}")
    print(f"  Topic Normalizer — {mode_label}")
    print(f"  Canonical topics: {len(CANONICAL_TOPICS)}")
    print(f"{'=' * 70}")

    total_changed = 0
    for fp in files:
        if not fp.exists():
            print(f"\n  [SKIP] {fp} — file not found")
            continue
        summary = _normalize_file(fp, dry_run=dry_run)
        _print_file_summary(fp, summary)
        total_changed += summary["changed"]

    print(f"\n{'=' * 70}")
    print(f"  Total records updated: {total_changed}")
    if dry_run:
        print("  Re-run without --dry-run to apply changes.")
    else:
        print("  Done. Originals backed up as *.bak next to each file.")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize topic fields in interview question JSON source files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Normalize a single file instead of the default set.",
    )
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = [Path(f) for f in QUESTION_FILES]

    run(files, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
