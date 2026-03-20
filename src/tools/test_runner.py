"""
Sandboxed test runner for LeetCode coding problems.

Loads test cases from leetcode_solutions.json and executes candidate code
against them using exec() in a restricted namespace with a per-test timeout.

Security note: exec() is used here for a local, single-user interview tool.
Do NOT expose this endpoint publicly without additional sandboxing (e.g. Docker).
"""

import ast
import copy
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

EXEC_TIMEOUT = 5  # seconds per test case

# ── Problems that require runtime infrastructure we don't inject ───────────────
# Trees need TreeNode; randomness has no deterministic expected value.
_SKIP_PROBLEMS: set[str] = {
    "code_007",  # binary tree (TreeNode)
    "code_009",  # binary tree (TreeNode)
    "code_014",  # tree serialization (TreeNode)
    "code_010",  # random / probabilistic — no deterministic expected
}

# Special harnesses (not standard class Solution + single-method)
_SPARSE_VECTOR_PROBLEM = "code_002"   # two SparseVector instances
_CODEC_PROBLEM = "code_020"           # encode → decode roundtrip
_STATEFUL_OPS_PROBLEM = "code_015"    # MedianFinder sequence of ops


# ── Solution cache ─────────────────────────────────────────────────────────────

def _load_solutions() -> dict[str, dict]:
    path = (
        Path(__file__).parent.parent.parent
        / "data/datasets/processed/solutions/leetcode_solutions.json"
    )
    try:
        with path.open() as f:
            data = json.load(f)
        return {sol["problem_id"]: sol for sol in data}
    except Exception as exc:
        logger.warning("test_runner: could not load solutions — %s", exc)
        return {}


_SOLUTIONS: dict[str, dict] = _load_solutions()


# ── Execution namespace ────────────────────────────────────────────────────────

def _base_ns() -> dict[str, Any]:
    """Restricted namespace with common algorithm imports."""
    ns: dict[str, Any] = {}
    for stmt in (
        "from collections import defaultdict, deque, Counter, OrderedDict",
        "from heapq import heappush, heappop, heapify, nlargest, nsmallest",
        "from typing import List, Optional, Dict, Set, Tuple, Any",
        "import math, bisect, functools, itertools",
        "from functools import lru_cache, cache",
    ):
        exec(stmt, ns)  # noqa: S102
    return ns


# ── AST helpers ────────────────────────────────────────────────────────────────

def _detect_main_method(code: str) -> tuple[str, bool]:
    """
    Parse the AST and return (method_name, is_void).

    Detection order:
    1. Class named 'Solution' (any casing) — return first non-__init__ method.
    2. Any top-level class — same search.
    3. Any top-level function — treat as standalone callable.

    is_void=True when the method/function has `-> None` annotation (in-place
    modification — check the modified first argument instead of the return value).
    Returns ("", False) only on SyntaxError or empty code.
    """
    if not code or not code.strip():
        return ("", False)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ("", False)

    def _first_method(cls_node: ast.ClassDef) -> tuple[str, bool] | None:
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef) and item.name != "__init__":
                is_void = isinstance(item.returns, ast.Constant) and item.returns.value is None
                return (item.name, is_void)
        return None

    # Pass 1: look for class Solution (case-insensitive)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.lower() == "solution":
            result = _first_method(node)
            if result:
                return result

    # Pass 2: any other top-level class
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            result = _first_method(node)
            if result:
                return result

    # Pass 3: standalone top-level function
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            is_void = isinstance(node.returns, ast.Constant) and node.returns.value is None
            return (node.name, is_void)

    return ("", False)


# ── Test harnesses ─────────────────────────────────────────────────────────────

def _exec_standard(code: str, test_case: dict, method_name: str, is_void: bool) -> dict:
    """
    Standard harness: class Solution with one main method, OR a standalone function.

    Resolution order for the callable:
    1. class Solution (any casing) → instantiate and call method
    2. Any other class whose method matches method_name
    3. Top-level callable named method_name (standalone function)

    Uses positional args (order from test_case["input"]) to handle key-name
    mismatches between test data and method signature (e.g. "new" vs "newInterval").
    """
    ns = _base_ns()
    exec(code, ns)  # noqa: S102

    # Find the callable — class Solution first, then standalone function.
    method = None

    # Look for a class named 'solution' (case-insensitive), guard instantiation.
    for name, obj in ns.items():
        if isinstance(obj, type) and name.lower() == "solution":
            try:
                m = getattr(obj(), method_name, None)
            except Exception:
                continue
            if callable(m):
                method = m
                break

    # Fallback: standalone function with the detected name
    if method is None:
        fn = ns.get(method_name)
        if callable(fn):
            method = fn

    if method is None:
        return {
            "passed": False, "actual": None,
            "error": f"Could not find callable '{method_name}' — "
                     "write 'class Solution' with the method, or a standalone function",
        }

    args = list(copy.deepcopy(test_case["input"]).values())
    expected = test_case["expected"]

    if is_void:
        method(*args)
        actual = args[0]   # first arg holds the in-place result
    else:
        actual = method(*args)
        # Fallback for standalone functions that modify in-place but lack -> None annotation:
        # if the function returned None and expected is non-None, check the first arg.
        if actual is None and expected is not None and args:
            actual = args[0]

    return {"passed": actual == expected, "actual": actual, "error": None}


def _exec_sparse_vector(code: str, test_case: dict) -> dict:
    """Harness for SparseVector (code_002): two instances, dotProduct."""
    ns = _base_ns()
    exec(code, ns)  # noqa: S102

    SV = ns.get("SparseVector")
    if not SV:
        return {"passed": False, "actual": None, "error": "'SparseVector' class not found"}

    v1 = SV(copy.deepcopy(test_case["input"]["v1"]))
    v2 = SV(copy.deepcopy(test_case["input"]["v2"]))
    actual = v1.dotProduct(v2)
    expected = test_case["expected"]
    return {"passed": actual == expected, "actual": actual, "error": None}


def _exec_codec(code: str, test_case: dict) -> dict:
    """Harness for Codec encode/decode (code_020): roundtrip equality."""
    ns = _base_ns()
    exec(code, ns)  # noqa: S102

    Codec = ns.get("Codec")
    if not Codec:
        return {"passed": False, "actual": None, "error": "'Codec' class not found"}

    codec = Codec()
    strs = copy.deepcopy(test_case["input"]["strs"])
    actual = codec.decode(codec.encode(strs))
    expected = test_case["expected"]
    return {"passed": actual == expected, "actual": actual, "error": None}


def _exec_stateful_ops(code: str, test_case: dict) -> dict:
    """
    Harness for MedianFinder (code_015).
    Runs ops like ['add(1)', 'add(2)', 'find()'] and collects find() results.
    """
    ns = _base_ns()
    exec(code, ns)  # noqa: S102

    cls = ns.get("MedianFinder")
    if not cls:
        return {"passed": False, "actual": None, "error": "'MedianFinder' class not found"}

    finder = cls()
    results: list[float] = []
    for op in test_case["input"]["ops"]:
        if op.startswith("add("):
            finder.addNum(int(op[4:-1]))
        elif op.startswith("find("):
            results.append(finder.findMedian())

    expected = test_case["expected"]
    return {"passed": results == expected, "actual": results, "error": None}


# ── Timeout wrapper ────────────────────────────────────────────────────────────

def _run_with_timeout(fn, *args) -> dict:
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn, *args)
        try:
            return future.result(timeout=EXEC_TIMEOUT)
        except FuturesTimeout:
            return {"passed": False, "actual": None, "error": f"Time limit exceeded ({EXEC_TIMEOUT}s)"}
        except Exception as exc:
            return {"passed": False, "actual": None, "error": type(exc).__name__ + ": " + str(exc)}


# ── Public API ─────────────────────────────────────────────────────────────────

def run_tests(question_id: str, code: str) -> dict:
    """
    Run all test cases for question_id against the candidate's code.

    Args:
        question_id: e.g. "code_001"
        code: raw candidate code string

    Returns:
        {
            "skipped": bool,
            "skip_reason": str | None,
            "passed": int,
            "failed": int,
            "total": int,
            "results": [
                {
                    "index": int,
                    "input": dict,
                    "expected": any,
                    "actual": any,
                    "passed": bool,
                    "error": str | None
                },
                ...
            ]
        }
    """
    # Guard: empty submission
    if not code or not code.strip():
        return _skipped("No code submitted")

    solution = _SOLUTIONS.get(question_id)
    if not solution:
        return _skipped("No test cases found for this problem")

    if question_id in _SKIP_PROBLEMS:
        return _skipped("Requires runtime infrastructure (trees / non-deterministic)")

    test_cases = solution.get("test_cases", [])
    if not test_cases:
        return _skipped("No test cases in solution file")

    # Guard: syntax check before dispatching to harness
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        return _skipped(f"Syntax error in submitted code — Line {exc.lineno}: {exc.msg}")

    # Select harness
    if question_id == _SPARSE_VECTOR_PROBLEM:
        raw_results = [_run_with_timeout(_exec_sparse_vector, code, tc) for tc in test_cases]
    elif question_id == _CODEC_PROBLEM:
        raw_results = [_run_with_timeout(_exec_codec, code, tc) for tc in test_cases]
    elif question_id == _STATEFUL_OPS_PROBLEM:
        raw_results = [_run_with_timeout(_exec_stateful_ops, code, tc) for tc in test_cases]
    else:
        method_name, is_void = _detect_main_method(code)
        if not method_name:
            return _skipped(
                "Could not detect a callable — write a class Solution or a standalone function"
            )
        raw_results = [
            _run_with_timeout(_exec_standard, code, tc, method_name, is_void)
            for tc in test_cases
        ]

    results = [
        {
            "index": i + 1,
            "input": tc["input"],
            "expected": tc["expected"],
            "actual": res.get("actual"),
            "passed": res.get("passed", False),
            "error": res.get("error"),
        }
        for i, (tc, res) in enumerate(zip(test_cases, raw_results))
    ]

    passed = sum(1 for r in results if r["passed"])
    return {
        "skipped": False,
        "skip_reason": None,
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }


def _skipped(reason: str) -> dict:
    return {
        "skipped": True,
        "skip_reason": reason,
        "passed": 0,
        "failed": 0,
        "total": 0,
        "results": [],
    }
