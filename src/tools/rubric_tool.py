"""
Rubric lookup tool for EvaluatorAgent.
Provides exact key-value access to question rubrics loaded from a static JSON file.
File-based storage is intentional — rubrics are static reference data with a pure
key-value access pattern. No DB layer needed.
"""
import json
from json.decoder import JSONDecodeError
from pathlib import Path
from langchain_core.tools import tool
from src.utils.config import get_settings
from src.utils.logging_config import get_logger


BASE_DIR = Path(__file__).parent.parent.parent
logger = get_logger(__name__)

_RUBRIC_CACHE: dict[str, dict] = {}

def _load_rubrics() -> dict[str, dict]:
    """
    Load rubrics from JSON file into memory.
    Called once at module level — all subsequent lookups are O(1) dict access.
    
    Returns empty dict on file not found or parse error — 
    tool handles miss case gracefully.
    """
    rubric_path = BASE_DIR / get_settings().rubric_path
    try:
        with rubric_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.warning("Rubric file not found at path: %s", rubric_path)
        return {}
    except JSONDecodeError as exc:
        logger.warning("Failed to parse rubric JSON at %s: %s", rubric_path, exc)
        return {}

def _format_rubric(raw: dict) -> dict:
    """
    Normalise raw rubric entry into the shape EvaluatorAgent expects.
    Adds 'found' flag and flattens criteria for prompt injection.
    
    Args:
        raw: the value from _RUBRIC_CACHE[question_id]
    
    Returns:
        {
            "found": bool,
            "criteria": dict,      # full_structure - for _build_rubric_context()
            "key_points": list,    # flattened - For ValidationGate._extract_key_points()
            "common_mistakes": list
        }
    """
    criteria = raw.get("criteria", {})

    key_points: list = []
    common_mistakes: list = []
    for criterion in criteria.values():
        if not isinstance(criterion, dict):
            continue
        kp = criterion.get("key_points")
        if kp and isinstance(kp, list):
            key_points.extend(kp)
        cm = criterion.get("common_mistakes")
        if cm and isinstance(cm, list):
            common_mistakes.extend(cm)

    return {
        "found": True,
        "criteria": criteria,
        "key_points": key_points,
        "common_mistakes": common_mistakes,
    }

_RUBRIC_CACHE = _load_rubrics()

@tool
async def rubric_lookup(question_id: str) -> dict:
    """
    Retrieve the scoring rubric for a specific interview question.

    This tool should be called when the system needs the official evaluation
    rubric for a given question_id in order to grade or analyze a candidate's
    response. The rubric is stored in a JSON file and loaded into a module-level
    in-memory cache.

    Args:
        question_id (str): Unique identifier of the interview question.

    Returns:
        dict: A dictionary containing the rubric details. The structure is:

        If the rubric is found:
        {
            "found": True,
            "criteria": dict,          # Sub-score categories used for evaluation
            "key_points": list[str],   # Important concepts expected in a good answer
            "common_mistakes": list[str]  # Frequent mistakes candidates make
        }

        If the rubric is not found:
        {
            "found": False,
            "criteria": {},
            "key_points": [],
            "common_mistakes": []
        }

    Notes:
    - The rubric defines how a candidate's response should be evaluated.
    - If "found" is False, the evaluator should fall back to general
      evaluation heuristics instead of rubric-based scoring.
    """
    raw = _RUBRIC_CACHE.get(question_id)
    if raw is not None:
        return _format_rubric(raw)

    logger.debug("No rubric found for question_id: %s", question_id)
    return {
        "found": False,
        "criteria": {},
        "key_points": [],
        "common_mistakes": [],
    }
