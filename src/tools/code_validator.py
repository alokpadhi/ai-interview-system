"""
AST-based syntax validator for candidate code responses.
Scope: syntax validation only — logic correctness and runtime behavior 
are not checked. Consumers must not infer correctness from a passing validation.
"""

from langchain_core.tools import tool
import ast
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _contains_code(text: str) -> bool:
    """
    Detect whether a candidate response contains code.
    
    Priority order:
    1. Code fence markers  → highest confidence
    2. Python-specific tokens → medium confidence
    3. Indented blocks after colon → lower confidence
    
    Returns True on first match.
    """
    # Tier 1: fenced code block marker (highest confidence)
    if "```" in text or "```python" in text:
        return True

    # Tier 2: Python-specific syntax markers
    if "self." in text or "__init__" in text or "->" in text:
        return True

    lines = text.splitlines()
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def ") and stripped.endswith(":") and "(" in stripped and ")" in stripped:
            return True
        if stripped.startswith("import ") or stripped.startswith("from "):
            return True

    # Tier 3: control/header line ending with ":" followed by an indented line
    for i in range(len(lines) - 1):
        current = lines[i].rstrip()
        if not current.endswith(":"):
            continue
        nxt = lines[i + 1]
        if nxt and nxt[:1].isspace():
            return True

    return False


def _extract_code(text: str) -> str:
    """
    Extract code block from response text.
    If code fence present, extract content between fences.
    Otherwise return full text — let ast.parse decide what's valid.
    """
    py_start = text.find("```python")
    if py_start != -1:
        start = py_start + len("```python")
        if start < len(text) and text[start] == "\n":
            start += 1
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    fence_start = text.find("```")
    if fence_start == -1:
        return text

    start = fence_start + 3
    if start < len(text):
        line_end = text.find("\n", start)
        if line_end != -1:
            lang = text[start:line_end].strip()
            if lang and all(c.isalnum() or c in "+#-._" for c in lang):
                start = line_end + 1
        if start < len(text) and text[start] == "\n":
            start += 1

    end = text.find("```", start)
    if end == -1:
        return text

    return text[start:end].strip()


def _validate_syntax(code: str) -> dict:
    """
    Run AST parse and return structured result.
    Never raises — all errors captured in return value.
    
    Returns:
        {
            "is_valid": bool,
            "errors": list[str],      # "Line N: message" format on failure
            "validation_scope": str,  # always "syntax_only"
            "language": str,          # always "python" for now
        }
    """
    result = {
        "is_valid": True,
        "errors": [],
        "validation_scope": "syntax_only",
        "language": "python",
    }
    try:
        ast.parse(code)
    except SyntaxError as exc:
        lineno = exc.lineno if exc.lineno is not None else "?"
        msg = exc.msg or str(exc)
        result["is_valid"] = False
        result["errors"] = [f"Line {lineno}: {msg}"]
    except Exception as exc:
        logger.warning("Unexpected validation error: %s", exc)
        result["is_valid"] = False
        result["errors"] = [f"Unexpected validation error: {exc}"]
    return result


# ─────────────────────────────────────────────────────────────────
# PUBLIC TOOL
# ─────────────────────────────────────────────────────────────────

@tool
async def code_validator(response: str) -> dict:
    """
    Validate Python code present in a candidate response for syntax correctness only.

    This tool should be called when the candidate's response may contain Python code
    and a syntax validation step is required. The validation strictly checks whether
    the code is syntactically valid Python. It does NOT check for logical correctness,
    runtime behavior, performance, or best practices.

    Behavior:
    - If no code is detected in the response, the tool returns `code_detected=False`
      and skips validation.
    - If code is detected, the tool extracts the code and performs a Python syntax check.

    Args:
        response (str): The candidate response which may contain Python code.

    Returns:
        dict: A dictionary with the following structure:
            {
                "code_detected": bool,      # Whether code was found in the response
                "is_valid": bool | None,    # True if syntax is valid, False if invalid, None if no code detected
                "errors": list[str],        # Syntax error messages (empty if valid or no code)
                "validation_scope": str,    # Always "syntax_only"
                "language": str             # Programming language validated ("python")
            }

    Notes:
    - This tool validates syntax only.
    - Logical errors, incorrect algorithms, missing imports, or runtime exceptions
      are intentionally not evaluated.
    """
    if not _contains_code(response):
        return {
            "code_detected": False,
            "is_valid": None,
            "errors": [],
            "validation_scope": "syntax_only",
            "language": "python",
        }

    code = _extract_code(response)
    result = _validate_syntax(code)
    result["code_detected"] = True
    return result
