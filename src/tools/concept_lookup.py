from __future__ import annotations
from typing import Optional
from langchain_core.tools import tool
from src.rag.retriever import VectorRetriever
from src.rag.models import RetrievalResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


_retriever: Optional["VectorRetriever"] = None


def initialize_concept_lookup(retriever: VectorRetriever) -> None:
    """
    Call once at application startup (lifespan context).
    Injects the VectorRetriever dependency into this module.
    """
    global _retriever
    _retriever = retriever

    logger.debug("Concept lookup initialized")

def _map_result_to_concept(result: RetrievalResult) -> dict:
    """
    Maps a single RetrievalResult → concept dict.

    Return shape:
        {
            "found": True,
            "explanation": str,        # full document text
            "simple_explanation": str, # truncated or metadata field
            "examples": list[str],      # from metadata if available, else []
            "related_concepts": list[str] # from metadataif available, else []

        }
    """
    explanation = (getattr(result, "text", "") or "").strip()
    metadata = getattr(result, "metadata", {}) or {}

    simple_explanation = (
        metadata.get("simple_explanation")
        or metadata.get("short_explanation")
        or metadata.get("summary")
    )
    if not simple_explanation:
        if len(explanation) <= 220:
            simple_explanation = explanation
        else:
            cutoff = explanation.rfind(" ", 0, 220)
            if cutoff == -1:
                cutoff = 220
            simple_explanation = f"{explanation[:cutoff].rstrip()}..."

    raw_examples = metadata.get("examples", [])
    if isinstance(raw_examples, list):
        examples = [str(item).strip() for item in raw_examples if str(item).strip()]
    elif isinstance(raw_examples, str):
        cleaned = raw_examples.strip()
        examples = [cleaned] if cleaned else []
    else:
        examples = []

    return {
        "found": True,
        "explanation": explanation,
        "simple_explanation": str(simple_explanation).strip(),
        "examples": examples,
        "related_concepts": metadata.get("related_concepted", [])
    }


def _not_found() -> dict:
    """Canonical 'miss' response — mirrors rubric_lookup pattern."""
    return {
        "found": False,
        "explanation": "",
        "simple_explanation": "",
        "examples": [],
        "related_concepts": []
    }

@tool
async def concept_lookup(concept_name: str) -> dict:
    """
    Look up an ML/AI concept explanation from the knowledge base.
    Returns explanation, simple_explanation, and examples.
    """
    if _retriever is None:
        logger.warning("concept_lookup called before retriever initialization")
        return _not_found()

    query = concept_name.strip()
    if not query:
        return _not_found()

    try:
        results = _retriever.retrieve_concepts(query=query, n_results=1)
        if not results:
            return _not_found()
        return _map_result_to_concept(results[0])
    except Exception as exc:
        logger.warning("Concept lookup failed for '%s': %s", concept_name, exc)
        return _not_found()
