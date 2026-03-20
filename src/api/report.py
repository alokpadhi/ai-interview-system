from datetime import datetime
import logging

from src.api.models import FinalReport
from src.graph.state import InterviewState

logger = logging.getLogger(__name__)

def generate_final_report(state: InterviewState) -> FinalReport:
    all_evaluations = state.get("all_evaluations", [])
    difficulties = state.get("difficulty_history", [])
    
    real_evals = [(e, d) for e, d in zip(all_evaluations, difficulties)
                  if not e.get("is_fallback", False)]
    fallback_count = len(all_evaluations) - len(real_evals)
    
    raw_scores = [e["overall_score"] for e, d in real_evals]
    raw_avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0
    
    weights = {"easy": 0.7, "medium": 1.0, "hard": 1.3}
    weighted_sum = sum(e["overall_score"] * weights.get(d, 1.0) for e, d in real_evals)
    max_possible = sum(10 * weights.get(d, 1.0) for e, d in real_evals)
    adjusted = (weighted_sum / max_possible) * 10 if max_possible > 0 else 0
    
    # Topic aggregation uses evaluation["topic"] field
    topic_scores = {}
    for e, d in real_evals:
        topic = e.get("topic", "general")
        if topic == "unknown":
            continue
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(e["overall_score"])
    topic_avgs = {t: sum(s)/len(s) for t, s in topic_scores.items()}
    
    notes = []
    if state.get("difficulty_reduced_due_to_performance"):
        original = state.get("original_difficulty", "medium")
        current = state["difficulty_level"]
        notes.append(f"Difficulty reduced from {original} to {current} due to performance")
    if fallback_count > 0:
        notes.append(f"{fallback_count} question(s) could not be evaluated (excluded from scoring)")
    
    elapsed = (datetime.now() - state["interview_start_time"]).total_seconds() / 60
    
    logger.info(f"all_evaluations count: {len(all_evaluations)}")
    logger.info(f"difficulty_history: {difficulties}")
    logger.info(f"scores: {[e['overall_score'] for e in all_evaluations]}")
    logger.info(f"real_evals count: {len(real_evals)}")
    logger.info(f"weighted_sum: {weighted_sum}, max_possible: {max_possible}")
    logger.info(f"raw_avg: {raw_avg}, adjusted: {adjusted}")
    # Bucket thresholds (all three buckets are now surfaced in the report):
    #   strengths          >= 7.0  — solid understanding
    #   needs_practice     6.0–6.9 — close; was silently dropped before
    #   areas_for_improvement < 6.0 — genuine gaps
    strengths             = [t for t, s in topic_avgs.items() if s >= 7.0]
    needs_practice        = [t for t, s in topic_avgs.items() if 6.0 <= s < 7.0]
    areas_for_improvement = [t for t, s in topic_avgs.items() if s < 6.0]

    return FinalReport(
        overall_score=round(raw_avg, 1),
        adjusted_score=round(adjusted, 1),
        questions_asked=state["question_count"],
        time_taken_minutes=round(elapsed, 1),
        difficulty_progression=difficulties,
        topic_scores={t: round(s, 1) for t, s in topic_avgs.items()},
        strengths=strengths,
        needs_practice=needs_practice,
        areas_for_improvement=areas_for_improvement,
        performance_notes=notes,
        fallback_count=fallback_count,
        detailed_evaluations=all_evaluations
    )