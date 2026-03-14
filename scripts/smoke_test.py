"""
Smoke test — validates full pipeline end-to-end with real LLMs.
Not a unit test. Run manually to confirm system is alive after wiring changes.

Usage:
    python -m scripts.smoke_test

Expected output:
    - Interview plan generated
    - First question retrieved
    - Evaluation + feedback generated
    - Next question selected
    - State snapshot showing correct keys updated
"""

import asyncio
import json
import sys
from pathlib import Path

# ── Ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.utils.config import get_settings
from src.utils.llm_factory import get_complex_llm, get_fast_llm
from src.utils.logging_config import get_logger
from src.rag.agentic_rag import AgenticRAGService
from src.rag.grader import DocumentGrader
from src.rag.query_refiner import QueryRefiner
from src.rag.cache import InterviewCacheStore
from src.rag.retriever import VectorRetriever
from src.data.vector_store import VectorStore
from src.tools.concept_lookup import initialize_concept_lookup
from src.graph.agent_registry import AgentRegistry
from src.graph.interview_graph import build_start_graph, build_interview_graph
from src.graph.state import initialize_state

logger = get_logger(__name__)

# ── Hardcoded test input ─────────────────────────────────────────────────────
# Deliberately partial answer (~6/10) to trigger follow-up path
CANDIDATE_RESPONSE = """
RAG is a way to provide extra context through retrieval and inject it into LLM.
"""

# Force a known topic so CANDIDATE_RESPONSE stays relevant regardless of plan
FORCED_TOPIC = "retrieval augmented generation"

SEPARATOR = "─" * 60


def _print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _print_state_snapshot(state: dict):
    """Print the keys we care about — not the entire state."""
    _print_section("STATE SNAPSHOT")

    plan = state.get("interview_plan", {})
    print(f"  topic_sequence     : {plan.get('topic_sequence', [])}")
    print(f"  difficulty_curve   : {plan.get('difficulty_curve', [])}")
    print(f"  question_count     : {state.get('question_count')}")
    print(f"  difficulty_level   : {state.get('difficulty_level')}")
    print(f"  should_continue    : {state.get('should_continue')}")
    print(f"  question_mode      : {state.get('question_mode')}")
    print(f"  follow_up_count    : {state.get('follow_up_count')}")
    print(f"  topics_covered     : {state.get('topics_covered')}")
    print(f"  stage              : {state.get('stage')}")
    print(f"  end_reason         : {state.get('end_reason')}")

    # Performance tracking
    trajectory = state.get("performance_trajectory", [])
    ema = state.get("ema_trajectory", [])
    print(f"  performance_traj   : {[round(s, 2) for s in trajectory]}")
    print(f"  ema_trajectory     : {[round(s, 2) for s in ema]}")

    # Evaluation (internal — hidden from candidate but visible here)
    evaluation = state.get("current_evaluation", {})
    if evaluation:
        print(f"\n  [Evaluation — internal]")
        print(f"  overall_score      : {evaluation.get('overall_score')}")
        print(f"  topic              : {evaluation.get('topic')}")
        print(f"  is_fallback        : {evaluation.get('is_fallback')}")
        missed = evaluation.get("key_points_missed", [])
        print(f"  key_points_missed  : {missed}")
        misconceptions = evaluation.get("misconceptions", [])
        print(f"  misconceptions     : {misconceptions}")

    # Conversation summary if populated
    summary = state.get("conversation_summary")
    if summary:
        print(f"\n  [Conversation Summary]")
        print(f"  {summary[:200]}{'...' if len(summary) > 200 else ''}")


async def main():
    settings = get_settings()

    print(f"\n{'═' * 60}")
    print("  AI INTERVIEW SYSTEM — SMOKE TEST")
    print(f"{'═' * 60}")
    print(f"  Primary LLM  : {settings.ollama_model}")
    print(f"  Secondary LLM: {settings.ollama_model_secondary}")
    print(f"  ChromaDB path: {settings.vector_db_path}")

    # ── 1. Infrastructure ────────────────────────────────────────────────────
    _print_section("1/6  Initializing Infrastructure")

    complex_llm = get_complex_llm()
    fast_llm = get_fast_llm()
    print("  ✓ LLMs loaded")

    vector_store = VectorStore()
    retriever = VectorRetriever(vector_store)
    print("  ✓ VectorStore + VectorRetriever ready")

    initialize_concept_lookup(retriever)
    print("  ✓ concept_lookup tool initialized")

    available_topics = retriever.get_available_topics()
    print(f"  ✓ Available topics ({len(available_topics)}): {available_topics}")

    cache_store = InterviewCacheStore()

    from src.rag.grader import DocumentGrader
    from src.rag.query_refiner import QueryRefiner
    grader = DocumentGrader(llm=fast_llm)
    refiner = QueryRefiner()
    rag_service = AgenticRAGService(
        retriever=retriever,
        grader=grader,
        refiner=refiner,
        cache_store=cache_store,
    )
    print("  ✓ CacheStore + AgenticRAGService ready")

    # ── 2. Agent Registry ────────────────────────────────────────────────────
    _print_section("2/6  Building AgentRegistry")

    agents = AgentRegistry(
        complex_llm=complex_llm,
        fast_llm=fast_llm,
        rag_service=rag_service,
        cache_store=cache_store,
        available_topics=available_topics,
        consistency_samples=settings.consistency_samples,
    )
    print("  ✓ All agents wired")

    # ── 3. Graphs ────────────────────────────────────────────────────────────
    async with AsyncSqliteSaver.from_conn_string("smoke_test_checkpoints.db") as checkpointer:

        _print_section("3/6  Compiling Graphs")

        start_graph = build_start_graph(agents, default_timeout=60.0)
        interview_graph = build_interview_graph(agents, checkpointer, default_timeout=60.0)
        print("  ✓ start_graph compiled")
        print("  ✓ interview_graph compiled")

        # ── 4. /start ────────────────────────────────────────────────────────
        _print_section("4/6  Running start_graph  (/start)")

        initial_state = initialize_state(
            difficulty="medium",
            time_budget_minutes=30,
            focus_topics=[FORCED_TOPIC],  # force known topic → CANDIDATE_RESPONSE stays valid
        )
        session_id = initial_state["interview_id"]
        print(f"  session_id: {session_id}")

        config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={"smoke_test": True, "turn": 0},
            tags=["smoke_test", "start"],
        )

        print("\n  Invoking start_graph (plan + first question)...")
        start_result = await start_graph.ainvoke(initial_state, config=config)

        plan = start_result.get("interview_plan", {})
        print(f"\n  ✓ Plan generated")
        print(f"    topic_sequence : {plan.get('topic_sequence', [])}")
        print(f"    difficulty_curve: {plan.get('difficulty_curve', [])}")
        print(f"    time_allocation : {plan.get('time_allocation', {})}")

        first_q = start_result.get("current_question", {})
        print(f"\n  ✓ First question selected")
        print(f"    topic      : {first_q.get('topic')}")
        print(f"    difficulty : {first_q.get('difficulty')}")
        print(f"    type       : {first_q.get('question_type')}")
        print(f"\n  ┌─ QUESTION ─────────────────────────────────────────")
        print(f"  │ {first_q.get('text', 'NO QUESTION TEXT')}")
        print(f"  └────────────────────────────────────────────────────")

        # ── 5. /submit_response ──────────────────────────────────────────────
        _print_section("5/6  Running interview_graph  (/submit_response)")

        print(f"\n  Candidate response (hardcoded):")
        print(f"  ┌─ RESPONSE ─────────────────────────────────────────")
        print(f"  │ {CANDIDATE_RESPONSE.strip()}")
        print(f"  └────────────────────────────────────────────────────")

        # Checkpointer loads existing state — only pass new input
        turn_config = RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={"smoke_test": True, "turn": 1},
            tags=["smoke_test", "submit"],
        )

        print("\n  Invoking interview_graph...")
        print("  (evaluator → feedback ∥ question_selector → supervisor → maybe_summarize)")

        # First turn: seed checkpoint with full start_graph state + candidate response
        # Subsequent turns only need {"candidate_response": ...} — checkpointer loads rest
        turn_result = await interview_graph.ainvoke(
            {**start_result, "candidate_response": CANDIDATE_RESPONSE},
            config=turn_config,
        )

        # Feedback (candidate-facing)
        feedback = turn_result.get("current_feedback", "NO FEEDBACK GENERATED")
        print(f"\n  ✓ Feedback generated")
        print(f"  ┌─ FEEDBACK ─────────────────────────────────────────")
        print(f"  │ {feedback}")
        print(f"  └────────────────────────────────────────────────────")

        # Next question
        next_q = turn_result.get("current_question", {})
        should_continue = turn_result.get("should_continue", False)

        if should_continue and next_q:
            mode = turn_result.get("question_mode", "unknown")
            print(f"\n  ✓ Next question selected  (mode: {mode})")
            print(f"  ┌─ NEXT QUESTION ────────────────────────────────────")
            print(f"  │ {next_q.get('text', 'NO QUESTION TEXT')}")
            print(f"  └────────────────────────────────────────────────────")
        else:
            end_reason = turn_result.get("end_reason", "unknown")
            print(f"\n  Interview ended — reason: {end_reason}")

        # ── 6. State snapshot ────────────────────────────────────────────────
        _print_section("6/6  State Snapshot")
        _print_state_snapshot(turn_result)

        # ── Result ───────────────────────────────────────────────────────────
        print(f"\n{'═' * 60}")
        print("  SMOKE TEST COMPLETE ✓")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())