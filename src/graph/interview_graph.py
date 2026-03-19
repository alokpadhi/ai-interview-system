import asyncio
import functools

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig

from src.utils.logging_config import get_logger
from src.graph.agent_registry import AgentRegistry
from src.graph.state import InterviewState

logger = get_logger(__name__)


def _wrap_with_timeout(agent_fn, timeout_seconds: float = 120.0):
    """Wrap an agent's execute method with asyncio.wait_for() timeout.
    TimeoutError is caught by the circuit breaker → fallback."""
    @functools.wraps(agent_fn)
    async def wrapped(state: InterviewState, config: RunnableConfig):
        return await asyncio.wait_for(
            agent_fn(state, config), timeout=timeout_seconds
        )
    return wrapped

def build_start_graph(agents: AgentRegistry, default_timeout: float = 120.0) -> CompiledStateGraph:
    """
    /start: Plan creation → first question retrieval.
    Supervisor only creates plan (no RAG dependency).
    QS handles first topic retrieval via its normal path.
    Background pre-warming triggered by API layer.
    """
    graph = StateGraph(InterviewState)

    graph.add_node("create_plan", _wrap_with_timeout(
        agents.supervisor.create_interview_plan))
    graph.add_node("first_question", _wrap_with_timeout(
        agents.question_selector.execute))

    graph.add_edge(START, "create_plan")
    graph.add_edge("create_plan", "first_question")
    graph.add_edge("first_question", END)

    compiled_start_graph =  graph.compile()
    logger.info("start_graph compiled successfully")

    return compiled_start_graph

def build_interview_graph(agents: AgentRegistry, 
                          checkpointer: BaseCheckpointSaver,
                          default_timeout: float = 120.0
    ) -> CompiledStateGraph:
    """
    Parallel execution with summarization as a checkpointed graph node.
    
    Flow: Evaluator → (Feedback || QS) → Supervisor → MaybeSummarize → END
    
    Key isolation ensures no state conflicts during parallel execution.
    Summarization runs after supervisor_check — if no update needed,
    returns empty dict (no-op).
    """
    graph = StateGraph(InterviewState)

    graph.add_node("evaluator", _wrap_with_timeout(
        agents.evaluator.execute
    ))
    graph.add_node("feedback", _wrap_with_timeout(
        agents.feedback.execute
    ))
    graph.add_node("question_selector", _wrap_with_timeout(
        agents.question_selector.execute
    ))
    graph.add_node("supervisor_check", _wrap_with_timeout(
        agents.supervisor.validate_and_decide
    ))
    graph.add_node("maybe_summarize", _wrap_with_timeout(
        agents.conversation_manager.maybe_update_summary
    ))

    # starting point after candidate response to first question 
    graph.add_edge(START, "evaluator")

    # fanout: both depend on evaluator, run in parallel
    graph.add_edge("evaluator", "feedback")
    graph.add_edge("evaluator", "question_selector")

    # fanin: supervisor waits for both to finish and update state
    graph.add_edge("feedback", "supervisor_check")
    graph.add_edge("question_selector", "supervisor_check")

    # summarization (if required) after supervisor check -> checkpointed
    graph.add_edge("supervisor_check", "maybe_summarize")
    graph.add_edge("maybe_summarize", END)

    compiled_interview_graph = graph.compile(
        checkpointer=checkpointer,
         # interrupt_before=["supervisor_check"],  # Uncomment for admin review mode
    )

    logger.info("interview_graph compiled successfully")

    return compiled_interview_graph


