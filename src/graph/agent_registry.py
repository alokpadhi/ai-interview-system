from langchain_core.language_models.chat_models import BaseChatModel

from src.rag.cache import InterviewCacheStore
from src.rag.agentic_rag import AgenticRAGService
from src.services.trend_analyzer import TrendAnalyzer
from src.agents.evaluator import EvaluatorAgent
from src.agents.feedback import FeedbackAgent
from src.agents.question_selector import QuestionSelectorAgent
from src.agents.supervisor import SupervisorAgent
from src.utils.logging_config import get_logger
from src.services.conversation import ConversationManager

logger = get_logger(__name__)


class AgentRegistry:
    def __init__(self, complex_llm: BaseChatModel, 
                 fast_llm: BaseChatModel, 
                 rag_service: AgenticRAGService,
                 available_topics: list[str],
                 cache_store: InterviewCacheStore, consistency_samples=1):
        trend_analyzer = TrendAnalyzer()

        self.evaluator = EvaluatorAgent(complex_llm, fast_llm, consistency_samples)
        self.conversation_manager = ConversationManager(complex_llm)
        self.supervisor = SupervisorAgent(
            complex_llm, 
            trend_analyzer,
            available_topics=available_topics)
        self.feedback = FeedbackAgent(fast_llm, cache_store)
        self.question_selector = QuestionSelectorAgent(
                                    rag_service, fast_llm,
                                    complex_llm,
                                    cache_store,
                                    self.supervisor.circuit_breaker)
        
        logger.info("AgentRegistry initialized with all agents")

