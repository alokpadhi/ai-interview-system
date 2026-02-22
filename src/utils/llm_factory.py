"""
LLM Factory - Provider-agnostic LLM instantiation.
Supports Ollama, OpenAI, Anthropic with unified interface.
"""


from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from src.utils.config import Settings


class LLMFactory:
    """Factory for creating LLM instances based on configuration.
    
    Dual-model strategy (architecture v2):
      - complex_llm (14B): Evaluation, plan creation, follow-up generation,
                           summarization — tasks needing deeper reasoning
      - fast_llm    (7B):  Grading, question selection, feedback generation,
                           reflection — latency-sensitive tasks
    """
    @staticmethod
    def create_llm(settings: Settings) -> BaseChatModel:
        """Create complex/primary LLM instance (14B for Ollama).
        
        Used by: EvaluatorAgent (CoT), SupervisorAgent (planning),
                 QuestionSelectorAgent (follow-ups), ConversationManager.

        Args:
            settings (Settings): config settings

        Returns:
            BaseChatModel: Langchain LLM instance

        Raises:
            ValueError: If provider is unsupported
        """
        config = settings.llm_config
        if settings.llm_provider == "ollama":
            llm = ChatOllama(model=config["model"])
        elif settings.llm_provider == "anthropic":
            llm = ChatAnthropic(model=config["model"], api_key=config["api_key"])
        elif settings.llm_provider == "openai":
            llm = ChatOpenAI(model=config["model"], api_key=config["api_key"])
        else:
            raise ValueError(f"Unsupported LLM Provider: {settings.llm_provider}")
        
        return llm
    
    @staticmethod
    def create_secondary_llm(settings: Settings) -> BaseChatModel:
        """Create fast/secondary LLM instance (7B for Ollama).
        
        Used by: DocumentGrader, FeedbackAgent, EvaluatorAgent (reflection),
                 QuestionSelectorAgent (ReAct selection).
        
        This uses a smaller/faster model for cost/latency optimization.
        
        Args:
            settings: Config settings
            
        Returns:
            BaseChatModel: LangChain LLM instance (smaller model)
            
        Raises:
            ValueError: If provider is unsupported
        """
        config = settings.llm_config_secondary
        
        if settings.llm_provider == "ollama":
            return ChatOllama(
                model=config["model"],
            )
        elif settings.llm_provider == "anthropic":
            return ChatAnthropic(
                model=config["model"],  # Uses claude-haiku
                api_key=config["api_key"]
            )
        elif settings.llm_provider == "openai":
            return ChatOpenAI(
                model=config["model"],  # Uses gpt-4o-mini
                api_key=config["api_key"]
            )
        else:
            raise ValueError(f"Unsupported LLM Provider: {settings.llm_provider}")
    
def get_llm() -> BaseChatModel:
    """Convenience function to get complex/primary LLM (14B)."""
    from src.utils.config import get_settings
    return LLMFactory.create_llm(get_settings())

def get_secondary_llm() -> BaseChatModel:
    """Convenience function to get fast/secondary LLM (7B)."""
    from src.utils.config import get_settings
    return LLMFactory.create_secondary_llm(get_settings())

# Standardized aliases matching architecture v2 naming
get_complex_llm = get_llm
get_fast_llm = get_secondary_llm

