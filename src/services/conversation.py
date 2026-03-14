from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
import re

from src.utils.logging_config import get_logger
from src.graph.state import InterviewState

logger = get_logger(__name__)

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical expert and your task is to summarize an AI interview"
     "for context continuity. Create a concise summary which should contain:\n"
     "- Topic covered\n"
     "- Key strengths demonstrated\n"
     "- Key weaknesses and gaps identified\n"
     "- Any misconceptions addressed\n"
     "DO NOT include full question or answer text, no sort of numerical scores"
     "or verbose descriptions and reasoning of scores."),
     (
         "human", (
             "Existing summary:\n{existing_summary}\n"
             "New turns to incorporate:\n{new_turns}\n"
             "Updated Summary:"
         )
     )
])

class ConversationManager:
    """
    Manages conversation context to prevent unbounded memory growth.
    Runs as a LangGraph node (not post-hoc outside the graph) so it
    is checkpointed like every other node.
    
    Strategy:
    - Keep last 3 turns in full detail (recent context)
    - Summarize older turns (compressed context)
    - Re-summarize every 3 new turns (batch efficiency)
    
    Memory Budget:
    - Summary: ~200 tokens (fixed)
    - Recent 3 turns: ~1500 tokens (variable)
    - Total: ~1700 tokens (bounded)
    """
    MAX_RECENT_TURNS = 3
    SUMMARIZE_EVERY_N_TURNS = 3

    def __init__(self, complex_llm: BaseChatModel):
        self.complex_llm = complex_llm
        self.summarize_chain = SUMMARIZATION_PROMPT | complex_llm

    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int) -> str:
        """Truncate at the last sentence boundary within max_chars."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('? '),
            truncated.rfind('! '),
            truncated.rfind('.\n'),
        )
        if last_period > max_chars * 0.5:
            return truncated[:last_period + 1]
        return truncated + "..."
    
    async def maybe_update_summary(
            self,
            state: InterviewState,
            config: RunnableConfig
    ) -> dict:
        """
        LangGraph node: check if summary needs update, perform if needed.
        Returns partial state update (only changed keys).
        No-op most turns — returns empty dict when no update needed.
        """
        messages = state["messages"]

        turn_count = sum(1 for message in messages if isinstance(message, HumanMessage))
        summarized_count = state.get("summary_turn_count", 0)

        unsummarized = turn_count - summarized_count - self.MAX_RECENT_TURNS

        if unsummarized < self.SUMMARIZE_EVERY_N_TURNS:
            return {}
        
        to_summarize_end = -(self.MAX_RECENT_TURNS * 2)
        to_summarize = messages[:to_summarize_end]

        new_summary = await self._create_summary(
            existing_summary=state.get("conversation_summary", ""),
            new_turns=to_summarize[-(self.SUMMARIZE_EVERY_N_TURNS * 2):],
            config=config
        )

        return {
            "conversation_summary": new_summary,
            "summary_turn_count": turn_count - self.MAX_RECENT_TURNS
        }
    async def _create_summary(
            self,
            existing_summary: str,
            new_turns: list,
            config: RunnableConfig
    ) -> str:
        response = await self.summarize_chain.ainvoke({
            "existing_summary": existing_summary or "No previous summary",
            "new_turns": self._format_for_summary(new_turns)
        }, config=config)

        return response.content.strip()
    
    def _format_for_summary(self, new_turns: list) -> str:
        """Truncate at sentence boundaries, not mid-statement."""
        formatted = []
        for i in range(0, len(new_turns), 2):
            q = new_turns[i].content if hasattr(new_turns[i], 'content') else str(new_turns[i])
            a = new_turns[i+1].content if i+1 < len(new_turns) and hasattr(new_turns[i+1], 'content') else ""
            formatted.append(
                f"Q: {self._truncate_at_sentence(q, 200)}\n"
                f"A: {self._truncate_at_sentence(a, 300)}"
            )
        return "\n".join(formatted)

    def _format_full(self, messages: list) -> str:
        formatted = []
        for i in range(0, len(messages), 2):
            q = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
            a = messages[i+1].content if i+1 < len(messages) and hasattr(messages[i+1], 'content') else ""
            formatted.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(formatted)
    
    def get_context_for_agent(self, 
                                    state: InterviewState) -> str:
        messages = state["messages"]
        turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))

        if turn_count <= self.MAX_RECENT_TURNS:
            return self._format_full(messages)
        
        recent_start = -(self.MAX_RECENT_TURNS*2)
        recent = messages[recent_start:]

        summary = state.get("conversation_summary", "")
        context = ""

        if summary:
            context += f"Previous context:\n{summary}\n\n"
        context += f"Recent turns:\n{self._format_full(recent)}"
        return context
