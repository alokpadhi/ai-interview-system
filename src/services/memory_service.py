"""
Memory Service - 4 type memory architecture implementation.
Provides clean API for all memory operations.
"""

from typing import List, Dict, Optional
from datetime import datetime

from src.utils.logging_config import get_logger
from src.data.database import Database

logger = get_logger(__name__)

class MemoryService:
    """Centralized memory management service.

    1. Short-term: Conversation buffer (handled by caller with this service)
    2. Episodic: Complete interview history (SQLite)
    3. Semantic: Knowledge base (ChromaDB via VectorRetriever)
    4. Working: Agent state (LangGraph state + SQLite session_state) 
    """
    def __init__(self, database: Database):
        """
        Args:
            database (Database): DB Instance
        """
        self.db = database
        logger.info("Memory Service initialized.")

    # ===== INTERVIEW LIFECYCLE =====
    def create_interview(
            self,
            interview_id: str,
            user_id: str = "default",
            difficulty_level: str = "medium"
    ) -> str:
        """Start a new interview session

        Args:
            interview_id (str): interview id
            user_id (str, optional): user id. Defaults to "default".
            difficulty_level (str, optional): difficulty level. Defaults to "medium".

        Returns:
            str: interview id
        """
        interview_id = self.db.create_interview(
            interview_id, user_id, difficulty_level
        )
        return interview_id
    
    def complete_interview(
        self,
        interview_id: str,
        overall_score: float
    ):
        """
        Mark interview as completed.
        
        Also cleans up session_state (no longer needed).
        """
        self.db.update_interview_status(interview_id, 'completed', overall_score)
        self.db.delete_state(interview_id)
        logger.info(f"Interview {interview_id} completed with score {overall_score}")
    
    def abandon_interview(self, interview_id: str):
        """Mark interview as abandoned (user quit mid-way)"""
        self.db.update_interview_status(interview_id, 'abandoned')
        logger.info(f"Interview {interview_id} marked as abandoned")
    
    # ========== TURN MANAGEMENT ==========
    
    def save_turn(
        self,
        interview_id: str,
        turn_number: int,
        question: Dict,
        response: str,
        response_time_seconds: int
    ) -> Optional[int]:
        """
        Save a complete Q&A turn.
        
        Args:
            interview_id: Interview identifier
            turn_number: Turn number (1-indexed)
            question: Question dict with id, text, metadata
            response: Candidate's response
            response_time_seconds: Time taken
            
        Returns:
            conversation_id
        """
        question_id = question.get('id', '')
        question_text = question.get('text', '')
        question_metadata = {k: v for k, v in question.items() if k not in ['id', 'text']}
        
        conversation_id = self.db.save_conversation_turn(
            interview_id=interview_id,
            turn_number=turn_number,
            question_id=question_id,
            question_text=question_text,
            question_metadata=question_metadata,
            candidate_response=response,
            response_time_seconds=response_time_seconds
        )
        logger.info(f"Saved turn {turn_number} for interview {interview_id}")
        return conversation_id
    
    def save_evaluation(
        self,
        conversation_id: int,
        evaluation: Dict,
        feedback: str
    ):
        """
        Save evaluation results for a turn.
        
        Args:
            conversation_id: ID from save_turn()
            evaluation: Scores dict
            feedback: Feedback text
        """
        self.db.save_evaluation(conversation_id, evaluation, feedback)
        logger.info(f"Saved evaluation for conversation {conversation_id}")
    
    # ========== RETRIEVAL (EPISODIC MEMORY) ==========
    
    def get_interview_history(self, interview_id: str) -> List[Dict]:
        """
        Get complete interview transcript.
        
        Returns:
            List of turns with questions, responses, scores, feedback
        """
        return self.db.get_conversation_history(interview_id)
    
    def get_conversation_buffer(
        self,
        interview_id: str,
        last_n: int = 5
    ) -> List[Dict]:
        """
        Get recent conversation turns (short-term memory).
        
        Args:
            interview_id: Interview to retrieve from
            last_n: Number of recent turns to get
            
        Returns:
            List of last N conversation turns
        """
        history = self.get_interview_history(interview_id)
        return history[-last_n:] if history else []
    
    def get_performance_trajectory(self, interview_id: str) -> List[float]:
        """
        Get list of scores over time (for difficulty adaptation).
        
        Returns:
            List of scores [7.0, 8.5, 7.2, ...]
        """
        return self.db.get_performance_trajectory(interview_id)
    
    # ========== STATE MANAGEMENT (WORKING MEMORY) ==========
    
    def save_state(self, interview_id: str, state_data: Dict):
        """
        Save interview state for resumability.
        
        Args:
            interview_id: Interview identifier
            state_data: Complete LangGraph state
        
        .. deprecated::
            Will be superseded by LangGraph's built-in checkpointer
            (AsyncPostgresSaver / AsyncSqliteSaver via RunnableConfig thread_id).
            Keep for debugging / pre-checkpointer MVP only.
        """
        self.db.save_state(interview_id, state_data)
    
    def load_state(self, interview_id: str) -> Optional[Dict]:
        """
        Load interview state for resuming.
        
        Returns:
            State dict or None
        
        .. deprecated::
            Will be superseded by LangGraph's built-in checkpointer.
            See save_state() docstring.
        """
        return self.db.load_state(interview_id)
    
    # ========== DEBUGGING & OBSERVABILITY ==========
    
    def log_agent_action(
        self,
        interview_id: str,
        agent_name: str,
        action: str,
        input_data: Dict,
        output_data: Dict,
        latency_ms: int,
        success: bool = True
    ):
        """
        Log agent action for debugging.
        
        This creates an audit trail of all agent decisions.
        """
        self.db.log_agent_action(
            interview_id, agent_name, action, input_data, output_data, latency_ms, success
        )
    
    def get_agent_traces(
        self,
        interview_id: str,
        agent_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Get execution traces for debugging.
        
        Args:
            interview_id: Interview to trace
            agent_name: Optional filter
            
        Returns:
            List of agent action traces
        """
        return self.db.get_agent_traces(interview_id, agent_name)
    
    # ========== ANALYTICS ==========
    
    def get_interview_summary(self, interview_id: str) -> Dict:
        """
        Get aggregate statistics for an interview.
        
        Returns:
            {
                'total_questions': 10,
                'average_score': 7.8,
                'duration_minutes': 45,
                'topics_covered': ['optimization', 'deep_learning'],
                'difficulty_level': 'medium'
            }
        """
        summary = self.db.get_interview_summary(interview_id)
        # Convert duration from seconds to minutes if present
        if summary.get('duration_seconds'):
            summary['duration_minutes'] = round(summary['duration_seconds'] / 60, 1)
        return summary


def get_memory_service() -> MemoryService:
    """Get memory service instance"""
    from src.data.database import get_database
    return MemoryService(get_database())