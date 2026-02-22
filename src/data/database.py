"""
SQLite database client with WAL mode and connection pooling.
Handles all interview presistence operations.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.utils.logging_config import get_logger
from contextlib import contextmanager


logger = get_logger(__name__)


class Database:
    """
    Sqlite database for interview episodic memory.
    """
    def __init__(self, db_path="data/sqlite/interviews.db"):
        """Initialize database with production settings

        Args:
            db_path (str, optional): Path to sqlite database file. Defaults to "data/sqlite/interviews.db".
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._in_transaction = False  # Track if we're in a transaction context

        # enable WAL mode
        self._enable_wal_mode()

        logger.info(f"Database initialized at {db_path}")

        self._create_tables()
        self._migrate_schema()

    def _enable_wal_mode(self):
        """Enable write ahead logging for concurrent access.
        """
        cursor = self.conn.cursor()
        # enabling WAL
        cursor.execute("PRAGMA journal_mode = WAL;")
        
        result = cursor.fetchone()[0]
        logger.info(f"Journal mode: {result}")

        # synchronous mode
        cursor.execute("PRAGMA synchronous = NORMAL;")

        # setting busy timeout
        cursor.execute("PRAGMA busy_timeout = 5000;") 

        # enabling foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")

        self.conn.commit()

    def _create_tables(self):

        """Create all required tables if they don't exist"""
        cursor = self.conn.cursor()

        # Interviews Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT CHECK(status IN ('in_progress', 'completed', 'abandoned')) DEFAULT 'in_progress',
                difficulty_level TEXT,
                total_questions INTEGER DEFAULT 0,
                overall_score REAL
            )
        """)

        # Conversations table
        # Fields: id (auto), interview_id (FK), turn_number, timestamp,
        #         question_id, question_text, question_metadata (JSON),
        #         candidate_response, response_time_seconds
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            question_id TEXT,
            question_text TEXT,
            question_metadata TEXT, -- JSON stored as TEXT
            candidate_response TEXT,
            response_time_seconds REAL,

            FOREIGN KEY (interview_id)
                REFERENCES interviews(id)
                ON DELETE CASCADE
            )
        """)

        # evaluations table
        # Fields: id (auto), conversation_id (FK), technical_accuracy,
        #         completeness, depth, clarity, overall_score,
        #         evaluation_reasoning, feedback
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            technical_accuracy REAL,
            completeness REAL,
            depth REAL,
            clarity REAL,
            overall_score REAL,
            evaluation_reasoning TEXT,
            feedback TEXT,
            evaluation_data TEXT,  -- Full evaluation dict as JSON blob (key_points, misconceptions, etc.)

            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
            )
        """
        )

        # session_state table
        # Fields: interview_id (PK, FK), state_data (TEXT/JSON), last_updated
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                interview_id TEXT PRIMARY KEY,
                state_data TEXT,  -- JSON blob
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (interview_id)
                    REFERENCES interviews(id)
                    ON DELETE CASCADE
            )
        """)

        # agent_traces table
        # Fields: id (auto), interview_id (FK), timestamp, agent_name,
        #         action, input_data (JSON), output_data (JSON),
        #         latency_ms, success (BOOLEAN)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                agent_name TEXT NOT NULL,
                action TEXT,
                input_data TEXT,    -- JSON
                output_data TEXT,   -- JSON

                latency_ms INTEGER,
                success BOOLEAN DEFAULT 1,

                FOREIGN KEY (interview_id)
                    REFERENCES interviews(id)
                    ON DELETE CASCADE
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_interview 
            ON conversations(interview_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evaluations_conversation 
            ON evaluations(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_traces_interview 
            ON agent_traces(interview_id)
        """)

        self.conn.commit()
        logger.info("Database tables created/verified")

    def _migrate_schema(self):
        """Apply incremental schema migrations for existing databases.

        Safe to run repeatedly — checks for column existence before ALTER.
        """
        cursor = self.conn.cursor()

        # Migration 1: Add evaluation_data JSON blob column
        cursor.execute("PRAGMA table_info(evaluations)")
        columns = {row[1] for row in cursor.fetchall()}
        if "evaluation_data" not in columns:
            cursor.execute(
                "ALTER TABLE evaluations ADD COLUMN evaluation_data TEXT"
            )
            self.conn.commit()
            logger.info("Migration: added 'evaluation_data' column to evaluations")
    
    def _commit(self):
        """Commit changes only if not in a transaction context"""
        if not self._in_transaction:
            self.conn.commit()

    # ========== INTERVIEW OPERATIONS ==========
    def create_interview(
            self,
            interview_id: str,
            user_id: str = "default",
            difficulty_level: str = "medium"
    ) -> str:
        """create new interview session.

        Args:
            interview_id (str): Unique interview identifier
            user_id (str, optional):  User identifier. Defaults to "default".
            difficulty_level (str, optional): Starting difficulty. Defaults to "medium".

        Returns:
            str: interview id
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO interviews (id, user_id, difficulty_level)
               VALUES (?, ?, ?)""",
            (interview_id, user_id, difficulty_level)
        )
        self._commit()
        logger.info(f"Created interview {interview_id}")
        return interview_id

    def update_interview_status(
        self,
        interview_id: str,
        status: str,
        overall_score: Optional[float] = None
    ):
        """
        Update interview status and score.
        
        Args:
            interview_id: Interview to update
            status: New status (in_progress, completed, abandoned)
            overall_score: Final score (if completed)
        """
        cursor = self.conn.cursor()
        if status == 'completed':
            cursor.execute(
                """UPDATE interviews 
                   SET status=?, overall_score=?, completed_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (status, overall_score, interview_id)
            )
        else:
            cursor.execute(
                """UPDATE interviews 
                   SET status=?, overall_score=?
                   WHERE id=?""",
                (status, overall_score, interview_id)
            )
        self._commit()
        logger.info(f"Updated interview {interview_id} status to {status}")

    def get_interview(self, interview_id: str) -> Optional[Dict]:
        """
        Get interview metadata.
        
        Args:
            interview_id: Interview to retrieve
            
        Returns:
            Dictionary with interview data or None
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM interviews WHERE id=?", (interview_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    # ========== CONVERSATION OPERATIONS ==========
    
    def save_conversation_turn(
        self,
        interview_id: str,
        turn_number: int,
        question_id: str,
        question_text: str,
        question_metadata: Dict,
        candidate_response: Optional[str] = None,
        response_time_seconds: Optional[int] = None
    ) -> Optional[int]:
        """
        Save a Q&A turn.
        
        Args:
            interview_id: Interview this belongs to
            turn_number: Turn number (1, 2, 3, ...)
            question_id: ID of the question
            question_text: Full question text
            question_metadata: Question metadata as dict
            candidate_response: Candidate's answer
            response_time_seconds: Time taken to respond
            
        Returns:
            conversation_id (auto-incremented primary key)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO conversations 
               (interview_id, turn_number, question_id, question_text, 
                question_metadata, candidate_response, response_time_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (interview_id, turn_number, question_id, question_text,
             json.dumps(question_metadata), candidate_response, response_time_seconds)
        )
        self._commit()
        conversation_id = cursor.lastrowid
        logger.info(f"Saved conversation turn {turn_number} for interview {interview_id}")
        return conversation_id

    def update_conversation_response(
        self,
        conversation_id: int,
        candidate_response: str,
        response_time_seconds: int
    ) -> None:
        """
        Update conversation with candidate's response.
        
        Used when question is asked first, response comes later.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """UPDATE conversations 
               SET candidate_response=?, response_time_seconds=?
               WHERE id=?""",
            (candidate_response, response_time_seconds, conversation_id)
        )
        self._commit()
        logger.info(f"Updated conversation {conversation_id} with response")

    def get_conversation_history(
        self,
        interview_id: str
    ) -> List[Dict]:
        """
        Get full conversation history for an interview.
        
        Args:
            interview_id: Interview to retrieve
            
        Returns:
            List of conversation turns with evaluations
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT c.*, e.overall_score, e.feedback
               FROM conversations c
               LEFT JOIN evaluations e ON c.id = e.conversation_id
               WHERE c.interview_id = ?
               ORDER BY c.turn_number""",
            (interview_id,)
        )
        rows = cursor.fetchall()
        history = []
        for row in rows:
            turn = dict(row)
            # Parse question_metadata from JSON string to dict
            if turn.get('question_metadata'):
                turn['question_metadata'] = json.loads(turn['question_metadata'])
            history.append(turn)
        return history

    # ========== EVALUATION OPERATIONS ==========
    
    def save_evaluation(
        self,
        conversation_id: int,
        evaluation: Dict,
        feedback: str
    ):
        """
        Save evaluation results for a turn.
        
        Args:
            conversation_id: Conversation this evaluates
            evaluation: Dict with scores (technical_accuracy, completeness, etc.)
            feedback: Feedback text
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO evaluations 
               (conversation_id, technical_accuracy, completeness, depth, 
                clarity, overall_score, evaluation_reasoning, feedback,
                evaluation_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (conversation_id,
             evaluation.get('technical_accuracy'),
             evaluation.get('completeness'),
             evaluation.get('depth'),
             evaluation.get('clarity'),
             evaluation.get('overall_score'),
             evaluation.get('evaluation_reasoning'),
             feedback,
             json.dumps(evaluation))
        )
        self._commit()
        logger.info(f"Saved evaluation for conversation {conversation_id}")

    def get_evaluation(self, conversation_id: int) -> Optional[Dict]:
        """Get evaluation for a conversation turn.
        
        If evaluation_data JSON blob exists, it is parsed and merged
        into the returned dict for full evaluation details.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE conversation_id=?", (conversation_id,))
        row = cursor.fetchone()
        if not row:
            return None
        result = dict(row)
        # Parse evaluation_data JSON blob if present
        if result.get('evaluation_data'):
            try:
                result['evaluation_data'] = json.loads(result['evaluation_data'])
            except (json.JSONDecodeError, TypeError):
                pass  # Keep as string if parsing fails
        return result

    # ========== STATE MANAGEMENT (RESUMABILITY) ==========
    
    def save_state(self, interview_id: str, state_data: Dict):
        """
        Save interview state for resumability.
        
        Uses UPSERT (INSERT OR REPLACE) pattern.
        
        Args:
            interview_id: Interview identifier
            state_data: Complete LangGraph state as dict
        
        .. deprecated::
            Will be superseded by LangGraph's built-in checkpointer
            (AsyncPostgresSaver / AsyncSqliteSaver via RunnableConfig thread_id).
            Keep for debugging / pre-checkpointer MVP only.
            See architecture(v2).md § Key Design Decisions.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO session_state 
               (interview_id, state_data, last_updated)
               VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (interview_id, json.dumps(state_data))
        )
        self._commit()
        logger.info(f"Saved state for interview {interview_id}")

    def load_state(self, interview_id: str) -> Optional[Dict]:
        """
        Load interview state for resuming.
        
        Args:
            interview_id: Interview to resume
            
        Returns:
            State dict or None if not found
        
        .. deprecated::
            Will be superseded by LangGraph's built-in checkpointer.
            See save_state() docstring.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT state_data FROM session_state WHERE interview_id=?",
            (interview_id,)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def delete_state(self, interview_id: str):
        """Delete saved state (after interview completes)"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM session_state WHERE interview_id=?", (interview_id,))
        self._commit()
        logger.info(f"Deleted state for interview {interview_id}")

    # ========== AGENT TRACING (DEBUGGING) ==========
    
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
        
        Args:
            interview_id: Interview context
            agent_name: Which agent (evaluator, question_selector, etc.)
            action: What action (evaluate, select_question, etc.)
            input_data: Agent's input as dict
            output_data: Agent's output as dict
            latency_ms: Time taken in milliseconds
            success: Whether action succeeded
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO agent_traces 
               (interview_id, agent_name, action, input_data, output_data, 
                latency_ms, success)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (interview_id, agent_name, action, 
             json.dumps(input_data), json.dumps(output_data),
             latency_ms, success)
        )
        self._commit()
        logger.debug(f"Logged {agent_name}.{action} for interview {interview_id}")

    def get_agent_traces(
        self,
        interview_id: str,
        agent_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Get agent execution traces for an interview.
        
        Args:
            interview_id: Interview to trace
            agent_name: Optional filter by agent
            
        Returns:
            List of trace records
        """
        cursor = self.conn.cursor()
        if agent_name:
            cursor.execute(
                """SELECT * FROM agent_traces 
                   WHERE interview_id=? AND agent_name=?
                   ORDER BY timestamp""",
                (interview_id, agent_name)
            )
        else:
            cursor.execute(
                """SELECT * FROM agent_traces 
                   WHERE interview_id=?
                   ORDER BY timestamp""",
                (interview_id,)
            )
        rows = cursor.fetchall()
        traces = []
        for row in rows:
            trace = dict(row)
            # Parse JSON fields
            if trace.get('input_data'):
                trace['input_data'] = json.loads(trace['input_data'])
            if trace.get('output_data'):
                trace['output_data'] = json.loads(trace['output_data'])
            traces.append(trace)
        return traces

    # ========== ANALYTICS ==========
    
    def get_performance_trajectory(self, interview_id: str) -> List[float]:
        """
        Get list of scores in order (for difficulty adaptation).
        
        Returns:
            List of overall_scores [7.0, 8.5, 7.2, ...]
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT e.overall_score
               FROM conversations c
               JOIN evaluations e ON c.id = e.conversation_id
               WHERE c.interview_id = ?
               ORDER BY c.turn_number""",
            (interview_id,)
        )
        rows = cursor.fetchall()
        return [row[0] for row in rows if row[0] is not None]


    def get_interview_summary(self, interview_id: str) -> Dict:
        """
        Get summary statistics for an interview.
        
        Returns:
            Dict with: total_questions, avg_score, topics_covered, duration, etc.
        """
        cursor = self.conn.cursor()
        
        # Get interview metadata
        interview = self.get_interview(interview_id)
        if not interview:
            return {}
        
        # Get conversation count and scores
        cursor.execute(
            """SELECT COUNT(*) as total_questions,
                      AVG(e.overall_score) as avg_score
               FROM conversations c
               LEFT JOIN evaluations e ON c.id = e.conversation_id
               WHERE c.interview_id = ?""",
            (interview_id,)
        )
        stats = dict(cursor.fetchone())
        
        # Get topics covered from question metadata
        cursor.execute(
            """SELECT question_metadata FROM conversations
               WHERE interview_id = ?""",
            (interview_id,)
        )
        topics = set()
        for row in cursor.fetchall():
            if row[0]:
                metadata = json.loads(row[0])
                if 'topic' in metadata:
                    topics.add(metadata['topic'])
        
        # Calculate duration
        duration_seconds = None
        if interview.get('created_at') and interview.get('completed_at'):
            created = datetime.fromisoformat(interview['created_at'])
            completed = datetime.fromisoformat(interview['completed_at'])
            duration_seconds = (completed - created).total_seconds()
        
        return {
            'interview_id': interview_id,
            'status': interview.get('status'),
            'total_questions': stats.get('total_questions', 0),
            'avg_score': stats.get('avg_score'),
            'topics_covered': list(topics),
            'duration_seconds': duration_seconds,
            'difficulty_level': interview.get('difficulty_level'),
            'created_at': interview.get('created_at'),
            'completed_at': interview.get('completed_at')
        }

    # ========== UTILITY ==========
    
    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.
        
        Usage:
            with db.transaction():
                db.save_turn(...)
                db.save_evaluation(...)
                # Both commit together or rollback together
        """
        self._in_transaction = True
        try:
            yield
            self.conn.commit()
            logger.debug("Transaction committed")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
        finally:
            self._in_transaction = False
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")


def get_database() -> Database:
    """Get database instance"""
    from src.utils.config import get_settings
    settings = get_settings()
    return Database(db_path=str(settings.sqlite_db_path))






    