"""
Unit tests for Database class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.data.database import Database


class TestDatabase:
    """Test suite for Database"""
    
    @pytest.fixture
    def db_temp(self):
        """Temporary database for each test"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"
        db = Database(str(db_path))
        yield db
        db.close()
        shutil.rmtree(temp_dir)
    
    def test_create_interview(self, db_temp):
        """Test interview creation"""
        interview_id = db_temp.create_interview(
            interview_id="test_001",
            user_id="user_123",
            difficulty_level="medium"
        )
        
        assert interview_id == "test_001"
        
        # Verify it was created
        interview = db_temp.get_interview("test_001")
        assert interview is not None
        assert interview['user_id'] == "user_123"
        assert interview['difficulty_level'] == "medium"
        assert interview['status'] == "in_progress"
    
    def test_save_conversation_turn(self, db_temp):
        """Test saving Q&A turn"""
        # Create interview first
        db_temp.create_interview("test_001")
        
        # Save turn
        conv_id = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=1,
            question_id="q_001",
            question_text="What is ML?",
            question_metadata={"difficulty": "easy"},
            candidate_response="ML is...",
            response_time_seconds=120
        )
        
        assert conv_id > 0
        
        # Verify
        history = db_temp.get_conversation_history("test_001")
        assert len(history) == 1
        assert history[0]['question_text'] == "What is ML?"
    
    def test_save_evaluation(self, db_temp):
        """Test saving evaluation"""
        # Setup
        db_temp.create_interview("test_001")
        conv_id = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=1,
            question_id="q_001",
            question_text="Test",
            question_metadata={},
            candidate_response="Answer",
            response_time_seconds=60
        )
        
        # Save evaluation
        evaluation = {
            'technical_accuracy': 8.0,
            'completeness': 7.5,
            'depth': 7.0,
            'clarity': 9.0,
            'overall_score': 7.9
        }
        db_temp.save_evaluation(
            conversation_id=conv_id,
            evaluation=evaluation,
            feedback="Good answer!"
        )
        
        # Verify
        saved_eval = db_temp.get_evaluation(conv_id)
        assert saved_eval is not None
        assert saved_eval['overall_score'] == 7.9
        assert saved_eval['feedback'] == "Good answer!"
    
    def test_state_save_load(self, db_temp):
        """Test state persistence"""
        db_temp.create_interview("test_001")
        
        state = {
            "question_count": 3,
            "topics_covered": ["optimization", "deep_learning"],
            "difficulty_level": "medium"
        }
        
        # Save
        db_temp.save_state("test_001", state)
        
        # Load
        loaded = db_temp.load_state("test_001")
        assert loaded is not None
        assert loaded['question_count'] == 3
        assert loaded['topics_covered'] == ["optimization", "deep_learning"]
    
    def test_complete_interview(self, db_temp):
        """Test completing interview"""
        db_temp.create_interview("test_001")
        
        db_temp.update_interview_status(
            interview_id="test_001",
            status="completed",
            overall_score=8.2
        )
        
        interview = db_temp.get_interview("test_001")
        assert interview['status'] == "completed"
        assert interview['overall_score'] == 8.2
        assert interview['completed_at'] is not None
    
    def test_performance_trajectory(self, db_temp):
        """Test getting performance trajectory"""
        # Setup interview with multiple turns and evaluations
        db_temp.create_interview("test_001")
        
        # Turn 1
        conv_id1 = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=1,
            question_id="q1",
            question_text="Question 1",
            question_metadata={},
            candidate_response="Answer 1",
            response_time_seconds=60
        )
        db_temp.save_evaluation(conv_id1, {'overall_score': 7.0}, "Good")
        
        # Turn 2
        conv_id2 = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=2,
            question_id="q2",
            question_text="Question 2",
            question_metadata={},
            candidate_response="Answer 2",
            response_time_seconds=90
        )
        db_temp.save_evaluation(conv_id2, {'overall_score': 8.5}, "Excellent")
        
        # Turn 3
        conv_id3 = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=3,
            question_id="q3",
            question_text="Question 3",
            question_metadata={},
            candidate_response="Answer 3",
            response_time_seconds=75
        )
        db_temp.save_evaluation(conv_id3, {'overall_score': 7.2}, "Good")
        
        # Get trajectory
        trajectory = db_temp.get_performance_trajectory("test_001")
        assert len(trajectory) == 3
        assert trajectory == [7.0, 8.5, 7.2]
    
    def test_agent_traces(self, db_temp):
        """Test agent action logging and retrieval"""
        db_temp.create_interview("test_001")
        
        # Log multiple agent actions
        db_temp.log_agent_action(
            interview_id="test_001",
            agent_name="evaluator",
            action="evaluate",
            input_data={"response": "test answer"},
            output_data={"score": 7.5},
            latency_ms=150,
            success=True
        )
        
        db_temp.log_agent_action(
            interview_id="test_001",
            agent_name="question_selector",
            action="select_question",
            input_data={"difficulty": "medium"},
            output_data={"question_id": "q123"},
            latency_ms=200,
            success=True
        )
        
        # Get all traces
        all_traces = db_temp.get_agent_traces("test_001")
        assert len(all_traces) == 2
        
        # Get filtered traces
        eval_traces = db_temp.get_agent_traces("test_001", agent_name="evaluator")
        assert len(eval_traces) == 1
        assert eval_traces[0]['agent_name'] == "evaluator"
        assert eval_traces[0]['action'] == "evaluate"
        assert eval_traces[0]['input_data']['response'] == "test answer"
    
    def test_transaction_rollback(self, db_temp):
        """Test transaction rollback on error"""
        db_temp.create_interview("test_001")
        
        # Attempt transaction that will fail
        try:
            with db_temp.transaction():
                conv_id = db_temp.save_conversation_turn(
                    interview_id="test_001",
                    turn_number=1,
                    question_id="q1",
                    question_text="Test",
                    question_metadata={},
                    candidate_response="Answer",
                    response_time_seconds=60
                )
                # Force an error
                raise ValueError("Simulated error")
        except ValueError:
            pass
        
        # Verify rollback - conversation should not be saved
        history = db_temp.get_conversation_history("test_001")
        assert len(history) == 0
    
    def test_interview_summary(self, db_temp):
        """Test interview summary generation"""
        db_temp.create_interview("test_001", difficulty_level="hard")
        
        # Add turns with different topics
        conv_id1 = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=1,
            question_id="q1",
            question_text="Question 1",
            question_metadata={"topic": "optimization"},
            candidate_response="Answer 1",
            response_time_seconds=60
        )
        db_temp.save_evaluation(conv_id1, {'overall_score': 8.0}, "Good")
        
        conv_id2 = db_temp.save_conversation_turn(
            interview_id="test_001",
            turn_number=2,
            question_id="q2",
            question_text="Question 2",
            question_metadata={"topic": "deep_learning"},
            candidate_response="Answer 2",
            response_time_seconds=90
        )
        db_temp.save_evaluation(conv_id2, {'overall_score': 7.0}, "Good")
        
        # Complete interview
        db_temp.update_interview_status("test_001", "completed", 7.5)
        
        # Get summary
        summary = db_temp.get_interview_summary("test_001")
        assert summary['total_questions'] == 2
        assert summary['avg_score'] == 7.5
        assert set(summary['topics_covered']) == {"optimization", "deep_learning"}
        assert summary['difficulty_level'] == "hard"
        assert summary['status'] == "completed"