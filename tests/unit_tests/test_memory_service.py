"""
Unit tests for MemoryService.
"""

import pytest
from src.services.memory_service import MemoryService
from src.data.database import Database
import tempfile
import shutil


class TestMemoryService:
    """Test suite for MemoryService"""
    
    @pytest.fixture
    def memory_service_temp(self):
        """Temporary memory service for each test"""
        temp_dir = tempfile.mkdtemp()
        db = Database(f"{temp_dir}/test.db")
        service = MemoryService(db)
        yield service
        db.close()
        shutil.rmtree(temp_dir)
    
    def test_interview_lifecycle(self, memory_service_temp):
        """Test full interview lifecycle"""
        # Create
        interview_id = memory_service_temp.create_interview(
            interview_id="test_001",
            difficulty_level="medium"
        )
        
        # Complete
        memory_service_temp.complete_interview(
            interview_id="test_001",
            overall_score=8.5
        )
        
        # Verify
        # State should be deleted after completion
        state = memory_service_temp.load_state("test_001")
        assert state is None
    
    def test_save_and_retrieve_turn(self, memory_service_temp):
        """Test saving and retrieving conversation turn"""
        memory_service_temp.create_interview("test_001")
        
        question = {
            'id': 'q_001',
            'text': 'What is gradient descent?',
            'difficulty': 'medium',
            'topic': 'optimization'
        }
        
        conv_id = memory_service_temp.save_turn(
            interview_id="test_001",
            turn_number=1,
            question=question,
            response="Gradient descent is...",
            response_time_seconds=120
        )
        
        assert conv_id > 0
        
        # Get history
        history = memory_service_temp.get_interview_history("test_001")
        assert len(history) == 1
        assert history[0]['question_text'] == question['text']
    
    def test_conversation_buffer(self, memory_service_temp):
        """Test conversation buffer (short-term memory)"""
        memory_service_temp.create_interview("test_001")
        
        # Add 7 turns
        for i in range(1, 8):
            question = {
                'id': f'q_{i:03d}',
                'text': f'Question {i}',
                'difficulty': 'medium',
                'topic': 'test'
            }
            memory_service_temp.save_turn(
                interview_id="test_001",
                turn_number=i,
                question=question,
                response=f"Answer {i}",
                response_time_seconds=60
            )
        
        # Get last 5 turns
        buffer = memory_service_temp.get_conversation_buffer("test_001", last_n=5)
        assert len(buffer) == 5
        assert buffer[0]['turn_number'] == 3
        assert buffer[-1]['turn_number'] == 7
        
        # Get last 3 turns
        buffer_3 = memory_service_temp.get_conversation_buffer("test_001", last_n=3)
        assert len(buffer_3) == 3
        assert buffer_3[0]['turn_number'] == 5
    
    def test_performance_trajectory(self, memory_service_temp):
        """Test performance trajectory retrieval"""
        memory_service_temp.create_interview("test_001")
        
        scores = [7.0, 8.5, 7.2, 9.0, 8.0]
        
        for i, score in enumerate(scores, 1):
            question = {
                'id': f'q_{i}',
                'text': f'Question {i}',
                'difficulty': 'medium'
            }
            conv_id = memory_service_temp.save_turn(
                interview_id="test_001",
                turn_number=i,
                question=question,
                response=f"Answer {i}",
                response_time_seconds=60
            )
            memory_service_temp.save_evaluation(
                conversation_id=conv_id,
                evaluation={'overall_score': score},
                feedback="Feedback"
            )
        
        trajectory = memory_service_temp.get_performance_trajectory("test_001")
        assert trajectory == scores
    
    def test_agent_logging(self, memory_service_temp):
        """Test agent action logging"""
        memory_service_temp.create_interview("test_001")
        
        # Log agent actions
        memory_service_temp.log_agent_action(
            interview_id="test_001",
            agent_name="evaluator",
            action="evaluate_response",
            input_data={"response": "test answer", "question": "test question"},
            output_data={"score": 8.5, "feedback": "Good answer"},
            latency_ms=250,
            success=True
        )
        
        memory_service_temp.log_agent_action(
            interview_id="test_001",
            agent_name="question_selector",
            action="select_next_question",
            input_data={"difficulty": "medium", "topics_covered": ["optimization"]},
            output_data={"question_id": "q_123", "topic": "deep_learning"},
            latency_ms=180,
            success=True
        )
        
        # Get all traces
        traces = memory_service_temp.get_agent_traces("test_001")
        assert len(traces) == 2
        
        # Get filtered traces
        eval_traces = memory_service_temp.get_agent_traces("test_001", agent_name="evaluator")
        assert len(eval_traces) == 1
        assert eval_traces[0]['action'] == "evaluate_response"
        assert eval_traces[0]['output_data']['score'] == 8.5
    
    def test_state_persistence(self, memory_service_temp):
        """Test state save and load"""
        memory_service_temp.create_interview("test_001")
        
        state = {
            "current_turn": 5,
            "difficulty_level": "hard",
            "topics_covered": ["optimization", "deep_learning"],
            "performance_trend": "improving"
        }
        
        # Save state
        memory_service_temp.save_state("test_001", state)
        
        # Load state
        loaded_state = memory_service_temp.load_state("test_001")
        assert loaded_state == state
        assert loaded_state['current_turn'] == 5
        assert loaded_state['topics_covered'] == ["optimization", "deep_learning"]
    
    def test_interview_summary(self, memory_service_temp):
        """Test interview summary generation"""
        memory_service_temp.create_interview("test_001", difficulty_level="medium")
        
        # Add turns
        for i in range(1, 4):
            question = {
                'id': f'q_{i}',
                'text': f'Question {i}',
                'topic': 'optimization' if i % 2 == 0 else 'deep_learning',
                'difficulty': 'medium'
            }
            conv_id = memory_service_temp.save_turn(
                interview_id="test_001",
                turn_number=i,
                question=question,
                response=f"Answer {i}",
                response_time_seconds=60
            )
            memory_service_temp.save_evaluation(
                conversation_id=conv_id,
                evaluation={'overall_score': 7.5 + i * 0.5},
                feedback="Good"
            )
        
        # Complete interview
        memory_service_temp.complete_interview("test_001", overall_score=8.5)
        
        # Get summary
        summary = memory_service_temp.get_interview_summary("test_001")
        assert summary['total_questions'] == 3
        assert summary['avg_score'] == 8.5
        assert summary['difficulty_level'] == "medium"
        assert summary['status'] == "completed"
        assert 'duration_minutes' in summary or 'duration_seconds' in summary