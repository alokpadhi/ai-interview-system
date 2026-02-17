"""
Manual test of database operations.
"""

from src.services.memory_service import get_memory_service
import time

def main():
    memory = get_memory_service()
    
    # Create interview
    print("Creating interview...")
    memory.create_interview("test_123", difficulty_level="medium")
    
    # Save turn
    print("Saving turn...")
    question = {
        'id': 'q_001',
        'text': 'What is gradient descent?',
        'difficulty': 'medium',
        'topic': 'optimization'
    }
    conv_id = memory.save_turn(
        interview_id="test_123",
        turn_number=1,
        question=question,
        response="Gradient descent is an optimization algorithm...",
        response_time_seconds=120
    )
    print(f"Saved turn with conversation_id: {conv_id}")
    
    # Save evaluation
    print("Saving evaluation...")
    evaluation = {
        'technical_accuracy': 8.0,
        'completeness': 7.5,
        'depth': 7.0,
        'clarity': 9.0,
        'overall_score': 7.9
    }
    memory.save_evaluation(conv_id, evaluation, "Good explanation!")
    
    # Get history
    print("\nRetrieving history...")
    history = memory.get_interview_history("test_123")
    print(f"History: {len(history)} turns")
    for turn in history:
        print(f"  Turn {turn['turn_number']}: {turn['question_text']}")
        print(f"  Score: {turn['overall_score']}")
    
    # Save state
    print("\nSaving state...")
    state = {'topics_covered': ['optimization'], 'question_count': 1}
    memory.save_state("test_123", state)
    
    # Load state
    loaded = memory.load_state("test_123")
    print(f"Loaded state: {loaded}")
    
    # Complete interview
    print("\nCompleting interview...")
    memory.complete_interview("test_123", overall_score=7.9)
    print("✅ All operations successful!")

if __name__ == "__main__":
    main()