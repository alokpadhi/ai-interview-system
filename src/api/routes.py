import uuid

from fastapi.background import BackgroundTasks
from fastapi import APIRouter, Request, HTTPException
from langchain_core.runnables import RunnableConfig

from src.graph.state import initialize_state
from src.api.models import StartRequest, StartResponse, QuestionInfo
from src.api.session import SessionMeta
from src.api.state import get_app_state
from src.utils.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter()

@router.post("/api/v1/interview/start")
async def start_interview(
    request: StartRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
) -> StartResponse:
    
    user_id = str(uuid.uuid4())
    request.user_id = user_id

    # access app.state via http_request
    app_state = get_app_state(http_request)
    state = initialize_state(
        difficulty=request.difficulty,
        time_budget_minutes=request.time_budget_minutes,
        focus_topics=request.focus_topics
    )

    config = RunnableConfig(
        configurable={"thread_id": state["interview_id"]},
        metadata={"user_id": request.user_id, "turn_number": 0},
        tags=["interview", "start"]
    )

    try:
        result = await app_state.start_graph.ainvoke(state, config=config)
    except Exception as e:
        logger.error(f"Failed to start interview: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview.")

    app_state.session_store[result["interview_id"]] = SessionMeta(
        user_id=request.user_id,
        start_result=result
    )

    # Background prewarming (RAG service runs full CRAG)
    plan = result["interview_plan"]
    remaining_topics = plan["topic_sequence"][1:5]
    if remaining_topics:
        background_tasks.add_task(
            app_state.cache_store.pre_warm_topics_background,
            session_id=result["interview_id"],
            rag_service=app_state.rag_service,
            topics=remaining_topics,
            difficulty=result["difficulty_level"]
        )

    return StartResponse(
        session_id=result["interview_id"],
        question=QuestionInfo(
            text=result["current_question"]["text"],
            topic=result["current_question"]["topic"],
            estimated_time_minutes=result["current_question"]["estimated_time_minutes"]
        ),
        time_budget_minutes=result["time_budget_minutes"],
        target_questions=len(plan["topic_sequence"])
    )


    