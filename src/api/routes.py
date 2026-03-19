import uuid
from datetime import datetime
import json

from fastapi.background import BackgroundTasks
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from src.api.report import generate_final_report
from src.graph.state import initialize_state
from src.api.models import StartRequest, StartResponse, ProgressInfo,\
    QuestionInfo, SubmitRequest, SubmitResponse, FinalReport
from src.api.session import SessionMeta
from src.api.state import get_app_state
from src.utils.logging_config import get_logger


logger = get_logger(__name__)
router = APIRouter()


def build_invoke_payload(meta: SessionMeta, response: str) -> dict:
    if not meta.first_turn_done:
        return {**meta.start_result, "candidate_response": response}
    return {"candidate_response": response}

@router.post("/api/v1/interview/start")
async def start_interview(
    request: StartRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
) -> StartResponse:
    
    user_id = str(uuid.uuid4())

    # access app.state via http_request
    app_state = get_app_state(http_request)
    state = initialize_state(
        difficulty=request.difficulty,
        time_budget_minutes=request.time_budget_minutes,
        focus_topics=request.focus_topics
    )

    config = RunnableConfig(
        configurable={"thread_id": state["interview_id"]},
        metadata={"user_id": user_id, "turn_number": 0},
        tags=["interview", "start"]
    )

    try:
        result = await app_state.start_graph.ainvoke(state, config=config)
    except Exception as e:
        logger.error(f"Failed to start interview: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview.")

    app_state.session_store[result["interview_id"]] = SessionMeta(
        user_id=user_id,
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
        user_id=user_id,
        session_id=result["interview_id"],
        question=QuestionInfo(
            text=result["current_question"]["text"],
            topic=result["current_question"].get("topic", "General"),
            estimated_time_minutes=result["current_question"].get("estimated_time_minutes", 5.0)
        ),
        time_budget_minutes=result["time_budget_minutes"],
        target_questions=len(plan["topic_sequence"])
    )

@router.post("/api/v1/interview/submit_response")
async def submit_response(request: SubmitRequest,
                          http_request: Request) -> SubmitResponse:
    """
    Non-streaming endpoint. Checkpointer handles state persistence
    automatically via thread_id — no manual save_state needed.
    """
    app_state = get_app_state(http_request)
    session_id: str = request.session_id

    session_meta = app_state.session_store.get(session_id)
    if session_meta is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    config = RunnableConfig(
        configurable={"thread_id": session_id},
        metadata={"user_id": session_meta.user_id, "session_id": request.session_id},
        tags=["interview", "submit"],
    )

    payload = build_invoke_payload(session_meta,response=request.response)

    try:
        result = await app_state.interview_graph.ainvoke(payload, config=config)
    except Exception as e:
        logger.error(f"Failed to process response for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process response")    
    
    session_meta.first_turn_done = True
    session_meta.turn_number += 1
    session_meta.start_result = None

    elapsed = (datetime.now() - result["interview_start_time"]).total_seconds() / 60
    remaining = result["time_budget_minutes"] - elapsed

    return SubmitResponse(
        feedback=result["current_feedback"],
        next_question=QuestionInfo(
            text=result["current_question"]["text"],
            topic=result["current_question"].get("topic", "General"),
            estimated_time_minutes=result["current_question"].get("estimated_time_minutes", 5.0),
        ) if result["should_continue"] else None,
        progress=ProgressInfo(
            questions_completed=result["question_count"],
            time_elapsed_minutes=round(elapsed, 1),
            time_remaining_minutes=round(max(0, remaining), 2)
        ),
        continue_interview=result["should_continue"]
    )

@router.post("/api/v1/interview/submit_response/stream")
async def submit_response_stream(request: SubmitRequest,
                                  http_request: Request) -> StreamingResponse:
    """
    SSE streaming endpoint. Streams feedback tokens as they're generated,
    then sends final structured response.
    """
    app_state = get_app_state(request=http_request)
    session_id: str = request.session_id
    session_meta = app_state.session_store.get(session_id)

    if session_meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    config = RunnableConfig(
        configurable={"thread_id": session_id},
        metadata={"user_id": session_meta.user_id, "session_id": session_id},
        tags=["interview", "submit", "stream"],
    )

    payload = build_invoke_payload(session_meta, request.response)

    async def event_generator():
        try:
            async for event in app_state.interview_graph.astream_events(
                payload,
                config=config,
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    node = event.get("metadata", {}).get("langgraph_node", "")
                    if node == "feedback":
                        chunk = event["data"]["chunk"]
                        response = json.dumps(
                                {'type': 'token',
                                'content': chunk.content}
                            )
                        if chunk.content:
                            yield f"data: {response}\n\n"
                elif event["event"] == "on_chain_end" and event["name"] == "feedback":
                    response = json.dumps(
                        {'type': 'feedback_complete'}
                    )
                    yield f"data: {response}\n\n"

                elif event["event"] == "on_chain_end" and event["name"] == "maybe_summarize":
                    snapshot = await app_state.interview_graph.aget_state(config)
                    full_result = snapshot.values
                    
                    session_meta.turn_number += 1
                    session_meta.first_turn_done = True
                    session_meta.start_result = None
                    
                    elapsed = (datetime.now() - full_result["interview_start_time"]).total_seconds() / 60
                    remaining = full_result["time_budget_minutes"] - elapsed

                    response = json.dumps(
                        {
                            'type': 'turn_complete',
                            'data': SubmitResponse(
                                feedback=full_result["current_feedback"],
                                next_question=QuestionInfo(
                                text=full_result["current_question"]["text"],
                                topic=full_result["current_question"].get("topic", "General"),
                                estimated_time_minutes=full_result["current_question"].get("estimated_time_minutes", 5.0),
                                ) if full_result["should_continue"] else None,
                                progress=ProgressInfo(
                                questions_completed=full_result["question_count"],
                                time_elapsed_minutes=round(elapsed, 1),
                                time_remaining_minutes=round(max(0, remaining), 2)),
                                continue_interview=full_result["should_continue"]
                                            ).model_dump()
                        }
                    )
                    yield f"data: {response}\n\n"
        except Exception as e:
            logger.error(f"Stream failed for session {session_id}: {e}")
            error_payload = json.dumps({"type": "error", "detail": "Stream failed"})
            yield f"data: {error_payload}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.delete("/api/v1/interview/end")
async def end_interview(session_id: str,
                        http_request: Request) -> FinalReport:
    app_state = get_app_state(request=http_request)

    session_meta = app_state.session_store.get(session_id)
    if session_meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    config = RunnableConfig(configurable={"thread_id": session_id})
    snapshot = await app_state.interview_graph.aget_state(config)

    state = snapshot.values

    report = generate_final_report(state)

    await app_state.cache_store.clear_session(session_id)
    del app_state.session_store[session_id]

    return report

@router.get("/api/v1/interview/topics")
async def get_topics(http_request: Request) -> dict:
    app_state = get_app_state(http_request)
    topics = [
        {
            "label": topic.replace("_", " ").title(),
            "value": topic
        }
        for topic in app_state.available_topics
    ]
    return {"topics": topics}
