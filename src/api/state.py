# src/api/state.py

from dataclasses import dataclass
from typing import cast
from fastapi import Request, FastAPI
from langgraph.graph.state import CompiledStateGraph
from src.rag.cache import InterviewCacheStore
from src.rag.agentic_rag import AgenticRAGService
from src.api.session import SessionMeta

@dataclass
class AppState:
    start_graph: CompiledStateGraph
    interview_graph: CompiledStateGraph
    cache_store: InterviewCacheStore
    rag_service: AgenticRAGService
    available_topics: list[str]
    session_store: dict[str, SessionMeta]

class InterviewApp(FastAPI):
    state: AppState

def get_app_state(request: Request) -> AppState:
    from src.api.main import InterviewApp  # local import — avoids circular
    return cast(InterviewApp, request.app).state