import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from fastapi import FastAPI, Request
from starlette.datastructures import State

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models import BaseChatModel

from src.utils.llm_factory import get_complex_llm, get_fast_llm
from src.rag.cache import InterviewCacheStore, get_cache_store
from src.data.vector_store import VectorStore
from src.rag.retriever import VectorRetriever
from src.tools.concept_lookup import initialize_concept_lookup
from src.rag.grader import DocumentGrader
from src.rag.query_refiner import QueryRefiner
from src.rag.agentic_rag import AgenticRAGService
from src.graph.interview_graph import build_start_graph, build_interview_graph
from src.graph.agent_registry import AgentRegistry
from src.api.session import SessionMeta
from src.rag.cache import InterviewCacheStore
from src.utils.logging_config import get_logger
from src.utils.config import get_settings
from src.api.routes import router
from src.api.state import InterviewApp

logger = get_logger(__name__)
settings = get_settings()

async def _prewarm_models(complex_llm: BaseChatModel, fast_llm: BaseChatModel):
    await asyncio.gather(
        complex_llm.ainvoke("hi"),
        fast_llm.ainvoke("hi"),
        return_exceptions=True
    )

async def _periodic_cleanup(cache_store: InterviewCacheStore):
    while True:
        await asyncio.sleep(900)
        cleaned = await cache_store.cleanup_abandoned_sessions()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} abandoned session(s)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # LLMs
    complex_llm = get_complex_llm()
    fast_llm = get_fast_llm()

    # Model prewarming
    await _prewarm_models(complex_llm, fast_llm)

    # Vector store + retriever
    vector_store = VectorStore()
    retriever = VectorRetriever(vector_store)

    # Tool singleton
    initialize_concept_lookup(retriever)

    # available topics
    available_topics = retriever.get_available_topics()

    # RAG pipeline
    grader = DocumentGrader(llm=fast_llm)
    refiner = QueryRefiner()
    cache_store = get_cache_store()
    rag_service = AgenticRAGService(
        retriever=retriever,
        grader=grader,
        refiner=refiner,
        cache_store=cache_store
    )

    # Agent Registry
    agents = AgentRegistry(
        complex_llm=complex_llm,
        fast_llm=fast_llm,
        rag_service=rag_service,
        available_topics=available_topics,
        cache_store=cache_store, 
        consistency_samples=settings.consistency_samples,
    )

    async with AsyncSqliteSaver.from_conn_string(
        settings.checkpoint_db) as checkpointer:
        app.state.available_topics = available_topics
        app.state.rag_service = rag_service
        app.state.start_graph = build_start_graph(agents=agents)
        app.state.interview_graph = build_interview_graph(agents=agents,
                                                         checkpointer=checkpointer)
        app.state.agents = agents
        app.state.cache_store = cache_store
        app.state.session_store = {}

        cleanup_task = asyncio.create_task(_periodic_cleanup(
            cache_store
        ))

        yield


        cleanup_task.cancel()

        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

app = InterviewApp(lifespan=lifespan)
app.include_router(router)