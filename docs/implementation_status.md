# AI Interview System — Implementation Status (Sessions 1–5)

> **Purpose**: Handoff context for continuing development. Pair with `architecture(v2).md`.

---

## Current State: Infrastructure Complete, Agents Not Yet Built

```
✅ IMPLEMENTED (Sessions 1–5)         │  ⬜ NOT YET BUILT (Sessions 6+)
──────────────────────────────────────│──────────────────────────────────
Config, Logging, LLM Factory          │  QuestionSelectorAgent
Embeddings (BGE-base-en-v1.5, 768d)   │  EvaluatorAgent
ChromaDB Vector Store (3 collections) │  FeedbackAgent
Data ingestion (700Q, 125C, 50S)      │  SupervisorAgent (OODA)
VectorRetriever (domain-level API)    │  Main InterviewGraph (LangGraph)
SQLite DB (5 tables, WAL mode)        │  ConversationManager (node)
MemoryService (4-type memory)         │  Validation Gates + Circuit Breakers
CRAG subgraph (LangGraph StateGraph)  │  API layer (FastAPI endpoints)
DocumentGrader (hybrid: score + LLM)  │  TrendAnalyzer (EMA smoothing)
QueryRefiner (3 strategies + anti-    │  Streaming (SSE)
  repeat)                             │  Inter-Agent Contracts (Pydantic)
AgenticRAGService (facade)            │  Resilience patterns (.with_retry,
InterviewCacheStore (dual-pool,       │    .with_fallbacks, timeouts)
  per-session locks, atomic select)   │
78 tests passing                      │
```

---

## File Map by Session

### Session 1 — Project Setup & LLM Factory

| File | Purpose |
|------|---------|
| [config.py](file:///home/alokpadhi/ai-interview-system/src/utils/config.py) | Pydantic Settings, `.env` loading, model defaults |
| [logging_config.py](file:///home/alokpadhi/ai-interview-system/src/utils/logging_config.py) | Structured JSON logging, centralized config |
| [llm_factory.py](file:///home/alokpadhi/ai-interview-system/src/utils/llm_factory.py) | `get_complex_llm()` (14B), `get_fast_llm()` (7B), Ollama integration |

**Key decisions**: Dual-model pattern via Ollama — `qwen2.5:14b-instruct-q5_K_M` (complex reasoning) and `qwen2.5:7b-instruct-q5_K_M` (fast tasks). All models typed as `BaseChatModel`.

---

### Session 2 — Embeddings & Vector Store

| File | Purpose |
|------|---------|
| [embeddings.py](file:///home/alokpadhi/ai-interview-system/src/data/embeddings.py) | `EmbeddingService` wrapping `BAAI/bge-base-en-v1.5`, 768 dims, normalized |
| [vector_store.py](file:///home/alokpadhi/ai-interview-system/src/data/vector_store.py) | `VectorStore` — ChromaDB `PersistentClient`, 3 collections |

**Collections**: `interview_questions`, `ml_concepts`, `code_solutions`. Cosine distance, persistent at `data/chromadb/`.

---

### Session 3 — Data Ingestion

| File | Purpose |
|------|---------|
| [ingest_to_chromadb.py](file:///home/alokpadhi/ai-interview-system/scripts/ingest_to_chromadb.py) | 3-pillar validation (schema → content → embedding), batch upsert |

**Ingested**: ~700 questions, ~125 concepts, ~50 code solutions. Pydantic validation, rejection rate <5%.

---

### Session 4 — Retriever & Data Layer

| File | Purpose |
|------|---------|
| [models.py](file:///home/alokpadhi/ai-interview-system/src/rag/models.py) | `RetrievalResult`, `RetrievalContext` — domain models for RAG |
| [retriever.py](file:///home/alokpadhi/ai-interview-system/src/rag/retriever.py) | `VectorRetriever` — agent-friendly API over `VectorStore` |
| [database.py](file:///home/alokpadhi/ai-interview-system/src/data/database.py) | SQLite DB (WAL mode), 5 tables, schema migration |
| [memory_service.py](file:///home/alokpadhi/ai-interview-system/src/services/memory_service.py) | 4-type memory: short-term buffer, episodic (SQLite), semantic (ChromaDB), working (state) |

**DB tables**: `interviews`, `conversations`, `evaluations` (with `evaluation_data` JSON column), `session_state`, `agent_traces`. WAL mode with `busy_timeout=5000ms`.

**Design rule**: No ChromaDB types leak to agents — all results are `List[RetrievalResult]`.

---

### Session 5 — Agentic RAG (CRAG)

| File | Purpose |
|------|---------|
| [agentic_rag.py](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py) | `AgenticRAGService` facade + `build_crag_graph()` LangGraph subgraph + `RAGResult` |
| [grader.py](file:///home/alokpadhi/ai-interview-system/src/rag/grader.py) | `DocumentGrader` — hybrid grading (score fast-path ≥0.75→HIGH, ≤0.45→LOW, borderline→7B LLM) |
| [query_refiner.py](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py) | `QueryRefiner` — 3 strategies (LLM refine, topic pivot, simplify) + anti-repeat (cosine >0.85 rejected) |
| [cache.py](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py) | `InterviewCacheStore` — dual-pool (Topic: 10/session, Concept: 30/session), per-session `asyncio.Lock`, atomic `select_and_mark()`, grade-based TTLs |

**CRAG subgraph flow**: `retrieve → grade → (refine → retrieve)* → package_results`

**Key APIs**:
- `AgenticRAGService.retrieve_with_crag(topic, difficulty, exclude_ids, remaining_time, n, session_id) → RAGResult`
- `AgenticRAGService.retrieve_batch(topic, difficulty, n) → RAGResult` (cache pre-warming)
- `RAGResult` fields: `candidates`, `grade`, `attempts`, `refined_query`, `served_from_cache`, `corrective_applied`, `queries_used`, `latency_ms`, `is_fallback`

---

## Test Coverage

```
tests/
├── unit_tests/
│   ├── test_data_ingestion.py      # 11 tests — schema, content, embedding validation
│   ├── test_embedding_service.py   # 7 tests  — dims, normalization, consistency
│   ├── test_vectorstore.py         # 9 tests  — CRUD, metadata filtering
│   ├── test_retriever.py           # 15 tests — retrieve, filter, exclude, concepts
│   ├── test_database.py            # 9 tests  — CRUD, transactions, WAL, migration
│   └── test_memory_service.py      # 7 tests  — turns, state, buffer, agent traces
├── test_rag/
│   └── test_cache.py               # 17 tests — dual-pool, TTL, LRU, select_and_mark, sessions
└── integration_tests/
    └── test_retrieval_quality.py    # 3 tests  — semantic relevance, embedding norms
```

**Total: 78 passed / 0 failed** (as of Session 5 end)

---

## Configuration

| Setting | Value |
|---------|-------|
| Primary LLM | `qwen2.5:14b-instruct-q5_K_M` (Ollama) |
| Secondary LLM | `qwen2.5:7b-instruct-q5_K_M` (Ollama) |
| Embedding model | `BAAI/bge-base-en-v1.5` (768d, HuggingFace) |
| Vector DB | ChromaDB (persistent, cosine distance) |
| Relational DB | SQLite (WAL mode, 5 connections) |
| Python | 3.11+ with `.venv` |
| LangGraph | ≥1.0.6 |
| LangChain | ≥1.2.4 |

---

## What's Next (Session 6+)

Refer to `architecture(v2).md` for the full design. The immediate next steps:

1. **Agent implementation** — Build `QuestionSelectorAgent`, `EvaluatorAgent`, `FeedbackAgent`, `SupervisorAgent` as LangGraph nodes
2. **Main interview graph** — `Evaluator → (Feedback ∥ QS) → Supervisor → MaybeSummarize → END`
3. **Resilience wiring** — `.with_retry()`, `.with_fallbacks()`, `configurable_alternatives()`, timeouts
4. **Inter-agent contracts** — `EvaluationOutput`, `FeedbackOutput`, `QuestionOutput` Pydantic models
5. **API layer** — FastAPI endpoints (`/start`, `/submit_response`, `/end`), SSE streaming
6. **ConversationManager** — Summarization node, HumanMessage-based turn counting

**Known data gap**: `estimated_time_minutes` metadata is not yet in ingested question data. Time-aware filtering in `retrieve_with_crag()` won't work until this is added.
