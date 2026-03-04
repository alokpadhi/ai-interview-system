# AI Interview System — Implementation Status (Sessions 1–6)

> **Purpose**: Handoff context for continuing development. Pair with `architecture(v2).md`.

---

## Current State

```
✅ IMPLEMENTED (Sessions 1–6)         │  ⬜ NOT YET BUILT (Sessions 7+)
──────────────────────────────────────│──────────────────────────────────
Config, Logging, LLM Factory          │  EvaluatorAgent
Embeddings (BGE-base-en-v1.5, 768d)   │  FeedbackAgent
ChromaDB Vector Store (3 collections) │  QuestionSelectorAgent
Data ingestion (700Q, 125C, 50S)      │  SupervisorAgent (OODA)
VectorRetriever (domain-level API)    │  Main InterviewGraph (LangGraph)
SQLite DB (5 tables, WAL mode)        │  ConversationManager (node)
MemoryService (4-type memory)         │  API layer (FastAPI endpoints)
CRAG subgraph (LangGraph StateGraph)  │  Streaming (SSE)
DocumentGrader (hybrid: score + LLM)  │  Resilience patterns (.with_retry,
QueryRefiner (3 strategies)           │    .with_fallbacks, timeouts)
AgenticRAGService (facade)            │
InterviewCacheStore (dual-pool,       │
  per-session locks, atomic select)   │
InterviewState (TypedDict + reducers) │
Inter-agent contracts (Pydantic)      │
TrendAnalyzer (EMA α=0.3)            │
Validation Gates + CircuitBreaker     │
78 + Session 6 tests passing          │
```

---

## Sessions 1–5 — Infrastructure

| File | Purpose |
|------|---------|
| `src/utils/config.py` | Pydantic Settings, `.env` loading |
| `src/utils/llm_factory.py` | `get_complex_llm()` (14B), `get_fast_llm()` (7B). Aliases: `get_complex_llm = get_llm`, `get_fast_llm = get_secondary_llm`. All models typed as `BaseChatModel` |
| `src/data/embeddings.py` | `EmbeddingService` — `BAAI/bge-base-en-v1.5`, 768d, normalized |
| `src/data/vector_store.py` | `VectorStore` — ChromaDB `PersistentClient`, 3 collections: `interview_questions`, `ml_concepts`, `code_solutions` |
| `scripts/ingest_to_chromadb.py` | 3-pillar validation (schema→content→embedding), batch upsert |
| `src/rag/models.py` | `RetrievalResult`, `RetrievalContext`. No ChromaDB types leak to agents |
| `src/rag/retriever.py` | `VectorRetriever` — agent-friendly API over VectorStore |
| `src/data/database.py` | SQLite WAL mode, 5 tables (`interviews`, `conversations`, `evaluations`, `session_state`, `agent_traces`), `busy_timeout=5000ms` |
| `src/services/memory_service.py` | 4-type memory: short-term buffer, episodic (SQLite), semantic (ChromaDB), working (state) |
| `src/rag/agentic_rag.py` | `AgenticRAGService` facade + `build_crag_graph()` LangGraph subgraph. CRAG flow: `retrieve → grade → (refine → retrieve)* → package_results` |
| `src/rag/grader.py` | `DocumentGrader` — hybrid: ≥0.75→HIGH, ≤0.45→LOW, borderline→7B LLM |
| `src/rag/query_refiner.py` | `QueryRefiner` — 3 strategies (LLM refine, topic pivot, simplify) + anti-repeat (cosine >0.85 rejected) |
| `src/rag/cache.py` | `InterviewCacheStore` — dual-pool (Topic: 10/session, Concept: 30/session), per-session `asyncio.Lock`, atomic `select_and_mark()`, grade-based TTLs |

---

## Session 6 — Foundation Layer

### Files Produced

| File | Purpose |
|------|---------|
| `src/graph/state.py` | `InterviewState` TypedDict, `last_value` reducer, `initialize_state()` |
| `src/agents/contracts.py` | `EvaluationOutput`, `FeedbackOutput`, `QuestionOutput`, `SubScore` |
| `src/services/trend_analyzer.py` | `TrendAnalyzer` — EMA smoothing, trend detection, difficulty decisions |
| `src/services/validation.py` | `EvaluatorValidationGate`, `FeedbackValidationGate`, `QuestionSelectorValidationGate`, `ValidationGateRegistry`, `CircuitBreaker`, `Check`, `ValidationResult` |
| `tests/unit_tests/test_trend_analyzer.py` | EMA math, all 6 `should_adjust_difficulty` combinations, edge cases |
| `tests/unit_tests/test_circuit_breaker.py` | Retry budget, per-agent independence, reset behavior |
| `tests/unit_tests/test_validation_gates.py` | Drift detection, sycophancy, score leakage, time budget |

---

### Key Interfaces

**`initialize_state(difficulty, time_budget_minutes, focus_topics) → InterviewState`**
Takes primitives not `StartRequest` — dependency boundary between core pipeline and FastAPI.
Called from API layer as `initialize_state(request.difficulty, ...)`.

**`InterviewState` reducer rules:**
- `operator.add` → agents return ONLY new items (e.g. `{"all_evaluations": [new_eval]}`)
- `last_value` → agents return replacement value directly
- `ema_trajectory` uses `last_value` despite being a list — EMA is full recalc each turn, not append

**`EvaluationOutput` key fields:**
- `overall_score: float` — `ge=0.0, le=10.0`
- `technical_accuracy / completeness / depth / clarity: SubScore` — nested `{score, reasoning}` dict, NOT raw float
- `evaluation_reasoning: str` — gate enforces min 50 chars
- `topic: str = ""` — injected by Evaluator from `current_question["topic"]`. Empty string (not `Optional`) — detectable injection failure signal
- `is_fallback: bool = False` — Supervisor excludes from EMA when True
- `needs_human_review: bool = False` — set when self-consistency divergence > 2.0
- `consistency_divergence: Optional[float] = None`

**`FeedbackOutput` key fields:**
- `feedback_text: str` — only field exposed to candidate
- `strength_acknowledgment / gap_hint / transition_phrase: str = ""` — empty string not None; FeedbackComposer uses `.format()`, None breaks it
- `structure_template: str` — tracked for variation across turns

**`QuestionOutput` key fields:**
- `question_type: str` — NOT `Literal`; gate owns constraint validation
- `target_concepts: list[str] = []` — dynamic rubric for follow_up/clarify
- `estimated_time_minutes: float = 5.0` — `# TODO: Session 12 — add_time_metadata.py`
- `parent_question_id / target_misconception: Optional[str] = None`

**`TrendAnalyzer(alpha=0.3)` methods:**
- `calculate_ema(trajectory) → list[float]` — same length as input, `[]` → `[]`
- `get_trend(trajectory) → "improving"|"declining"|"stable"` — needs ≥4 scores, uses last 4 EMA values
- `should_adjust_difficulty(trajectory) → (bool, str)` — returns one of: `(False,"insufficient_data")`, `(True,"increase")`, `(True,"decrease")`, `(False,"stable")`
- `get_current_ema(trajectory) → float` — returns `NEUTRAL_EMA=5.0` for empty

**`CircuitBreaker(max_retries=1)` methods:**
- `should_retry(agent_name) → bool` — True if under budget, increments count
- `reset(agent_name=None)` — no args clears all agents; `# TODO: call at top of SupervisorAgent.validate_and_decide (Session 9)`

**`ValidationGateRegistry.get(agent_name)` — raises `KeyError` (not None) for unknown agents**

**`EvaluatorValidationGate._extract_key_points(question)` priority:**
1. `question["rubric"]["criteria"]["technical_accuracy"]["key_points"]` (retrieved)
2. `question["target_concepts"]` (follow_up)
3. `[f"Corrects misconception: {question['target_misconception']}"]` (clarify)
4. `[]` → drift check silently skipped, debug logged

---

### Critical Design Decisions

| Decision | Rationale |
|----------|-----------|
| `last_value` written explicitly | Self-documenting policy; `None` guard prevents silent overwrites during fan-in |
| `focus_topics` added to `InterviewState` | Gap in architecture doc — Supervisor reads `state.get("focus_topics", [])` in `create_interview_plan()` |
| `is_valid` as `@property` on `ValidationResult` | Eliminates impossible state of `is_valid=True` with non-empty `failed_checks` |
| `question_type: str` not `Literal` on `QuestionOutput` | Gate owns the constraint; two enforcement points = two places to update |
| `strength_acknowledgment: str = ""` | FeedbackComposer uses `.format(strength=...)` — None raises AttributeError |
| `ema_trajectory` uses `last_value` | Full recalc each turn; `operator.add` would accumulate stale history |
| `SubScore` nested model | Validation gate does `isinstance(val, dict)` check — SubScore is production path |

---

### Bugs Caught During Implementation

| Bug | Fix |
|-----|-----|
| `focus_topics` missing from `InterviewState` | Added to immutable metadata group |
| `current_response` wrong field name | Renamed to `candidate_response` |
| `end_reason: Optional[dict]` wrong type | Corrected to `Optional[str]` |

---

## Configuration

| Setting | Value |
|---------|-------|
| Primary LLM | `qwen2.5:14b-instruct-q5_K_M` (Ollama) |
| Secondary LLM | `qwen2.5:7b-instruct-q5_K_M` (Ollama) |
| Embedding model | `BAAI/bge-base-en-v1.5` (768d, HuggingFace) |
| Vector DB | ChromaDB (persistent, cosine distance) |
| Relational DB | SQLite (WAL mode, 5 connections) |
| Python | 3.11+ |
| LangGraph | ≥1.0.6 |
| LangChain | ≥1.2.4 |

---

## Known Data Gaps

| Gap | Impact | Resolution |
|-----|--------|------------|
| `estimated_time_minutes` not in ingested questions | Time-aware filtering inactive; defaults to `5.0` | Session 12: `scripts/add_time_metadata.py` |

---

## What's Next — Session 7: EvaluatorAgent (4-5h)

```
src/agents/evaluator.py      # EvaluatorAgent (CoT 14B + Reflection 7B)
src/tools/rubric_tool.py     # @tool rubric_lookup
src/tools/code_validator.py  # @tool code_validator (AST parse)
```

**Deliverables:**
- `EvaluatorAgent.execute(state, config) → dict` — returns `{"current_evaluation": eval_dict}`
- CoT chain: `EVAL_COT_PROMPT | complex_llm.with_structured_output(EvaluationOutput)`
- Reflection chain: `REFLECTION_PROMPT | fast_llm`
- Optional self-consistency: configurable N-sample parallel eval, median selection, flags divergence >2.0
- Topic injection: `eval_dict["topic"] = state["current_question"]["topic"]`
- `is_fallback` flag wired to `ValidationGateRegistry` + `CircuitBreaker`
- `.with_retry(stop_after_attempt=2)` + `.with_fallbacks([lenient_parser])` on CoT chain
- `asyncio.wait_for(timeout=15.0)` wrapper
- `rubric_lookup` and `code_validator` `@tool` decorated functions