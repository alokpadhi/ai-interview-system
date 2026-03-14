# AI Interview System — Implementation Status (Sessions 1–10)

> **Purpose**: Handoff context for continuing development. Pair with `architecture(v2).md`.

---

## Current State

```
✅ IMPLEMENTED (Sessions 1–10)        │  ⬜ NOT YET BUILT (Sessions 11+)
──────────────────────────────────────│──────────────────────────────────
Config, Logging, LLM Factory          │  API layer (FastAPI endpoints)
Embeddings (BGE-base-en-v1.5, 768d)   │  Streaming (SSE)
ChromaDB Vector Store (3 collections) │  Resilience patterns (.with_retry,
Data ingestion (700Q, 125C, 50S)      │    .with_fallbacks, timeouts)
VectorRetriever (domain-level API)    │  Observability (LangSmith, Prometheus)
SQLite DB (5 tables, WAL mode)        │  Evaluation Framework (e2e + LLM judge)
MemoryService (4-type memory)         │  Docker + Deployment
CRAG subgraph (LangGraph StateGraph)  │  Data gap: estimated_time_minutes
DocumentGrader (hybrid: score + LLM)  │    (scripts/add_time_metadata.py)
QueryRefiner (3 strategies)           │  Final Report (generate_final_report)
AgenticRAGService (facade)            │
InterviewCacheStore (dual-pool,       │
  per-session locks, atomic select)   │
InterviewState (TypedDict + reducers) │
Inter-agent contracts (Pydantic)      │
TrendAnalyzer (EMA α=0.3)            │
Validation Gates + CircuitBreaker     │
EvaluatorAgent (CoT + Reflection      │
  + Self-Consistency)                 │
rubric_tool.py (@tool, JSON file)     │
code_validator.py (@tool, AST parse)  │
concept_lookup.py (@tool, ChromaDB)   │
FeedbackAgent (5-layer pipeline)      │
FeedbackComposer (variation engine)   │
QuestionSelectorAgent (3 modes)       │
ConversationManager (graph node)      │
SupervisorAgent (OODA + plan)         │
AgentRegistry (DI container)          │
InterviewGraph (LangGraph wiring)     │
78 + Session 6 + Session 7 +          │
  Session 8 + Session 9 tests         │
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
| `src/rag/retriever.py` | `VectorRetriever` — agent-friendly API over VectorStore. `retrieve_concepts(query, category=None, n_results=3)` |
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

---

### Key Interfaces (Session 6)

**`initialize_state(difficulty, time_budget_minutes, focus_topics) → InterviewState`**
Takes primitives not `StartRequest` — dependency boundary between core pipeline and FastAPI.

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
- `target_concepts: list[str] = []` — dynamic rubric for follow_up/clarify; set at runtime by QS
- `estimated_time_minutes: float = 5.0` — `# TODO: Session 12 — add_time_metadata.py`
- `parent_question_id / target_misconception: Optional[str] = None`

**`TrendAnalyzer(alpha=0.3)` methods:**
- `calculate_ema(trajectory) → list[float]`
- `get_trend(trajectory) → "improving"|"declining"|"stable"` — needs ≥4 scores
- `should_adjust_difficulty(trajectory) → (bool, str)`
- `get_current_ema(trajectory) → float` — returns `NEUTRAL_EMA=5.0` for empty

**`CircuitBreaker(max_retries=1)` methods:**
- `should_retry(agent_name) → bool`
- `reset(agent_name=None)` — called at top of `SupervisorAgent.validate_and_decide`

---

### Critical Design Decisions (Session 6)

| Decision | Rationale |
|----------|-----------|
| `last_value` written explicitly | Self-documenting policy; `None` guard prevents silent overwrites during fan-in |
| `focus_topics` added to `InterviewState` | Gap in architecture doc — Supervisor reads `state.get("focus_topics", [])` |
| `is_valid` as `@property` on `ValidationResult` | Eliminates impossible state of `is_valid=True` with non-empty `failed_checks` |
| `question_type: str` not `Literal` on `QuestionOutput` | Gate owns the constraint; two enforcement points = two places to update |
| `strength_acknowledgment: str = ""` | FeedbackComposer uses `.format(strength=...)` — None raises AttributeError |
| `ema_trajectory` uses `last_value` | Full recalc each turn; `operator.add` would accumulate stale history |
| `SubScore` nested model | Validation gate does `isinstance(val, dict)` check — SubScore is production path |

---

## Session 7 — EvaluatorAgent

### Files Produced

| File | Purpose |
|------|---------|
| `src/agents/evaluator.py` | `EvaluatorAgent` — CoT (14B) + Reflection (7B) + optional Self-Consistency |
| `src/tools/rubric_tool.py` | `@tool rubric_lookup` — module-level JSON cache, `_format_rubric()` flattens criteria |
| `src/tools/code_validator.py` | `@tool code_validator` — 3-tier detection, `_extract_code()`, AST `_validate_syntax()` |

---

### Key Interfaces (Session 7)

**`rubric_lookup(question_id: str) → dict`**
- Module-level `_RUBRIC_CACHE` loaded at import time — zero I/O per call
- Returns `{found, criteria, key_points, common_mistakes}`
- `found: False` is the graceful miss signal

**`code_validator(response: str) → dict`**
- Returns `{code_detected, is_valid, errors, validation_scope, language}`
- `code_detected: False` → `is_valid: None` (not False)

**`EvaluatorAgent(complex_llm, fast_llm, consistency_samples=1)`**
- `consistency_samples` sourced from `Settings`
- `build_eval_chain()` — module-level factory; strict + lenient fallback chain

**`_apply_reflection(eval_dict, reflection) → dict`**
- Score clamped to `[0.0, 10.0]`
- All three mutation blocks independent — no early return

---

### Critical Design Decisions (Session 7)

| Decision | Rationale |
|----------|-----------|
| `build_eval_chain()` as module-level factory | Independently unit-testable without instantiating agent |
| `rubric_lookup` patching via module namespace replacement | `StructuredTool` Pydantic model blocks `patch("...ainvoke")` |
| `EvaluatorAgent.__new__` in tests | Bypasses `__init__` to avoid LangChain `|` operator rejection of AsyncMock |
| `target_concepts` is runtime-only | Set by QS on follow_up/clarify — not a ChromaDB field |
| `return_exceptions=True` in self-consistency gather | Single LLM failure doesn't abort all N evals |

---

## Session 8 — FeedbackAgent, Tools, QuestionSelectorAgent

### Files Produced

| File | Purpose |
|------|---------|
| `src/tools/concept_lookup.py` | `@tool concept_lookup` — module-level singleton, `initialize_concept_lookup()`, ChromaDB via `VectorRetriever.retrieve_concepts()` |
| `src/agents/feedback.py` | `FeedbackAgent` (5-layer pipeline) + `FeedbackComposer` (variation engine) |
| `src/agents/question_selector.py` | `QuestionSelectorAgent` — 3 modes (retrieve/follow_up/clarify), atomic select, dynamic rubric |

---

### Key Interfaces (Session 8)

**`concept_lookup(concept_name: str) → dict`**
- Module-level `_retriever: Optional[VectorRetriever] = None`
- `initialize_concept_lookup(retriever)` — called once at app startup in lifespan context
- Returns `{found, explanation, simple_explanation, examples, related_concepts}`
- Tool is stateless — agent owns cache read/write, not the tool

**`FeedbackComposer`**
- `STRUCTURES` dict — 4 templates per band (high/medium/low)
- Score bands: `>= 8.0` → high, `>= 5.0` → medium, `< 5.0` → low
- `compose(components, score, turn_number, previous_structures) → tuple[str, str]` — returns `(feedback_text, used_template)`

**`FeedbackAgent(fast_llm, cache_store)`** — 5-layer pipeline:
1. `_get_concept_context()` — cost-guarded conditional RAG (skips if score >= 7.0)
2. `feedback_chain.ainvoke()` — structured generation of `FeedbackComponents`
3. `composer.compose()` — deterministic assembly → `(feedback_text, used_template)`
4. `_check_semantic_repetition()` — fires only if `len(recent_feedbacks) >= 2`
5. `validation_gate.validate()` → circuit breaker → recursive retry → fallback

**`QuestionSelectorAgent(rag_service, fast_llm, complex_llm, cache_store, circuit_breaker)`**
- `_get_performance_trend()` reads `ema_trajectory` from state directly — TrendAnalyzer cannot be injected into QS (Supervisor runs after QS in DAG)
- `_get_next_topic_from_plan()` indexes by `len(topics_covered)`, NOT `question_count`
- `_react_select()` closure for `selector_fn` — captures `state`/`config` without polluting cache interface
- `topic` explicitly injected on all question dicts at construction point

---

### Critical Design Decisions (Session 8)

| Decision | Rationale |
|----------|-----------|
| `initialize_concept_lookup()` module-level singleton | Tools needing infrastructure deps — injected once at startup, not passed as tool parameter |
| Tool is stateless, agent owns cache | Tools should be pure retrieval — side effects belong in agent layer |
| `FeedbackComponents` local to `feedback.py` | Internal LLM output schema, not an inter-agent contract |
| `compose()` returns `tuple[str, str]` | `execute()` needs `used_template` for `previous_feedback_structures` |
| `CircuitBreaker` injected into QS, not instantiated | Supervisor owns and resets it — Inversion of Control |

---

## Session 9 — ConversationManager + SupervisorAgent

### Files Produced

| File | Purpose |
|------|---------|
| `src/services/conversation.py` | `ConversationManager` — LangGraph graph node, rolling window + batch summarization |
| `src/agents/supervisor.py` | `SupervisorAgent` — OODA loop + plan creation, owns `question_count`, EMA authority |

---

### Key Interfaces (Session 9)

**`ConversationManager(complex_llm: BaseChatModel)`**

- `maybe_update_summary(state, config) → dict` — LangGraph node. Returns `{}` (no-op) most turns. Triggers when `unsummarized >= SUMMARIZE_EVERY_N_TURNS`
- `get_context_for_agent(state) → str` — Facade: assembles summary + recent verbatim turns for agent consumption. `def` not `async` — no I/O
- `_create_summary(existing_summary, new_turns, config) → str` — calls `summarize_chain.ainvoke()`
- `_format_for_summary(messages) → str` — lossy, truncates via `_truncate_at_sentence(q=200, a=300)`
- `_format_full(messages) → str` — lossless, joins with `"\n\n"`
- `_truncate_at_sentence(text, max_chars) → str` — `@staticmethod`, sentence-boundary truncation, 50% budget guard

Turn counting: `sum(1 for m in messages if isinstance(m, HumanMessage))` — both in `maybe_update_summary` AND `get_context_for_agent`. Robust against system/tool messages.

Constants (class-level): `MAX_RECENT_TURNS = 3`, `SUMMARIZE_EVERY_N_TURNS = 3`

**`SupervisorAgent(complex_llm: BaseChatModel, trend_analyzer: TrendAnalyzer)`**

- `create_interview_plan(state, config) → dict` — 1 LLM call (14B). Returns `interview_plan`, `difficulty_level` (seeded from `curve[0]`), `original_difficulty`, `difficulty_reduced_due_to_performance=False`, `stage="questioning"`
- `validate_and_decide(state, config) → dict` — 0 LLM calls. Full OODA loop. Sole owner of `question_count` increment
- `_observe(state, ema) → Observation` — pure snapshot, no side effects
- `_orient(obs, trajectory) → Analysis` — calls TrendAnalyzer, derives conclusions
- `_decide_continuation(analysis, state) → tuple[bool, Optional[str]]` — two termination conditions: `time_up`, `completed`
- `_resolve_difficulty(analysis, state) → tuple[str, bool]` — four guard clauses in order: time_pressure → follow_up/clarify → no_adjust → EMA reconciliation
- `_get_plan_difficulty_for_next_topic(state) → str` — indexes by `len(topics_covered)`, NOT `question_count`
- `_harder_of(a, b) → str` / `_easier_of(a, b) → str` — one-liners via `DIFFICULTY_ORDER`
- `_calculate_target_questions(time_budget) → int` — `max(5, min(12, time_budget // 4))`
- `_get_elapsed_minutes(state) → float` — safe, returns `0` if no `interview_start_time`

Module-level: `PlanOutput` Pydantic model, `INTERVIEW_PLAN_PROMPT`, `_build_plan_chain(complex_llm)`, `Observation` dataclass, `Analysis` dataclass, `DIFFICULTY_ORDER`, `DIFFICULTY_FROM_ORDER`, `NEUTRAL_EMA = 5.0`

---

### Critical Design Decisions (Session 9)

| Decision | Rationale |
|----------|-----------|
| `get_context_for_agent` is `def` not `async` | No I/O — marking async speculatively misleads callers and adds coroutine overhead |
| ConversationManager returns `{}` for no-op | Explicit partial update pattern — don't return current values, reducers preserve them |
| Two slices for summarization | `messages[:-(MAX_RECENT_TURNS*2)]` excludes recent; `[-SUMMARIZE_EVERY_N_TURNS*2:]` takes only new batch — avoids re-summarizing already-summarized turns |
| `PlanOutput` Pydantic model with `with_structured_output()` | Consistent with rest of system — every LLM output parsed via Pydantic, never raw JSON |
| `_build_plan_chain()` as module-level factory | Independently unit-testable without instantiating agent — consistent with Session 7 pattern |
| `CircuitBreaker` instantiated in Supervisor, injected into QS | Supervisor owns lifecycle and calls `reset()`. QS is consumer only — Inversion of Control |
| `difficulty_curve` indexed by `len(topics_covered)` | Curve has one entry per topic. Follow-ups don't advance the index — `question_count` would be wrong |
| `ema_trajectory` uses `last_value` | Full recalc each turn — `operator.add` would accumulate stale history |
| `performance_trajectory` returns `[]` for fallback | `operator.add + []` = no change. Fallback scores never corrupt EMA trajectory |
| `question_count` incremented post-OODA | State is immutable during node execution. `_decide_continuation` uses `state["question_count"] + 1` to look ahead at post-increment value |
| `difficulty_reduced_due_to_performance` uses OR | Once reduced, stays reduced for final report — never resets even if candidate recovers |
| Logging at decision points in `validate_and_decide` | Per-turn EMA, difficulty, and continuation decisions are the most critical observability points in the system |
| `new_ema[-1] if new_ema else NEUTRAL_EMA` in logging | Guards against all-fallback sessions where `full_trajectory = []` and EMA returns `[]` |

---

## Session 10 — Graph Wiring

### Files Produced

| File | Purpose |
|------|---------|
| `src/graph/agent_registry.py` | `AgentRegistry` — DI container, wires all agents with correct dependency order |
| `src/graph/interview_graph.py` | `build_start_graph()` + `build_interview_graph()` — compiled LangGraph StateGraphs |

---

### Key Interfaces (Session 10)

**`AgentRegistry(complex_llm, fast_llm, rag_service, cache_store, consistency_samples=1)`**
- Plain class, all agents as public attributes — direct access (`registry.evaluator`, `registry.supervisor` etc.)
- `TrendAnalyzer` instantiated internally — no external deps, caller has no reason to control it
- Instantiation order enforced: `SupervisorAgent` first (creates `CircuitBreaker`) → `QuestionSelectorAgent` (consumes `supervisor.circuit_breaker`)
- No side effects in `__init__` — `initialize_concept_lookup()` called in lifespan, not here
- Single `logger.info` at end of `__init__` — confirms successful wiring

**`_wrap_with_timeout(agent_fn, timeout_seconds=15.0) → Callable`**
- Module-level utility — not a method, not a decorator on agents
- Uses `functools.wraps` — preserves `__name__`, `__qualname__` for LangSmith trace visibility
- Applied at node registration time, not at agent definition time — graph layer owns execution policy

**`build_start_graph(agents: AgentRegistry) → CompiledStateGraph`**
- Linear: `START → create_plan → first_question → END`
- No checkpointer — runs once, output handed to API layer which seeds first `interview_graph` checkpoint
- Both nodes wrapped with `_wrap_with_timeout`

**`build_interview_graph(agents: AgentRegistry, checkpointer: BaseCheckpointSaver) → CompiledStateGraph`**
- Entry: `START → evaluator`
- Fan-out: `evaluator → feedback`, `evaluator → question_selector` (parallel)
- Fan-in: `feedback → supervisor_check`, `question_selector → supervisor_check`
- Linear tail: `supervisor_check → maybe_summarize → END`
- All 5 nodes wrapped with `_wrap_with_timeout`
- `interrupt_before=["supervisor_check"]` commented out but present — activates human-in-the-loop without refactoring
- Assign → log → return pattern: `compiled = graph.compile(...)` then `logger.info(...)` then `return compiled`

---

### Critical Design Decisions (Session 10)

| Decision | Rationale |
|----------|-----------|
| `AgentRegistry` instantiates once in `__init__`, not via methods | Methods would create new instances on every call — registry must be a singleton container, not a factory |
| Raw LLM params not stored as public attributes on registry | Registry's job ends at construction — exposing `complex_llm` leaks implementation details, callers should use agents not raw LLMs |
| `_wrap_with_timeout` at registration, not as decorator on agent | Agent stays clean and independently testable; graph layer owns execution policy — cross-cutting infrastructure concern |
| `functools.wraps` on timeout wrapper | Preserves function identity for LangSmith/LangFuse trace spans — without it every span shows as `"wrapped"` |
| No try/except around `graph.compile()` | Compilation is a startup operation — fail fast. Swallowing the exception returns `None` which explodes later with a confusing error |
| `BaseCheckpointSaver` as type hint for checkpointer param | Dependency inversion — function doesn't care if it's Postgres or SQLite, depends on abstraction |
| `CompiledStateGraph` return type (not `StateGraph`) | Prevents callers from calling `.add_node()` on compiled graph — different interfaces, different contracts |
| Assign → log → return for compiled graphs | Log only fires after successful compilation — if `compile()` raises, log never fires, which is correct |

---

## Smoke Test — Post-Session 10 Validation

### File Produced

| File | Purpose |
|------|---------|
| `scripts/smoke_test.py` | Manual end-to-end validation script. Real LLMs, real ChromaDB, real state flow. Not a unit test — run manually after wiring changes |

### Bugs Found and Fixed During Smoke Test

| Bug | Location | Fix |
|-----|----------|-----|
| ChromaDB multi-field `where` clause | `src/rag/retriever.py` `_build_where_clause()` | Build `filters` list, use `{"$and": [...]}` for 2+ filters, single dict for 1, `None` for 0 |
| `RetrievalResult` item assignment | `src/agents/question_selector.py` `_retrieve_question()` | Replace `.model_dump()` with `.to_question_dict()` — `@property` fields not serialized by Pydantic |
| `to_question_dict()` missing | `src/rag/models.py` `RetrievalResult` | Added explicit serialization method mapping all `@property` fields (`difficulty`, `topic`, `question_type`) alongside declared fields |
| `+2` invalid JSON from reflection LLM | `src/agents/evaluator.py` `_apply_reflection()` | Added to reflection prompt: score_adjustment must be plain integer, no leading `+`. Removed redundant `re.sub` sanitization — structured output already parses |
| `_apply_reflection` received dict, applied regex | `src/agents/evaluator.py` | `reflect_chain` uses `.with_structured_output()` — returns dict, not raw string. Removed `re.sub` call, kept `isinstance(reflection, dict)` guard |
| `question_mode` always `retrieve` on first turn | `src/agents/question_selector.py` `_determine_question_mode()` | Replaced `state["question_count"] == 0` guard with `not state.get("current_evaluation")` — QS runs before Supervisor increments `question_count` |
| `start_graph` state not seeded into checkpointer | `scripts/smoke_test.py` | First `interview_graph.ainvoke()` must pass `{**start_result, "candidate_response": ...}` — subsequent turns pass only `{"candidate_response": ...}` |
| `topic` mismatch — LLM generates free-form topic names | `src/agents/supervisor.py` | Added `available_topics: list[str]` to `SupervisorAgent.__init__` and plan prompt. Prompt explicitly constrains `topic_sequence` to known ChromaDB topics |
| `AgentRegistry` missing `available_topics` | `src/graph/agent_registry.py` | Added `available_topics: list[str]` parameter. Passed through to `SupervisorAgent` |

### Architectural Decisions from Smoke Test

| Decision | Rationale |
|----------|-----------|
| `available_topics` computed outside `AgentRegistry` | Configuration injection pattern — registry receives result of infrastructure query, not the infrastructure itself. Testable with hardcoded list |
| `get_available_topics()` on `VectorRetriever` | Distinct topic values fetched once at startup via `collection.get(include=["metadatas"])`. Called in lifespan before `AgentRegistry` instantiation |
| `to_question_dict()` explicit serialization | `model_dump()` silently drops `@property` fields. Named method signals intent — "converts RAG result to interview pipeline question" — and makes boundary conversion explicit |
| `_determine_question_mode` guards on `current_evaluation` not `question_count` | `question_count` is incremented by Supervisor AFTER fan-out. QS runs during fan-out — `question_count` is always 0 on first turn when QS runs |
| Timeout configurable via `default_timeout` param | Production value 15s is calibrated for warm models. Smoke test and dev need higher values (60-120s) for cold start + VRAM swap. Default stays 15s |

### Known Issues (Not Blocking Session 11)

| Issue | Impact | Resolution |
|-------|--------|------------|
| `time_allocation` format wrong — LLM generates summary dict instead of per-topic | Plan display only — not consumed by agents currently | Fix supervisor prompt with concrete example in Session 11 |
| Feedback gate fails on first attempt (~18 word responses) | Adds one retry, ~1-1.5s latency | Lower minimum to 15 words OR strengthen feedback prompt |
| Difficulty for next topic retrieval uses current turn's difficulty | QS retrieves at current difficulty; Supervisor updates difficulty after fan-in | Accepted architectural tradeoff — documented. Difficulty update takes effect on turn after topic transition |
| q5_K_M models cannot fit simultaneously in 12GB VRAM | VRAM swap adds 3-8s latency per model switch | Switch to q4_K_M when download complete — both models fit in ~10.5GB |

### Smoke Test Flow (for Session 11 reference)

```
initialize infrastructure (LLMs, ChromaDB, RAG)
    ↓
initialize_concept_lookup(retriever)          ← module-level singleton
    ↓
available_topics = retriever.get_available_topics()
    ↓
AgentRegistry(... available_topics=available_topics)
    ↓
AsyncSqliteSaver (async context manager)
    ↓
build_start_graph() + build_interview_graph()
    ↓
start_graph.ainvoke(initial_state, config)    ← plan + first question
    ↓
interview_graph.ainvoke(                      ← first turn: seed checkpoint
    {**start_result, "candidate_response": ...}, config
)
    ↓
interview_graph.ainvoke(                      ← subsequent turns
    {"candidate_response": ...}, config       ← checkpointer loads rest
)
```

---

## What's Next — Session 11: FastAPI Layer

```
src/api/main.py        # FastAPI app + lifespan context manager
src/api/routes.py      # /start, /submit_response, /end endpoints
src/api/models.py      # Request/Response Pydantic models
src/api/streaming.py   # SSE streaming for /submit_response/stream
```

**Critical reminders for Session 11:**
- Lifespan owns: checkpointer init, `initialize_concept_lookup()` call, `get_available_topics()` call, `AgentRegistry` instantiation, graph compilation — in that order
- First `/submit_response` must pass `{**start_result, "candidate_response": response}` — not just `{"candidate_response": response}`
- API layer needs to track whether it's the first turn for a session — use session store or checkpointer state check
- `initialize_state()` takes primitives — not `StartRequest` directly
- `RunnableConfig(configurable={"thread_id": session_id})` propagated on every `graph.ainvoke()`
- Background pre-warming via `BackgroundTasks` — not raw `asyncio.create_task`
- Scores hidden from user — `/submit_response` returns feedback + next question, never evaluation internals
- `/end` triggers `cache_store.clear_session()` and returns final report
- Fix `time_allocation` supervisor prompt in this session
- Model pre-warming: ping both LLMs with dummy request inside lifespan before accepting traffic

---

## Configuration

| Setting | Value |
|---------|-------|
| Primary LLM | `qwen2.5:14b-instruct-q5_K_M` (Ollama) |
| Secondary LLM | `qwen2.5:7b-instruct-q5_K_M` (Ollama) |
| Embedding model | `BAAI/bge-base-en-v1.5` (768d, HuggingFace) |
| Vector DB | ChromaDB (persistent, cosine distance) |
| Relational DB | SQLite (WAL mode, 5 connections) |
| Rubric file | `Settings.rubric_path` (JSON, question IDs as keys) |
| Consistency samples | `CONSISTENCY_SAMPLES: int` — 1 (dev), 2 (prod) |
| Python | 3.11+ |
| LangGraph | ≥1.0.6 |
| LangChain | ≥1.2.4 |

---

## Known Data Gaps

| Gap | Impact | Resolution |
|-----|--------|------------|
| `estimated_time_minutes` not in ingested questions | Time-aware filtering inactive; defaults to `5.0` | Session 12: `scripts/add_time_metadata.py` |