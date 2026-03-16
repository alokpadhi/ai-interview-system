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
| `src/rag/retriever.py` | `VectorRetriever` — agent-friendly API over VectorStore. `retrieve_concepts(query, category=None, n_results=3)`. Fetches pool of `min(n*4, 20)` from ChromaDB, then `random.sample()` down to `n` for cross-session question diversity |
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
- `_get_next_topic_from_plan()` — soft priority filter: uncovered topics first, cycles back to `topic_sequence[0]` when all covered. Indexes by `len(topics_covered)`, NOT `question_count`
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
| Topic soft priority filter | Hard exclusion prevented topic reappearance after first visit. Uncovered-first + cycle-back allows natural topic revisits while still respecting plan sequence |

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

**`AgentRegistry(complex_llm, fast_llm, rag_service, cache_store, available_topics, consistency_samples=1)`**
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

### Post-Smoke-Test Fixes (Retrieval Layer)

| Fix | Location | Detail |
|-----|----------|--------|
| Topic hard-exclusion → soft priority filter | `src/agents/question_selector.py` `_get_next_topic_from_plan()` | `uncovered = [t for t in topic_sequence if t not in topics_covered]`. `remaining = uncovered if uncovered else topic_sequence`. When all topics covered, cycles back to start of sequence instead of calling `_select_weakest_topic()`. Prevents topics from being permanently exhausted after one visit |
| Cross-session question diversity | `src/rag/retriever.py` `retrieve_questions()` | Fetches pool of `min(n*4, 20)` from ChromaDB in a single query, then `random.sample(results, min(n, len(results)))` before returning. Prevents same top-k question appearing first every session. Single fetch — no latency penalty |

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

---

## Post-Session 10 Bug Fixes and Design Improvements

These changes were made after the smoke test revealed real-world flow and feedback quality issues. All fixes are in finalized state.

### Files Modified

| File | Change Summary |
|------|---------------|
| `src/agents/question_selector.py` | Mode priority reordered; off-topic re-engagement; is_off_topic threshold |
| `src/agents/feedback.py` | Off-topic LLM path; concept-context-as-compass; off-topic detection threshold |
| `src/agents/evaluator.py` | Sub-score normalization; off-topic clarity prompt fix |
| `src/services/validation.py` | No-questions check added to FeedbackValidationGate |
| `scripts/smoke_test_modes.py` | New: 3-session mode smoke test |

---

### 1. Question Selector — Mode Determination Overhaul (`_determine_question_mode`)

**Problem 1 — clarify was structurally unreachable**

The previous order checked `follow_up` before `clarify`. For any response with `score < 7.0 + missed non-empty + misconceptions`, the follow_up condition fired first and clarify was never reached. Since a misconception-containing response typically scores < 7.0 (technical_accuracy has 40% weight), clarify was effectively dead code.

**Fix: clarify now fires before follow_up.**

Rationale: a misconception left unchallenged distorts every subsequent answer. Probing gaps when the candidate holds a wrong belief is counterproductive. Wrong beliefs must be corrected first.

**New mode decision tree:**
```
is_off_topic → follow_up (re-engagement via _generate_reengagement)
misconceptions detected → clarify                        ← priority 1
score < 7 + gaps + follow_ups < MAX → follow_up          ← priority 2
score 7-8 + gaps + follow_ups < 1 → follow_up
otherwise → retrieve
```

**Problem 2 — off-topic always retrieved, abandoning the question**

Off-topic responses previously triggered `retrieve`, advancing to the next topic. A real interviewer rephrases the question instead.

**Fix: off-topic triggers `follow_up` (up to `MAX_FOLLOW_UPS` times), which internally routes to `_generate_reengagement`. After MAX attempts, retrieve fires.**

`topics_covered` returns `[]` in follow_up mode — the topic stays on the uncovered list until the candidate actually addresses it.

**Problem 3 — `is_off_topic` false-positive when evaluator has no rubric**

Previous check: `not bool(covered) or score < 3.0`

When a question has no rubric in the DB, the evaluator leaves `key_points_covered = []` even for a good on-topic response. This incorrectly flagged high-scoring responses as off-topic.

**Fix:** `is_off_topic = score < 3.0 or (not bool(covered) and score < 5.0)`

Score >= 5.0 with empty covered is treated as on-topic (evaluator limitation). Score < 5.0 with empty covered is genuinely off-topic.

---

### 2. Question Selector — Re-engagement Path (`_generate_reengagement`)

New method and prompt (`RE_ENGAGE_PROMPT`) that rephrases the original question when the candidate goes off-topic.

Key constraints:
- Receives only `original_question` and `topic` — no missed concepts (would give away the answer)
- Generates a simpler entry point or different angle on the same question
- `_generate_follow_up` detects off-topic (`not bool(covered) or score < 3.0`) and delegates to `_generate_reengagement`
- Question dict uses `question_type = "follow_up"` so the graph routing works without changes

---

### 3. Feedback Agent — Off-topic Path Redesign

**Problem:** Previous approach used hardcoded rotating strings. They were generic ("your response didn't address the question"), failed the 15-word minimum gate, and felt robotic.

**Problem with LLM-based off-topic feedback:** Giving the LLM the question text caused it to hint at the answer domain even with strict prompt instructions ("indicate the area the question is about" → reveals the answer direction).

**Fix: Separate `OFF_TOPIC_FEEDBACK_PROMPT` that receives ONLY `{candidate_response}` — never the question text.**

The LLM acknowledges what the candidate actually said (specific to their words) and redirects warmly. Because it never sees the question, it physically cannot reveal the answer direction. 15-word fallback guard if LLM output is too short.

**`is_off_topic` threshold in feedback:** Changed from `not covered` to `not covered and score < 5.0`. Same rationale as question selector — evaluator limitation (no rubric) should not trigger off-topic feedback path for high-scoring responses.

---

### 4. Feedback Agent — Concept Context as Compass

**Problem:** `_get_concept_context` returned actual concept examples or misconceptions from the knowledge base (e.g., `"Real-world application: gradient descent updates weights by computing gradients..."`). The LLM used this to construct gap_hint text that was essentially the answer.

**Fix:** Updated `FEEDBACK_PROMPT` instruction: concept_context is used ONLY as an internal directional check — the gap_hint text must come entirely from how the question itself is framed, not from concept_context content. Prompt phrase: "Treat concept_context as a compass, not as material to write from."

---

### 5. Feedback Validation Gate — No-Questions Check

**Problem:** The LLM occasionally generated feedback containing a question (e.g., "Could you elaborate on...?") despite the `ABSOLUTE RULE: Do NOT ask questions` in the prompt. The candidate then saw feedback with a question AND a separate next question, causing confusion about which to answer.

**Fix:** Added `_no_questions(text)` check to `FeedbackValidationGate`:

```python
if "?" in text:
    return Check(passed=False, message="Feedback contains a question mark...")
```

If check fails → retry (circuit breaker). If retry also fails → fallback text (no question mark, always safe).

---

### 6. Evaluator — Sub-score Normalization (`_normalize_subscores`)

**Problem:** The LLM occasionally produced extreme outlier sub-scores like `[0.0, 0.0, 0.0, 8.0]` (high clarity despite zero scores everywhere else). The validation gate rejected these, triggering a retry that produced the same result. Gate: `max - min > MAX_SCORE_VARIANCE (6.0)`.

**Root cause of original fix failing:** The normalization used `median ± MAX_SPREAD` as the window. For `[2.0, 8.0, 8.0, 9.0]`, median=8.0, floor=2.0 — the outlier was exactly at the floor and didn't move. Spread remained 7.0 > 6.0.

**Fix:** Window changed to `median ± MAX_SPREAD/2`. With half=3.0, the 2.0 outlier gets capped to 5.0. Resulting spread = 4.0 ≤ 6.0. Maximum possible spread = 2 × half = MAX_SPREAD. Guaranteed.

**Evaluator prompt fix:** Added explicit rule: "If candidate_response is empty or off-topic, assign 0 across ALL four criteria including clarity. Clarity does not reward linguistic fluency when content does not address the question."

`_normalize_subscores` is called before validation and reflection — validation never fails for this reason.

---

### 7. Mode Smoke Test (`scripts/smoke_test_modes.py`)

New test script that validates all three modes in isolated sessions.

| Session | Mode | Strategy |
|---------|------|----------|
| Session 1 | follow_up | Full pipeline. LLM generates partial answer to actual retrieved question. |
| Session 2 | clarify | Mocked evaluation with explicit `misconceptions` list injected into state. QS and Feedback called directly — no evaluator re-run. Bypasses evaluator reliability problem. |
| Session 3 | retrieve | Full pipeline. LLM generates comprehensive correct answer. |

**Why mocked evaluation for clarify:** Getting the evaluator to reliably detect LLM-generated misconceptions is not tractable. The LLM resists generating clearly-false statements, and the evaluator only flags misconceptions squarely within the question's rubric scope. The mocked approach tests what matters: given `misconceptions` in state, does the mode selector choose clarify and does `_generate_clarification` produce a valid question?

**Dynamic response generation:** Sessions 1 and 3 use LLM-generated responses (`_PARTIAL_PROMPT`, `_COMPREHENSIVE_PROMPT`) keyed to the actual retrieved question text. Hardcoded responses failed because the retrieved question is random within the topic — a response about overfitting scores 0 against a confusion-matrix question.

Usage: `python -m scripts.smoke_test_modes`

---

### Design Decisions Table

| Decision | Rationale |
|----------|-----------|
| Clarify before follow_up | Misconceptions distort reasoning; must be corrected before probing gaps |
| Re-engagement via follow_up mode | No graph changes needed; `topics_covered: []` keeps topic on the uncovered list |
| OFF_TOPIC_FEEDBACK_PROMPT receives no question text | LLM cannot hint at what it doesn't know; prevents answer leakage by construction |
| `is_off_topic` threshold at score < 5.0 | Evaluator returns `covered=[]` when no rubric exists; score is the reliable signal |
| `_normalize_subscores` uses `±MAX_SPREAD/2` | Guarantees spread ≤ MAX_SPREAD. Full ±MAX_SPREAD allows outliers exactly at boundary to remain |
| Clarify smoke test uses mocked evaluation | Evaluator-generated misconceptions are unreliable; test should verify mode logic, not LLM consistency |
| `_no_questions` in validation gate | Prompt instructions cannot guarantee LLM compliance; gate is the enforcement layer |