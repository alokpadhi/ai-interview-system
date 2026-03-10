# AI Interview System — Implementation Status (Sessions 1–8)

> **Purpose**: Handoff context for continuing development. Pair with `architecture(v2).md`.

---

## Current State

```
✅ IMPLEMENTED (Sessions 1–8)         │  ⬜ NOT YET BUILT (Sessions 9+)
──────────────────────────────────────│──────────────────────────────────
Config, Logging, LLM Factory          │  SupervisorAgent (OODA)
Embeddings (BGE-base-en-v1.5, 768d)   │  ConversationManager (graph node)
ChromaDB Vector Store (3 collections) │  Main InterviewGraph (LangGraph)
Data ingestion (700Q, 125C, 50S)      │  API layer (FastAPI endpoints)
VectorRetriever (domain-level API)    │  Streaming (SSE)
SQLite DB (5 tables, WAL mode)        │  Resilience patterns (.with_retry,
MemoryService (4-type memory)         │    .with_fallbacks, timeouts)
CRAG subgraph (LangGraph StateGraph)  │
DocumentGrader (hybrid: score + LLM)  │
QueryRefiner (3 strategies)           │
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
78 + Session 6 + Session 7 +          │
  Session 8 tests                     │
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

**`FeedbackOutput` key fields (inter-agent contract in `contracts.py`):**
- `feedback_text: str` — only field exposed to candidate
- `strength_acknowledgment / gap_hint / transition_phrase: str = ""` — empty string not None; FeedbackComposer uses `.format()`, None breaks it
- `structure_template: str` — tracked for variation across turns

**`QuestionOutput` key fields:**
- `question_type: str` — NOT `Literal`; gate owns constraint validation
- `target_concepts: list[str] = []` — dynamic rubric for follow_up/clarify; set at runtime by QS, not a ChromaDB field
- `estimated_time_minutes: float = 5.0` — `# TODO: Session 12 — add_time_metadata.py`
- `parent_question_id / target_misconception: Optional[str] = None`

**`TrendAnalyzer(alpha=0.3)` methods:**
- `calculate_ema(trajectory) → list[float]`
- `get_trend(trajectory) → "improving"|"declining"|"stable"` — needs ≥4 scores
- `should_adjust_difficulty(trajectory) → (bool, str)`
- `get_current_ema(trajectory) → float` — returns `NEUTRAL_EMA=5.0` for empty

**`CircuitBreaker(max_retries=1)` methods:**
- `should_retry(agent_name) → bool`
- `reset(agent_name=None)` — `# TODO: call at top of SupervisorAgent.validate_and_decide (Session 9)`

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
- `found: False` is graceful miss — agent checks before using result
- Pure semantic search via `retrieve_concepts(query=concept_name, n_results=1)` — no category filter
- Tool is stateless — agent owns cache read/write, not the tool

**`FeedbackComposer`**
- `STRUCTURES` dict — 4 templates per band (high/medium/low). Score bands: `>= 8.0` → high, `>= 5.0` → medium, `< 5.0` → low
- `TRANSITIONS` list — 8 entries; `""` removed from list, every-3rd-turn logic handles empty transition exclusively
- `compose(components, score, turn_number, previous_structures) → tuple[str, str]` — returns `(feedback_text, used_template)`

**`FeedbackAgent(fast_llm, cache_store)`**

5-layer pipeline in `execute()`:
1. `_get_concept_context()` — cost-guarded conditional RAG (skips if score >= 7.0 or no missed concepts)
2. `feedback_chain.ainvoke()` — structured generation of `FeedbackComponents`
3. `composer.compose()` — deterministic assembly → `(feedback_text, used_template)`
4. `_check_semantic_repetition()` — fires only if `len(recent_feedbacks) >= 2`
5. `validation_gate.validate()` → circuit breaker → recursive retry → fallback

**`QuestionSelectorAgent(rag_service, fast_llm, complex_llm, cache_store, circuit_breaker)`**

3-mode execution in `execute()`:
- `RETRIEVE`: atomic `select_and_mark()` → CRAG on miss → `select_and_mark()` again → fallback
- `FOLLOW_UP`: 14B generation → `target_concepts = eval_data["key_points_missed"][:2]`
- `CLARIFY`: 14B generation → `target_concepts = [misconception]` + `target_misconception = misconception`

Key methods:
- `_determine_question_mode(state, remaining_time) → str` — pure rule-based, no LLM
- `_retrieve_question(state, remaining_time, config) → tuple[dict, str]` — always injects `topic` explicitly
- `_react_select(candidates, state, config) → dict` — 7B `.with_structured_output(QuestionSelection)`, resolves `selected_id` back to full dict
- `_get_next_topic_from_plan(state) → str` — indexes by `len(topics_covered)`, NOT `question_count`
- `_select_weakest_topic(state) → str` — reads `evaluation["topic"]`, skips `"unknown"` fallback evals
- `_get_performance_trend(state) → str` — reads `ema_trajectory` from state directly (no TrendAnalyzer — Supervisor runs after QS in the DAG)
- `_get_fallback_question() → dict` — UUID-based ID, always valid

**State returns:**
```python
# retrieve mode
{"current_question": q, "question_mode": "retrieve",
 "follow_up_count": 0, "conversation_thread": [q["id"]],
 "topics_covered": [topic]}

# follow_up / clarify mode
{"current_question": q, "question_mode": mode,
 "follow_up_count": count + 1, "conversation_thread": [q["id"]],
 "topics_covered": []}   # operator.add — empty = no new topic
```

---

### Critical Design Decisions (Session 8)

| Decision | Rationale |
|----------|-----------|
| `initialize_concept_lookup()` module-level singleton | Standard pattern for tools needing infrastructure deps — injected once at startup, not passed as tool parameter (would expose to LLM schema) |
| Tool is stateless, agent owns cache | Tools should be pure retrieval — side effects (cache writes) belong in agent layer |
| `FeedbackComponents` local to `feedback.py` | Internal LLM output schema, not an inter-agent contract. Only `FeedbackOutput` (in `contracts.py`) crosses agent boundary |
| `compose()` returns `tuple[str, str]` | `execute()` needs `used_template` for `previous_feedback_structures`. Clean interface — caller unpacks, no separate method needed |
| `_check_semantic_repetition` returns `tuple[str, FeedbackComponents, str]` | All three values are causally linked — returning subset would leave stale references in caller |
| Fallback uses `"fallback"` sentinel for `previous_feedback_structures` | Fallback text doesn't correspond to any real template — recording stale template would corrupt future rotation exclusion window |
| Fallback bypasses validation | Fallbacks are last-resort guaranteed-safe minimums. Validating fallback creates unrecoverable loop |
| `CircuitBreaker` injected into QS, not instantiated | Supervisor owns and resets it; QS only holds a reference — Inversion of Control |
| `_get_performance_trend()` reads `ema_trajectory` from state | TrendAnalyzer cannot be injected into QS — Supervisor (who owns it) runs after QS in the fan-out/fan-in DAG. State is the communication channel between turns |
| `_react_select()` closure for `selector_fn` | Cache defines minimal `list[dict] -> dict` contract; closure captures `state`/`config` without polluting cache interface |
| `topic` explicitly injected on all question dicts | "It comes from somewhere upstream" is not a guarantee — explicit injection at construction point is |
| Module-level chain factories (`_build_followup_chain` etc.) | Independently unit-testable without instantiating agent; mirrors Session 7 pattern |

---

### Bugs Caught (Session 8 — QS)

| Bug | Fix |
|-----|-----|
| Tuple comma in `REACT_SELECTION_PROMPT` human message | Removed comma — adjacent string literals auto-concatenate |
| `.with_retry()` on LLM only, not full chain | Moved to wrap entire chain |
| `_get_remaining_time` vs `_get_remaining_minutes` name mismatch | Unified to `_get_remaining_minutes` |
| Missing `datetime` and `InterviewState` imports | Added |
| `response.content.strip()` on `StrOutputParser` output | Changed to `response.strip()` — StrOutputParser returns `str` not `BaseMessage` |
| `_get_performance_trend` both conditions identical and wrong sign | Fixed to `> 0.8` (improving) and `< -0.8` (declining) |
| `_retrieve_question` missing `remaining_time` parameter | Added to signature |
| Missing `crag_grade=crag_result.grade` in `set_topic_questions` | Added — grade-based TTL was silently defaulting to MEDIUM |
| `uuid` not imported, `id` shadows Python builtin | Added import, renamed to `fallback_id` |
| REACT prompt never instructed model to return `selected_id` | Added explicit instruction to return the `id` field |

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

## What's Next — Session 9: SupervisorAgent + ConversationManager

```
src/agents/supervisor.py        # SupervisorAgent (OODA loop + plan creation)
src/services/conversation.py    # ConversationManager (LangGraph graph node)
```

**SupervisorAgent deliverables:**
- `create_interview_plan(state, config) → dict` — 14B, plan-and-execute at `/start`
- `validate_and_decide(state, config) → dict` — rule-based OODA, 0 LLM calls per turn
- `_observe()`, `_orient()`, `_decide_continuation()`, `_resolve_difficulty()`
- `_get_plan_difficulty_for_next_topic()` — indexes by `len(topics_covered)`, NOT `question_count`
- `CircuitBreaker.reset()` called at top of `validate_and_decide` — deferred from Sessions 7/8
- Owns `question_count` increment — sole owner, incremented post fan-in
- Fallback score exclusion from EMA — filters `is_fallback=True` before calling TrendAnalyzer

**ConversationManager deliverables:**
- `maybe_update_summary(state, config) → dict` — LangGraph node, conditional no-op most turns
- `get_context_for_agent(state) → str` — assembles summary + recent turns for agent prompts
- Sentence-boundary truncation via `_truncate_at_sentence()`
- Trigger: every 3 new turns; keeps last 3 turns full, summarizes older
- Turn counting via `isinstance(m, HumanMessage)` — robust against system messages

**Critical reminders for Session 9:**
- Supervisor decoupled from RAG — plan creation only, QS handles first question retrieval
- `ema_trajectory` uses `last_value` — full recalc, not append
- `performance_trajectory` uses `operator.add` — return only NEW score per turn
- `difficulty_curve` indexed by topic index, follow-ups don't advance the curve
- No early termination — difficulty reduction instead
- ConversationManager runs as graph node AFTER `supervisor_check`, not post-hoc