# AI Interview System — Implementation Status (Sessions 1–7)

> **Purpose**: Handoff context for continuing development. Pair with `architecture(v2).md`.

---

## Current State

```
✅ IMPLEMENTED (Sessions 1–7)         │  ⬜ NOT YET BUILT (Sessions 8+)
──────────────────────────────────────│──────────────────────────────────
Config, Logging, LLM Factory          │  FeedbackAgent
Embeddings (BGE-base-en-v1.5, 768d)   │  QuestionSelectorAgent
ChromaDB Vector Store (3 collections) │  SupervisorAgent (OODA)
Data ingestion (700Q, 125C, 50S)      │  Main InterviewGraph (LangGraph)
VectorRetriever (domain-level API)    │  ConversationManager (node)
SQLite DB (5 tables, WAL mode)        │  API layer (FastAPI endpoints)
MemoryService (4-type memory)         │  Streaming (SSE)
CRAG subgraph (LangGraph StateGraph)  │  Resilience patterns (.with_retry,
DocumentGrader (hybrid: score + LLM)  │    .with_fallbacks, timeouts)
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
78 + Session 6 + Session 7 tests      │
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

**`FeedbackOutput` key fields:**
- `feedback_text: str` — only field exposed to candidate
- `strength_acknowledgment / gap_hint / transition_phrase: str = ""` — empty string not None; FeedbackComposer uses `.format()`, None breaks it
- `structure_template: str` — tracked for variation across turns

**`QuestionOutput` key fields:**
- `question_type: str` — NOT `Literal`; gate owns constraint validation
- `target_concepts: list[str] = []` — dynamic rubric for follow_up/clarify; set at runtime by QS, not a ChromaDB field
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
2. `question["target_concepts"]` (follow_up / clarify — set by QS at runtime)
3. `[f"Corrects misconception: {question['target_misconception']}"]` (clarify — safety net only)
4. `[]` → drift check silently skipped, debug logged

---

### Critical Design Decisions (Session 6)

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

### Bugs Caught (Session 6)

| Bug | Fix |
|-----|-----|
| `focus_topics` missing from `InterviewState` | Added to immutable metadata group |
| `current_response` wrong field name | Renamed to `candidate_response` |
| `end_reason: Optional[dict]` wrong type | Corrected to `Optional[str]` |

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
- Module-level `_RUBRIC_CACHE` loaded at import time from `Settings.rubric_path` — zero I/O per call
- Returns `{found, criteria, key_points, common_mistakes}` — `key_points` and `common_mistakes` are flat lists aggregated across all criteria
- `found: False` is the graceful miss signal — Evaluator falls back to `target_concepts`
- `_RUBRIC_CACHE` is a plain `dict[str, dict]` — single JSON file, question IDs as keys

**`code_validator(response: str) → dict`**
- Detection is tiered: fenced blocks → Python syntax markers (`self.`, `__init__`, `->`) → structural heuristics (def/import lines, indent-after-colon)
- Returns `{code_detected, is_valid, errors, validation_scope, language}`
- `code_detected: False` → `is_valid: None` (not False) — caller must check `code_detected` before using `is_valid`
- `validation_scope` is always `"syntax_only"`, `language` always `"python"`

**`EvaluatorAgent(complex_llm, fast_llm, consistency_samples=1)`**
- `consistency_samples` sourced from `Settings` (`CONSISTENCY_SAMPLES: int = 1` in `.env`) — prod sets to 2, dev leaves at 1
- `build_eval_chain(complex_llm)` — module-level factory; strict chain (`.with_structured_output(EvaluationOutput)`) + `.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)` + `.with_fallbacks([lenient_chain])`. Lenient chain uses `JsonOutputParser()` — returns plain dict, not `EvaluationOutput`
- `reflect_chain` — `REFLECTION_PROMPT | fast_llm | JsonOutputParser()` — reflection output is already a plain dict when it reaches `_apply_reflection`

**`_build_rubric_context(question) → str`** — async (rubric_lookup is async @tool)

Rubric context priority:
1. `rubric_lookup(question["id"])` → if `found: True`, formats `key_points` + `common_mistakes` and **returns immediately** (target_concepts never bleeds through)
2. `question["target_concepts"]` — dynamic rubric for follow_up / clarify (set by QS at runtime, not a ChromaDB field)
3. `question["target_misconception"]` — safety net for clarify questions; only fires if QS set `target_misconception` but omitted `target_concepts`
4. `"No rubric available..."` — graceful fallback string

**`_response_contains_code(response) → bool`** — sync gate; delegates to `_contains_code` from `code_validator.py` directly — no logic duplication. Called before `code_validator.ainvoke()` to avoid unnecessary async tool overhead on text-only answers.

**`_apply_reflection(eval_dict, reflection) → dict`**

Reflection output shape (from `REFLECTION_PROMPT | fast_llm | JsonOutputParser()`):
```json
{
  "adjustment_needed": bool,
  "reason": str,
  "score_adjustment": -2 to +2,
  "missed_misconceptions": [],
  "additional_key_points_missed": []
}
```
- No-op when `adjustment_needed: False`
- Score clamped to `[0.0, 10.0]`, rounded to 1 decimal
- `missed_misconceptions` merged (appended) into `eval_dict["misconceptions"]`
- `additional_key_points_missed` merged into `eval_dict["key_points_missed"]`
- Non-dict reflection input → warning logged, `eval_dict` returned unchanged
- All three mutation blocks (score, misconceptions, key_points) are independent — no early return after score block

**`_single_evaluate(state, config) → dict`**

Flow: `_build_rubric_context` → `eval_chain.ainvoke` → normalise to plain dict → topic + question_id injection → `_response_contains_code` gate → (if True) `code_validator.ainvoke` → syntax errors appended to `misconceptions` → `reflect_chain.ainvoke` → `_apply_reflection`

- `model_dump()` called immediately after `eval_chain` — both strict (EvaluationOutput) and lenient (plain dict) paths normalised here. Downstream methods always receive plain dict
- `topic` defaults to `"general"` (not `""`) on injection — empty string is the injection failure signal reserved for EvaluationOutput default
- Code syntax errors are appended as `"Code syntax error: {error}"` strings into `misconceptions`

**`execute(state, config) → dict`**

- `asyncio.wait_for()` timeout applied at **graph level**, not inside `execute()` — keeps agent framework-agnostic and unit-testable
- `CircuitBreaker.reset()` deferred to Session 9 (SupervisorAgent)
- Validation gate runs **after** `_apply_reflection` (inside `_single_evaluate`)
- Gate failure → `circuit_breaker.should_retry("evaluator")` → recursive `execute()` call (safe: CB returns True at most once)
- CB exhausted → `gate.get_fallback()` with `topic` injected from `current_question` (defaults to `"unknown"`)
- Self-consistency: `asyncio.gather(*[...], return_exceptions=True)` — single eval failure doesn't abort gather; valid results still used. All exceptions → fallback
- Divergence > 2.0 → `needs_human_review: True`, `consistency_divergence: float` set on eval_dict

---

### Critical Design Decisions (Session 7)

| Decision | Rationale |
|----------|-----------|
| `build_eval_chain()` as module-level factory | Independently unit-testable without instantiating agent |
| Lenient chain returns plain dict | `JsonOutputParser()` output — `execute()` normalises both paths via `isinstance(raw, EvaluationOutput)` check in `_single_evaluate` |
| `_response_contains_code` delegates to `_contains_code` | No logic duplication; sync gate avoids async tool overhead on text-only answers |
| `rubric_lookup` patching via module namespace replacement | `StructuredTool` is a Pydantic model — `patch("...rubric_lookup.ainvoke")` fails. Must replace whole object with `MagicMock(ainvoke=AsyncMock(...))` |
| `EvaluatorAgent.__new__` in tests | `build_eval_chain()` calls `.with_structured_output()` on LLM mock → returns coroutine → LangChain `|` operator rejects it. `__new__` + manual attribute assignment bypasses `__init__` entirely |
| `target_concepts` is runtime-only | Set by QS on follow_up/clarify questions — not a ChromaDB field. `key_concepts` in ChromaDB is a separate retrieval metadata field |
| Clarify questions have both `target_misconception` and `target_concepts` | QS sets both: `target_concepts: [misconception]` (list for rubric), `target_misconception: str` (specific string). Step 3 in `_build_rubric_context` is a safety net only |
| `return_exceptions=True` in self-consistency gather | Single LLM failure doesn't abort all N evals — valid results still usable |
| `consistency_samples` from Settings | Prod/dev separation without code change — `CONSISTENCY_SAMPLES=1` dev, `CONSISTENCY_SAMPLES=2` prod |
| Validation gate after `_apply_reflection` | Gate validates the final adjusted output, not the raw LLM output |

---

### Bugs Caught (Session 7)

| Bug | Fix |
|-----|-----|
| `evaluation_reasoning` vs `"reasoning"` key mismatch in validation gate | Gate checks `evaluation_reasoning` — `"reasoning"` was a typo in architecture doc |
| `patch("...rubric_lookup.ainvoke")` raises `AttributeError: 'StructuredTool' has no attribute 'ainvoke'` | Replace whole tool object in module namespace via `patch("src.agents.evaluator.rubric_lookup", MagicMock(ainvoke=AsyncMock(...)))` |
| `AsyncMock().with_structured_output()` returns coroutine, LangChain `|` rejects it | Use `EvaluatorAgent.__new__` + manual attribute assignment in all test fixtures |

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

## What's Next — Session 8: FeedbackAgent (3-4h)

```
src/agents/feedback.py      # FeedbackAgent + FeedbackComposer
```

**Deliverables:**
- `FeedbackAgent.execute(state, config) → dict` — returns `{current_feedback, previous_feedback_structures, recent_feedbacks}`
- `FeedbackComposer` — 4 templates per score band (high/medium/low), turn-based rotation, `previous_structures` exclusion window of 2
- Structured output: `FEEDBACK_PROMPT | fast_llm.with_structured_output(FeedbackComponents)`
- Anti-sycophancy: `FeedbackValidationGate` already enforces; tone_guidance passed via prompt
- Conditional concept enrichment: `concept_lookup @tool` → `InterviewCacheStore` concept pool → `simple_explanation` woven into `gap_hint`
- Semantic repetition reflection: 7B checks new feedback against `recent_feedbacks[-2:]`; regenerates with diversity instruction if similar
- `.with_retry(stop_after_attempt=2)` on feedback chain
- `asyncio.wait_for()` at graph level (same pattern as EvaluatorAgent)