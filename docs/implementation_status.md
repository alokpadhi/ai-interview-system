## Session 12 — Streamlit Frontend

### Files Produced

| File | Purpose |
|------|---------|
| `src/frontend/app.py` | Streamlit frontend — setup, active interview, final report screens |

---

### Key Interfaces (Session 12)

**Three-screen flow:**
- `render_setup()` — difficulty, time budget, topic multiselect populated from `/topics` endpoint
- `render_active()` — current question, feedback, progress, history, submit/end
- `render_complete()` — scores, topic breakdown, per-question expandable detail

**`api_get_topics()`**
- Calls `GET /api/v1/interview/topics` on setup load
- Cached in `st.session_state.available_topics` — fetched once per session
- Returns `list[{label, value}]` — label for display, value sent to `/start`
- Graceful fallback to empty list if server unreachable

**`build_invoke_payload` equivalent — `api_submit()`**
- Uses non-streaming `/submit_response` — Streamlit cannot do true real-time token streaming due to full-page rerender model
- `api_submit_stream()` exists but only waits for `turn_complete` event — tokens ignored

**Progress bar**
- Time-driven: `1.0 - (time_remaining / time_budget)` fills as time runs out
- No question target shown — interview is time-driven not question-driven
- `min(1.0, ...)` guard prevents `StreamlitAPIException` when questions exceed plan target

**Final report screen**
- Color-coded scores — green ≥7.0, yellow ≥5.0, red <5.0
- Topic breakdown with progress bar per topic
- Per-question expandable detail — sub-scores, key points covered, key points missed, misconceptions
- Fallback evaluations filtered out — `is_fallback=True` excluded from display
- Internal fields never shown — `evaluation_reasoning`, `consistency_divergence`, `needs_human_review`, `question_id` all hidden

---

### Architectural Decisions (Session 12)

| Decision | Rationale |
|----------|-----------|
| Topic checklist from `/topics` endpoint not free-text input | Free-text caused topic name mismatches with ChromaDB — checklist shows exactly what system knows, eliminates alignment bug at source |
| `GET /api/v1/interview/topics` endpoint added | Single source of truth — frontend always in sync with what's in ChromaDB. Returns `{label, value}` pairs for clean display/value separation |
| Streaming toggle removed | Streamlit rerenders entire page on state change — true token streaming not possible. Toggle was misleading |
| Non-streaming endpoint used by default | Functionally identical result, simpler code path, no token noise |
| `api_submit_stream()` kept but only consumes `turn_complete` | Backend SSE is correctly implemented — limitation is Streamlit, not server. React frontend would consume tokens in real-time |
| Progress bar time-driven not question-driven | Supervisor now uses time as primary driver — question count is a guide not a hard cap. Progress bar must match actual behavior |
| `min(1.0, pct)` on progress bar | `questions_done / target` can exceed 1.0 when follow-ups push count past plan target — Streamlit raises `StreamlitAPIException` without clamp |
| Per-question detail in expanders | Keeps report scannable — top-level shows summary, detail available on demand |
| `available_topics` added to `AppState` | `/topics` endpoint needs it — stored in lifespan same as other infrastructure |

---

### Bugs Found and Fixed (Session 12)

| Bug | Location | Fix |
|-----|----------|-----|
| `StreamlitAPIException: Progress Value [0.0, 1.0]: 1.5` | `app.py` | `min(1.0, pct)` clamp — follow-ups push `question_count` past `target_questions` |
| Raw JSON showing as feedback in UI | `app.py` | Stale `st.session_state` from previous session — clean browser restart fixed it |
| SSE token stream showed all agents' JSON not just feedback | `app.py` | `langgraph_node` metadata filter unreliable with Ollama — switched to ignore tokens, use `turn_complete` only |
| Topic text input caused ChromaDB mismatches | `app.py` + `routes.py` | Replaced with `st.multiselect` populated from `/topics` endpoint |

---

### Known Issues Carried Into Session 13

| Issue | Impact | Resolution |
|-------|--------|------------|
| True SSE token streaming not available in Streamlit | Feedback appears all at once not character by character | React frontend would fix this — documented as future improvement |
| `time_allocation` format from supervisor LLM still wrong | Plan display only — not consumed by agents | Fix supervisor prompt with escaped JSON example |
| Topic name alignment between supervisor plan and ChromaDB still fragile | Pre-warm occasionally targets invalid topics | Strengthen supervisor prompt topic constraint |
| `estimated_time_minutes` still defaulting to 5.0 | Time-aware filtering inactive | `scripts/add_time_metadata.py` |

---

### Smoke Test — Post-Session 12 Validation

```
streamlit run src/frontend/app.py
    ↓
Setup screen — select difficulty, time budget, topics from checklist
    ↓
Interview screen — answer questions, see feedback, track time progress
    ↓
End interview — view final report with per-question breakdown
    ↓
Start new interview — session state fully cleared
```

Validated: topic checklist populated from backend, feedback displays correctly, final report shows color-coded scores and per-question detail, session cleanup confirmed.

---

## Session 11 — FastAPI Layer

### Files Produced

| File | Purpose |
|------|---------|
| `src/api/state.py` | `AppState` dataclass, `InterviewApp(FastAPI)` subclass, `get_app_state()` typed accessor |
| `src/api/session.py` | `SessionMeta` dataclass — bridges stateless HTTP layer and stateful LangGraph checkpointer |
| `src/api/models.py` | `StartRequest`, `StartResponse`, `SubmitRequest`, `SubmitResponse`, `ProgressInfo`, `QuestionInfo`, `FinalReport` |
| `src/api/main.py` | `lifespan` context manager, `InterviewApp` instantiation, router registration |
| `src/api/routes.py` | `/start`, `/submit_response`, `/submit_response/stream`, `/end` endpoints + `build_invoke_payload()` |
| `src/api/report.py` | `generate_final_report()` — builds `FinalReport` from final `InterviewState` |

---

### Key Interfaces (Session 11)

**`SessionMeta` dataclass**
- `user_id: str` — server-generated UUID, never client-provided
- `start_result: dict | None` — full `start_graph` output, needed to seed first checkpoint. Cleared to `None` after first turn
- `turn_number: int = 0` — incremented after each successful `ainvoke`
- `first_turn_done: bool = False` — controls which `ainvoke` payload pattern to use

**`AppState` dataclass**
- `start_graph` — compiled, no checkpointer
- `interview_graph` — compiled with `AsyncSqliteSaver` checkpointer
- `cache_store: InterviewCacheStore`
- `rag_service: AgenticRAGService`
- `session_store: dict[str, SessionMeta]`

**`InterviewApp(FastAPI)`**
- Subclasses `FastAPI` to type `state: AppState`
- Enables full Pylance autocomplete on `app.state` throughout codebase

**`get_app_state(request: Request) -> AppState`**
- Lives in `state.py` — not `main.py` (avoids circular import)
- Uses `cast(InterviewApp, request.app).state`
- Called at top of every route handler via `http_request: Request` injection

**`build_invoke_payload(meta: SessionMeta, response: str) -> dict`**
- Shared helper in `routes.py` — used by both `/submit_response` and SSE endpoint
- First turn: `{**meta.start_result, "candidate_response": response}`
- Subsequent turns: `{"candidate_response": response}`

**Lifespan initialization order:**
1. LLMs (`get_complex_llm`, `get_fast_llm`)
2. Model pre-warming (`asyncio.gather` with `return_exceptions=True`)
3. `VectorStore` + `VectorRetriever`
4. `initialize_concept_lookup(retriever)` — tool singleton
5. `retriever.get_available_topics()` — before `AgentRegistry`
6. RAG infrastructure (`DocumentGrader`, `QueryRefiner`, `get_cache_store()`, `AgenticRAGService`)
7. `AgentRegistry`
8. `async with AsyncSqliteSaver` — graphs compiled inside, `yield` inside
9. `app.state` populated, `_periodic_cleanup` task started

---

### Architectural Decisions (Session 11)

| Decision | Rationale |
|----------|-----------|
| `SessionMeta` not in architecture doc | Emerged from implementation — architecture said "track first_turn_done", implementation revealed need for `user_id`, `turn_number`, `start_result` too |
| `InterviewApp(FastAPI)` subclass | `app.state: AppState` annotation on instance not valid Python — subclassing is correct pattern for typed `state` |
| `get_app_state()` in `state.py` not `main.py` | `main.py` is top of import tree — nothing should import from it. Moving accessor to `state.py` breaks circular dependency |
| `user_id` server-generated | No auth layer means client-provided ID is unverifiable. Server generates UUID, returns in `StartResponse` |
| `/end` is `DELETE` not `POST` | REST semantics — ending an interview destroys a resource |
| SSE uses `aget_state()` after `maybe_summarize` | Per-node `event["data"]["output"]` is partial update only. `aget_state()` returns fully merged accumulated state after all reducers have run |
| `build_invoke_payload()` shared helper | First-turn branching logic identical in both submit endpoints — one place to maintain |
| `generate_final_report()` in `report.py` | Business logic does not belong in route handlers — routes own HTTP concerns only |
| `session_store` is in-memory dict | Single process only. NOTE: use Redis for multi-worker deployments |
| `return_exceptions=True` in pre-warm | Pre-warm failures are non-fatal — interview continues with higher first-question latency |
| `aget_state()` not `ainvoke()` in `/end` | Read-only checkpoint lookup — no need to run graph again to get final state |

---

### Bugs Found and Fixed (Session 11)

| Bug | Location | Fix |
|-----|----------|-----|
| `select_and_mark` inverted `is_reusable()` check | `src/rag/cache.py` | `not entry.is_reusable()` — was rejecting valid HIGH/MEDIUM entries, cache never hit |
| `pre_warm_topics_background` called `rag_service.retrieve` | `src/rag/cache.py` | Changed to `rag_service.retrieve_batch(topic, difficulty, n=5)` |
| `RAGResult.documents` wrong field name | `src/rag/cache.py` | Changed to `RAGResult.candidates` — correct field on `RAGResult` |
| `CacheMetrics.invalidations` was `Field` object | `src/rag/cache.py` | Added `@dataclass` decorator — `Field()` from Pydantic has no effect on plain classes |
| `select_and_mark` called `.id` on dict | `src/rag/cache.py` + `src/agents/question_selector.py` | `selected["id"]` in cache; `_select_fn` converts `RetrievalResult` → dict via `to_question_dict()` before passing to `_react_select` |
| `RetrievalResult` / dict boundary in selector | `src/agents/question_selector.py` | `hasattr` guard in `_select_fn`: `c.to_question_dict() if hasattr(c, "to_question_dict") else c` |
| Circular import `routes.py` → `main.py` | `src/api/routes.py` | Replaced `from src.api.main import app` with `Request` injection + `get_app_state()` |

---

### Known Issues Carried Into Session 12

| Issue | Impact | Resolution |
|-------|--------|------------|
| `time_allocation` format from supervisor LLM still wrong | Plan display only — not consumed by agents | Fix supervisor prompt with escaped JSON example |
| Topic names in plan don't always match ChromaDB topics | Pre-warm targets invalid topics, cache miss on first question | Strengthen supervisor prompt topic constraint |
| Short time budgets (≤10 min) produce 1-topic plans → interview ends after 1 question | Edge case — not a bug, LLM makes reasonable judgment | Add minimum topic count constraint to plan prompt |
| `estimated_time_minutes` still defaulting to 5.0 | Time-aware filtering inactive | Session 12+: `scripts/add_time_metadata.py` |

---

### Smoke Test — Post-Session 11 Validation

```
uvicorn src.api.main:app --reload --port 8000
    ↓
POST /api/v1/interview/start
    ↓
POST /api/v1/interview/submit_response    ← first turn seeds checkpoint
    ↓
POST /api/v1/interview/submit_response    ← subsequent turns via checkpointer
    ↓
DELETE /api/v1/interview/end              ← final report + session cleanup
```

Validated: feedback generated, no score leakage, session cleanup confirmed (second `/end` returns 404).