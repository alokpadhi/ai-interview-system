# AI Interview System — Planning & Session Roadmap

## Project Overview

**What you're building:** Production-grade AI interview system for ML/AI engineering roles

**Your context:**
- Name: Alok
- Experience: 6 years AI/ML engineering
- Goal: Senior-level interview preparation + portfolio showcase
- Timeline: ~5.5 weeks remaining
- Daily availability: 2-3h weekdays, 5-6h weekends

**System purpose:**
- Adaptive ML interview conductor
- Demonstrates: Multi-agent systems, RAG patterns, LangGraph orchestration, production engineering
- Scope: 700 questions, 125 concepts, 50 code solutions
- Stack: Ollama (qwen2.5 14B + 7B), ChromaDB, SQLite, LangGraph ≥1.0.6, LangChain ≥1.2.4

**Related docs:**
- `docs/architecture(v2).md` — Full target design (source of truth)
- `docs/implementation_status.md` — Sessions 1–5 implementation details, file map, test coverage

---

## Current State

**Sessions 1–5 complete** — Infrastructure built, 78 tests passing. See `implementation_status.md` for details.

**Sessions 6–14 remaining** — Agents, graph assembly, API, testing, evaluation, deployment.

```
✅ DONE: Config, Embeddings, ChromaDB, Ingestion, Retriever, DB, Memory, CRAG, Cache
⬜ TODO: Agents (4), Graph, API, Observability, Testing, Eval Framework, Docker
```

**Known data gap**: `estimated_time_minutes` not yet in ingested questions — time-aware filtering inactive.

---

## Session Roadmap (6–14)

> All designs defined in `docs/architecture(v2).md`.
> Tools built alongside the agent that uses them.

### Session 6: InterviewState + Contracts + TrendAnalyzer + Gates (3-4h)

Foundation for all agents — shared types and infrastructure.

```
src/graph/state.py               # InterviewState TypedDict with reducers
src/agents/contracts.py          # EvaluationOutput, FeedbackOutput, QuestionOutput
src/services/trend_analyzer.py   # TrendAnalyzer (EMA α=0.3, thresholds 7.5/5.0)
src/services/validation.py       # ValidationGates, CircuitBreaker
```

**Deliverables:** InterviewState with explicit reducers, `last_value` reducer, Pydantic inter-agent contracts, TrendAnalyzer (EMA math + difficulty adjustment), 3 validation gates, CircuitBreaker, tests.

---

### Session 7: EvaluatorAgent (4-5h)

CoT evaluation with reflection and tool use.

```
src/agents/evaluator.py          # EvaluatorAgent (CoT 14B + Reflection 7B)
src/tools/rubric_tool.py         # @tool rubric_lookup
src/tools/code_validator.py      # @tool code_validator (AST parse)
```

**Deliverables:** `execute(state, config) → dict`, CoT chain with `.with_retry()` + `.with_fallbacks()`, reflection chain (7B), `rubric_lookup` and `code_validator` tools, topic injection, `is_fallback` flag, tests.

---

### Session 8: FeedbackAgent + QuestionSelectorAgent (4-5h)

Two parallel agents — independent, run concurrently in the graph.

```
src/agents/feedback.py           # FeedbackAgent (7B, 4 templates/band)
src/agents/question_selector.py  # QS Agent (3 modes: retrieve/follow_up/clarify)
src/tools/concept_lookup.py      # @tool concept_lookup
```

**Feedback:** 4 templates per score band + rotation, anti-sycophancy, no score leakage, concept cache integration.

**QS:** 3 modes, `_determine_question_mode()`, `MAX_FOLLOW_UPS=2`, `target_concepts` for dynamic rubric, atomic `select_and_mark()`.

---

### Session 9: SupervisorAgent + ConversationManager (3-4h)

Orchestrator brain + memory compression.

```
src/agents/supervisor.py              # OODA loop, rule-based (0 LLM calls/turn)
src/services/conversation_manager.py  # Graph node, summarization
```

**Supervisor:** `create_interview_plan()` (14B, once), `validate_and_decide()` (rule-based), `question_count` ownership, difficulty authority, `configurable_alternatives()` wiring.

**ConversationManager:** Keep 3 recent turns full, summarize older, `HumanMessage`-based counting, sentence-boundary truncation.

---

### Session 10: Graph Assembly + API + Observability (5-6h)

Wire everything into the runnable system.

```
src/graph/interview_graph.py     # build_interview_graph(), build_start_graph()
src/api/main.py                  # FastAPI, lifespan, checkpointer
src/api/routes/interview.py      # /start, /submit_response, /stream, /end
src/monitoring/agent_trace.py    # AgentTrace, error attribution
```

**Graph:** `Evaluator → (Feedback ∥ QS) → Supervisor → MaybeSummarize → END`, `_wrap_with_timeout()`, `interrupt_before` ready.

**API:** Lifespan manager, `AsyncSqliteSaver` checkpointer, `RunnableConfig` propagation, SSE streaming.

**Observability:** LangSmith env vars, `AgentTrace` dataclass, structured logging, latency/cache metrics.

---

### Session 11: Integration Testing + End-to-End (3-4h)

```
tests/integration_tests/test_full_interview.py
tests/integration_tests/test_difficulty_adapt.py
tests/integration_tests/test_followup_loop.py
tests/integration_tests/test_cache_flow.py
tests/integration_tests/test_resilience.py
tests/benchmarks/test_latency.py
```

**Scenarios:** Full interview flow, EMA difficulty adaptation, follow-up/clarify transitions, cache pre-warming, fallback chain recovery, latency targets.

---

### Session 12: Polish + Data Gaps + Final Report (3-4h)

```
scripts/add_time_metadata.py     # estimated_time_minutes for all questions
src/services/report_generator.py # Weighted scoring, per-topic breakdown
```

**Deliverables:** Time metadata ingestion, final report generation, error attribution wiring, README.

---

### Session 13: Evaluation Framework (3-4h)

```
scripts/evaluation/golden_set.json
scripts/evaluation/evaluate_system.py
scripts/evaluation/prompt_regression.py
scripts/evaluation/metrics_report.py
```

**Dimensions:** Evaluator accuracy (MAE), feedback quality (ROUGE), LLM-as-Judge, prompt regression detection, CRAG precision/recall.

---

### Session 14: Docker + Deployment (3-4h)

```
Dockerfile                       # Multi-stage build
docker-compose.yml               # app + Ollama + volumes
.env.production
scripts/deploy.sh
```

**Deliverables:** Containerization, Docker HEALTHCHECK, deployment guide, demo video, portfolio README.

---

## Summary

| Phase | Sessions | Hours | Focus |
|-------|----------|-------|-------|
| Foundation | 1–5 ✅ | ~20h | Infrastructure, RAG, CRAG, Cache |
| Agents | 6–9 | ~15h | InterviewState, 4 agents, tools, contracts |
| Assembly | 10 | ~6h | Graph, API, observability |
| Validation | 11–13 | ~10h | Testing, polish, eval framework |
| Deployment | 14 | ~4h | Docker, deployment |
| **Total** | **14** | **~55h** | |

---

## Resources

- LangGraph: https://langchain-ai.github.io/langgraph/
- LangChain structured output: https://python.langchain.com/docs/how_to/structured_output
- ChromaDB: https://docs.trychroma.com/
- Pydantic: https://docs.pydantic.dev/