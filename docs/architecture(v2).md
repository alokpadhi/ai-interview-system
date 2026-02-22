# AI Interview System - Final Architecture (v2)

## Design Principles

1. **Natural Interview Flow**: Hide internal scoring from user; no early termination; adapt difficulty instead
2. **Latency Optimization**: Dual-model strategy + parallel execution + background pre-warming + topic-aware caching
3. **Production-Grade**: Validation gates, circuit breakers, EMA-smoothed trends, drift detection
4. **Time-Bounded**: Real interviews have time limits; system respects them
5. **Consistent UX**: Varied feedback structures; no Mad Libs repetition
6. **LangChain-Native**: Use `BaseChatModel`, `ChatPromptTemplate`, `with_structured_output()`, `RunnableConfig`, and LangGraph state reducers throughout — no hand-rolled alternatives
7. **Resilient by Default**: `.with_retry()` on all chains, `.with_fallbacks()` for structured output, `configurable_alternatives` for model routing, `asyncio.wait_for()` timeouts on agent nodes

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  POST /start          → Plan + First Q (background pre-warms rest)   │   │
│  │  POST /submit_response → Feedback + next question (BUNDLED)          │   │
│  │  POST /submit_response/stream → SSE streaming variant                │   │
│  │                         Scores HIDDEN from user                      │   │
│  │  POST /end            → Final report WITH weighted scores            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Background Task Management:                                                │
│  • Pre-warming uses FastAPI BackgroundTasks (not raw asyncio.create_task)   │
│  • Tied to ASGI server lifecycle via lifespan context manager               │
│  • API layer triggers background work; agents remain framework-agnostic    │
│  • RunnableConfig with thread_id propagated on every graph invocation      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DUAL MODEL CONFIGURATION                             │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │   Qwen2.5-14B (5-bit)           │  │   Qwen2.5-7B (5-bit)            │  │
│  │   "Complex Model"               │  │   "Fast Model"                  │  │
│  │   Type: BaseChatModel            │  │   Type: BaseChatModel            │  │
│  │                                 │  │                                 │  │
│  │   Used for:                     │  │   Used for:                     │  │
│  │   • Interview Plan (start)      │  │   • CRAG grading (DocumentGrader)│  │
│  │   • CoT Evaluation              │  │   • ReAct selection             │  │
│  │   • Follow-up generation        │  │   • Reflection steps            │  │
│  │   • Clarification generation    │  │   • Feedback generation         │  │
│  │   • Conversation summary        │  │                                 │  │
│  │     (every 3 turns)             │  │                                 │  │
│  │                                 │  │                                 │  │
│  │   Latency: ~2-3s per call       │  │   Latency: ~1-1.5s per call     │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                              │
│  All LLM calls use:                                                         │
│  • BaseChatModel (not BaseLLM) — .ainvoke() with message lists             │
│  • ChatPromptTemplate — composable with | pipe operator                    │
│  • .with_structured_output(PydanticModel) for typed responses              │
│  • .with_retry(stop_after_attempt=2, wait_exponential_jitter=True)         │
│  • .with_fallbacks([lenient_parser]) on critical chains (Evaluator, QS)    │
│  • RunnableConfig propagation for tracing                                  │
│  • asyncio.wait_for(timeout=15.0) wrapper on each agent graph node         │
│                                                                              │
│  Graceful Degradation (via configurable_alternatives):                      │
│  • If 7B unavailable → route all tasks to 14B (higher latency, functional) │
│  • If 14B unavailable → 7B handles all tasks (lower quality, functional)   │
│  • Health checks on both models at /start; fallback routing per-request    │
│  • Uses ConfigurableField(id="model_tier") for runtime model switching     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER (LangGraph)                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   Supervisor Agent (OODA Loop)                       │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │   │
│  │  │ Observe  │→ │  Orient  │→ │  Decide  │→ │   Act    │            │   │
│  │  │ (State)  │  │(Analyze) │  │ (Route)  │  │(Dispatch)│            │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │   │
│  │                                                                       │   │
│  │  • Plan-and-Execute at /start (1 LLM call - 14B)                     │   │
│  │  • Rule-based routing (0 LLM calls per turn)                         │   │
│  │  • EMA-smoothed trend detection (α=0.3)                              │   │
│  │  • NO early termination - difficulty reduction instead               │   │
│  │  • Owns question_count increment after fan-in                        │   │
│  │                                                                       │   │
│  │  Difficulty Authority:                                                │   │
│  │  • Plan provides initial difficulty_curve (suggestion)               │   │
│  │  • EMA is the SOLE runtime authority for difficulty                  │   │
│  │  • Plan curve seeds the starting difficulty per topic                │   │
│  │  • After 4+ questions, EMA can override in any direction            │   │
│  │  • Indexes difficulty_curve by topic index (len(topics_covered)),    │   │
│  │    NOT by question_count                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Validation Gates                              │   │
│  │                                                                       │   │
│  │  Evaluator Gate:           Feedback Gate:         Question Gate:     │   │
│  │  • Scores in 0-10 range    • Length 20-200 words  • Question present │   │
│  │  • Reasoning >50 chars     • No forbidden phrases • Valid type       │   │
│  │  • Variance <5.0           • No sycophancy at     • Estimated time   │   │
│  │  • Required fields           low scores             fits budget      │   │
│  │  • Key-point coverage      • No score leakage                        │   │
│  │    alignment (drift check)                                           │   │
│  │  • Dynamic rubric support                                            │   │
│  │    for follow-up/clarify Qs                                          │   │
│  │                                                                       │   │
│  │  Circuit Breaker: Max 1 retry per agent │ Fallback on failure        │   │
│  │  Fallback scores are flagged and EXCLUDED from EMA trajectory        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              PARALLEL EXECUTION (Fan-out / Fan-in)                   │   │
│  │                                                                       │   │
│  │                      ┌──────────────┐                                │   │
│  │                      │   Evaluator  │                                │   │
│  │                      │    (14B)     │                                │   │
│  │                      └──────┬───────┘                                │   │
│  │                             │                                         │   │
│  │               ┌─────────────┴─────────────┐                          │   │
│  │               │       FAN-OUT             │                          │   │
│  │               ▼                           ▼                          │   │
│  │      ┌──────────────┐           ┌──────────────┐                    │   │
│  │      │   Feedback   │           │   Question   │                    │   │
│  │      │    Agent     │           │   Selector   │                    │   │
│  │      │    (7B)      │           │   (7B/14B)   │                    │   │
│  │      └──────┬───────┘           └──────┬───────┘                    │   │
│  │             │                          │                             │   │
│  │             │  KEY ISOLATION:          │                             │   │
│  │             │  • Feedback owns:        │  • QS owns:                │   │
│  │             │    current_feedback      │    current_question        │   │
│  │             │    previous_feedback_    │    question_mode           │   │
│  │             │    structures            │    follow_up_count         │   │
│  │             │                          │    conversation_thread     │   │
│  │             │                          │    topics_covered          │   │
│  │             │                          │                             │   │
│  │             └──────────┬───────────────┘                             │   │
│  │                        │  FAN-IN                                     │   │
│  │                        ▼                                             │   │
│  │               ┌──────────────┐                                       │   │
│  │               │  Supervisor  │  • Owns: stage, should_continue,     │   │
│  │               │    Check     │    difficulty_level, performance_*,  │   │
│  │               │ (Rule-based) │    all_evaluations, end_reason,     │   │
│  │               │              │    question_count (INCREMENTS HERE) │   │
│  │               └──────┬───────┘                                       │   │
│  │                      ▼                                               │   │
│  │               ┌──────────────┐                                       │   │
│  │               │   Maybe      │  • Conditional no-op most turns      │   │
│  │               │  Summarize   │  • Runs every 3 turns (14B)          │   │
│  │               │   (14B)      │  • Checkpointed like all other nodes │   │
│  │               └──────────────┘                                       │   │
│  │                                                                       │   │
│  │  Latency: Evaluator(2-3s) + max(Feedback, QS)(1.5-2s) = 3.5-5s      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WORKER AGENTS LAYER                               │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Evaluator Agent                              │   │
│  │                                                                       │   │
│  │  Responsibility:           Patterns:            Model Assignment:    │   │
│  │  • Score answers           • Chain-of-Thought   • CoT Eval: 14B     │   │
│  │  • Identify knowledge gaps • Reflection         • Reflection: 7B    │   │
│  │  • Detect misconceptions                                             │   │
│  │  • Track key-point coverage                                          │   │
│  │  • Inject topic from                                                 │   │
│  │    current_question into                                             │   │
│  │    evaluation output                                                 │   │
│  │                                                                       │   │
│  │  Rubric Handling:          Tools:               Output (INTERNAL):   │   │
│  │  • Static (DB lookup)      • rubric_checker     • overall_score      │   │
│  │    via @tool decorator       (LangChain @tool)  • key_points_missed  │   │
│  │  • Dynamic (for follow-ups)• code_validator     • key_points_covered │   │
│  │    QS generates target_      (LangChain @tool)  • misconceptions     │   │
│  │    concepts alongside Q                         • reasoning          │   │
│  │                                                 • is_fallback        │   │
│  │  Drift Prevention:                              • topic (injected    │   │
│  │  • Explicit key-point                             from current_      │   │
│  │    checklist in CoT                               question)          │   │
│  │  • Coverage-score                                                    │   │
│  │    alignment validation                                              │   │
│  │  • Dynamic rubric path for                                           │   │
│  │    follow-up/clarify Qs                                              │   │
│  │                                                                       │   │
│  │  Structured Output:        LLM Calls: 2 (CoT + Reflection)          │   │
│  │  • complex_llm.with_      Latency: ~2.5-3.5s                       │   │
│  │    structured_output(                                                │   │
│  │    EvaluationOutput)                                                 │   │
│  │                                                                       │   │
│  │  State Keys Owned:                                                   │   │
│  │  • current_evaluation                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Feedback Agent                                │   │
│  │                                                                       │   │
│  │  Responsibility:           Patterns:            Model Assignment:    │   │
│  │  • Generate constructive   • Structured Output  • Feedback: 7B      │   │
│  │    feedback (no scores!)   • Anti-sycophancy     via .with_          │   │
│  │  • Acknowledge strengths   • Varied structures    structured_output  │   │
│  │  • Hint at gaps naturally    (no Mad Libs)       (FeedbackComponents)│   │
│  │                                                                       │   │
│  │  FeedbackComposer:         Caching:             Does NOT:            │   │
│  │  • Multiple structures     • Concept cache      • Expose scores      │   │
│  │    per score band            (separate pool)    • Generate questions │   │
│  │  • Turn-based variation    • Avoids repeated    • Say "you missed X" │   │
│  │  • Context-aware             concept lookups    • Update stage       │   │
│  │    transitions                                                       │   │
│  │                                                                       │   │
│  │  State Keys Owned:         LLM Calls: 1                              │   │
│  │  • current_feedback        Latency: ~1-1.5s                          │   │
│  │  • previous_feedback_                                                │   │
│  │    structures                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Question Selector Agent                         │   │
│  │                                                                       │   │
│  │  Responsibility:           Operating Modes:     Model Assignment:    │   │
│  │  • ALL question decisions  • RETRIEVE: Cache    • ReAct select: 7B  │   │
│  │  • Mode determination        + RAG service      • Follow-up gen: 14B │   │
│  │  • Time budget enforcement • FOLLOW_UP: LLM gen • Clarify gen: 14B   │   │
│  │  • Topic tracking          • CLARIFY: LLM gen                        │   │
│  │    (owns topics_covered)                                             │   │
│  │  • Topic re-prioritization   Does NOT:                               │   │
│  │    when struggling           • Own CRAG logic                        │   │
│  │  • Generates target_concepts • Grade retrieval                       │   │
│  │    for follow-up/clarify Qs    results                               │   │
│  │    (used as dynamic rubric)                                          │   │
│  │                                                                       │   │
│  │  Caching Strategy:         Time-Aware:          State Keys Owned:    │   │
│  │  • Topic-aware batches     • Passes remaining   • current_question   │   │
│  │  • Separate pool from        time to RAG        • question_mode      │   │
│  │    concept cache             service for         • follow_up_count    │   │
│  │  • Partial reuse tracking    filtering           • conversation_thread│   │
│  │  • TTL set by RAG service                       • topics_covered     │   │
│  │    (real CRAG grade)                                                  │   │
│  │  • Atomic select+mark                                                │   │
│  │    within single lock                                                │   │
│  │                                                                       │   │
│  │  Retrieval Delegation:     LLM Calls: 1 (RETRIEVE) or 2 (GENERATE)   │   │
│  │  • QS calls RAG service's  Latency: ~1.5-2s (cache hit)              │   │
│  │    retrieve_with_crag()             ~2-2.5s (cache miss)             │   │
│  │  • RAG service owns CRAG            ~2.5-3s (generate)               │   │
│  │    loop (grade + refine                                              │   │
│  │    + retry)                                                          │   │
│  │  • QS owns ReAct selection                                           │   │
│  │    from returned candidates                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Per-Turn Summary:                                                          │
│  • Total LLM Calls: 3-4 (+ 1 every 3 turns for summarization)              │
│  • Total Latency: 3.5-5s                                                   │
│  • 14B Calls: 1-2 (heavy lifting)                                          │
│  • 7B Calls: 2-3 (fast decisions)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONVERSATION MANAGER                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │          Rolling Window + Batch Summarization (Graph Node)          │   │
│  │                                                                       │   │
│  │  Runs as a LangGraph node AFTER supervisor_check — checkpointed    │   │
│  │  like every other node. Conditional no-op most turns.               │   │
│  │                                                                       │   │
│  │  Strategy:                                                           │   │
│  │  • Keep last 3 turns in full detail                                  │   │
│  │  • Summarize older turns (batch every 3 new turns)                   │   │
│  │  • Summary contains: topics, strengths, weaknesses (no full Q&A)     │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Turn 1-3: [Summarized]                                      │    │   │
│  │  │  "Covered optimization basics. Strong on gradient descent,  │    │   │
│  │  │   weak on learning rate impact. One misconception corrected."│    │   │
│  │  │                                                               │    │   │
│  │  │  Turn 4: [Full] Q: "What is regularization?" A: "..."        │    │   │
│  │  │  Turn 5: [Full] Q: "How does L2 differ..." A: "..."          │    │   │
│  │  │  Turn 6: [Full] Q: "When would you use..." A: "..."          │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  │  Memory Budget:                                                      │   │
│  │  • Summary: ~200 tokens (fixed)                                      │   │
│  │  • Recent 3 turns: ~1500 tokens (variable)                           │   │
│  │  • Total context: ~1700 tokens (bounded)                             │   │
│  │                                                                       │   │
│  │  Summarization Trigger:                                              │   │
│  │  • After every 3 new turns                                           │   │
│  │  • Uses 14B model for quality                                        │   │
│  │  • Async, runs as graph node (not post-hoc)                          │   │
│  │  • Truncates at sentence boundaries (not char count)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SESSION-ISOLATED CACHE LAYER                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              InterviewCacheStore (Singleton, Separate Pools)         │   │
│  │                                                                       │   │
│  │  ┌───────────────────────────┐  ┌───────────────────────────────┐   │   │
│  │  │   TOPIC QUESTION POOL     │  │      CONCEPT POOL             │   │   │
│  │  │                           │  │                               │   │   │
│  │  │  Max Entries: 10/session  │  │  Max Entries: 30/session      │   │   │
│  │  │  (Topics are high-value)  │  │  (Concepts are smaller)       │   │   │
│  │  │                           │  │                               │   │   │
│  │  │  Key Pattern:             │  │  Key Pattern:                 │   │   │
│  │  │  {topic}:{difficulty}     │  │  {concept_name}               │   │   │
│  │  │                           │  │                               │   │   │
│  │  │  Value:                   │  │  Value:                       │   │   │
│  │  │  • questions: List[dict]  │  │  • explanation: str           │   │   │
│  │  │  • used_ids: Set[str]     │  │  • simple_explanation: str    │   │   │
│  │  │  • crag_grade: str        │  │  • examples: List[str]        │   │   │
│  │  │    (REAL grade from RAG   │  │                               │   │   │
│  │  │     service CRAG loop)    │  │                               │   │   │
│  │  │                           │  │                               │   │   │
│  │  │  LRU: Within pool only    │  │  LRU: Within pool only        │   │   │
│  │  │  (no cross-eviction)      │  │  (no cross-eviction)          │   │   │
│  │  └───────────────────────────┘  └───────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Cache Features:                                                     │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │   │
│  │  │  Composite Key  │ │  Grade-based    │ │  Atomic Select +   │    │   │
│  │  │                 │ │  TTL            │ │  Mark Used          │    │   │
│  │  │  Session +      │ │                 │ │                     │    │   │
│  │  │  Type +         │ │  HIGH: 30 min   │ │  select_and_mark()  │    │   │
│  │  │  Identifier     │ │  MEDIUM: 15 min │ │  single locked op   │    │   │
│  │  │                 │ │  LOW: 5 min     │ │  eliminates TOCTOU  │    │   │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │   │
│  │  │  Pool-Isolated  │ │  Per-Session    │ │  Session Lifecycle  │    │   │
│  │  │  LRU Eviction   │ │  Locks          │ │                     │    │   │
│  │  │                 │ │                 │ │  Abandoned session  │    │   │
│  │  │  Topics: 10 max │ │  asyncio.Lock   │ │  cleanup via        │    │   │
│  │  │  Concepts: 30   │ │  per session    │ │  periodic sweep     │    │   │
│  │  │                 │ │  (not global)   │ │                     │    │   │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC RAG SERVICE                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              CRAG Owner (LangGraph StateGraph Subgraph)               │   │
│  │                                                                       │   │
│  │  AgenticRAGService owns the full CRAG lifecycle via a compiled        │   │
│  │  StateGraph: retrieve → grade → (refine → retrieve) → package        │   │
│  │                                                                       │   │
│  │  LangGraph subgraph — conditional edges, automatic tracing,          │   │
│  │  composable into the main interview graph.                            │   │
│  │                                                                       │   │
│  │  Consumers:                                                          │   │
│  │  • QuestionSelectorAgent: calls retrieve_with_crag() on cache miss  │   │
│  │  • InterviewCacheStore: calls retrieve_batch() for pre-warming      │   │
│  │  Both get CRAG-graded results with accurate grades for TTL.          │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                   CRAG Retrieval Flow                        │   │   │
│  │   │                                                               │   │   │
│  │   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │   │
│  │   │  │  ChromaDB   │    │  Document   │    │  Grade      │      │   │   │
│  │   │  │  Retrieval  │ →  │  Grader     │ →  │  Check      │      │   │   │
│  │   │  │             │    │  (7B)       │    │             │      │   │   │
│  │   │  │ Metadata    │    │             │    │ HIGH: Return │      │   │   │
│  │   │  │ filtering   │    │ Standalone  │    │ MED: Return  │      │   │   │
│  │   │  │ (topic,     │    │ service,    │    │ LOW: Refine  │      │   │   │
│  │   │  │  difficulty,│    │ testable    │    │   + Retry    │      │   │   │
│  │   │  │  time)      │    │ independently│   │             │      │   │   │
│  │   │  └─────────────┘    └─────────────┘    └──────┬──────┘      │   │   │
│  │   │                                                │              │   │   │
│  │   │                                   ┌────────────┴────────────┐│   │   │
│  │   │                                   │   Corrective Actions    ││   │   │
│  │   │                                   │  (on LOW grade)         ││   │   │
│  │   │                                   │                         ││   │   │
│  │   │                                   │  • QueryRefiner:        ││   │   │
│  │   │                                   │    strategy rotation,   ││   │   │
│  │   │                                   │    similarity check     ││   │   │
│  │   │                                   │  • Broaden filters      ││   │   │
│  │   │                                   │    (drop difficulty)    ││   │   │
│  │   │                                   │  • Max 2 retry attempts ││   │   │
│  │   │                                   └─────────────────────────┘│   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │             ReAct Selection (Lives in QS, NOT here)          │   │   │
│  │   │                                                               │   │
│  │   │  RAG service returns graded candidates → QS does ReAct       │   │
│  │   │  selection from the returned batch.                           │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Time-Aware Filtering:                                               │   │
│  │  • estimated_time_minutes <= remaining_time (passed by QS)           │   │
│  │  • Prioritize shorter questions when time is low                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    ChromaDB      │  │     SQLite       │  │   File System    │          │
│  │                  │  │                  │  │                  │          │
│  │  Collections:    │  │  Tables:         │  │  Directories:    │          │
│  │  • interview_    │  │  • interviews    │  │  • rubrics/      │          │
│  │    questions     │  │  • conversations │  │  • prompts/      │          │
│  │  • ml_concepts   │  │  • evaluations   │  │    *.yaml        │          │
│  │  • code_solutions│  │  • session_state │  │  • logs/         │          │
│  │    (Phase 2)     │  │  • agent_traces  │  │                  │          │
│  │                  │  │                  │  │                  │          │
│  │  Embedding:      │  │  Config:         │  │                  │          │
│  │  BGE-base-en-v1.5│  │  • WAL mode      │  │                  │          │
│  │  768 dimensions  │  │  • Pool: 5 conn  │  │                  │          │
│  │  Normalized      │  │  • busy_timeout  │  │                  │          │
│  │  cosine distance │  │    5000ms        │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY LAYER                                  │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │   LangSmith/     │  │    Prometheus    │  │   Structured     │          │
│  │   LangFuse       │  │    Metrics       │  │   Logging        │          │
│  │                  │  │                  │  │                  │          │
│  │  • Agent traces  │  │  • Latency/agent │  │  • agent_traces  │          │
│  │  • LLM calls     │  │  • Token usage   │  │    table         │          │
│  │  • RAG retrieval │  │  • Cache hit rate│  │  • JSON logs     │          │
│  │  • Reasoning     │  │  • Error rates   │  │  • Request IDs   │          │
│  │    chains        │  │  • EMA trends    │  │  • Correlation   │          │
│  │  • Prompt        │  │  • Difficulty    │  │    IDs           │          │
│  │    templates     │  │    adjustments   │  │                  │          │
│  │    (visible via  │  │                  │  │                  │          │
│  │    ChatPrompt    │  │                  │  │                  │          │
│  │    Template)     │  │                  │  │                  │          │
│  │  • RunnableConfig│  │                  │  │                  │          │
│  │    thread_id per │  │                  │  │                  │          │
│  │    session       │  │                  │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Difficulty Authority Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DIFFICULTY LIFECYCLE                                   │
│                                                                          │
│  /start (Plan Creation)                                                  │
│  ┌──────────────────────────────────────────────┐                       │
│  │  Plan generates difficulty_curve per topic:   │                       │
│  │  ["medium", "medium", "hard", "hard", ...]   │                       │
│  │                                               │                       │
│  │  This becomes the SEED for each topic.        │                       │
│  │  state["difficulty_level"] = curve[0]         │                       │
│  │  Pre-warming uses plan difficulty.            │                       │
│  │                                               │                       │
│  │  IMPORTANT: difficulty_curve is indexed by    │                       │
│  │  TOPIC INDEX (len(topics_covered)), NOT by    │                       │
│  │  question_count. Follow-ups don't advance     │                       │
│  │  the curve.                                   │                       │
│  └──────────────────────────────────────────────┘                       │
│                          │                                               │
│                          ▼                                               │
│  Per-Turn (Supervisor OODA)                                             │
│  ┌──────────────────────────────────────────────┐                       │
│  │  On topic transition (new topic from plan):   │                       │
│  │  → effective_difficulty = max(                │                       │
│  │      plan_curve[topic_index],                │                       │
│  │      ema_adjusted_difficulty                 │                       │
│  │    )  ... if EMA says "increase"             │                       │
│  │  → effective_difficulty = min(                │                       │
│  │      plan_curve[topic_index],                │                       │
│  │      ema_adjusted_difficulty                 │                       │
│  │    )  ... if EMA says "decrease"             │                       │
│  │  → effective_difficulty = plan_curve[idx]     │                       │
│  │    ... if EMA has insufficient data (<4 Qs)  │                       │
│  │                                               │                       │
│  │  Within same topic (follow-ups):              │                       │
│  │  → difficulty stays at current level          │                       │
│  └──────────────────────────────────────────────┘                       │
│                          │                                               │
│                          ▼                                               │
│  Pre-warming Impact                                                      │
│  ┌──────────────────────────────────────────────┐                       │
│  │  Pre-warming uses plan difficulty (at /start) │                       │
│  │  RAG service runs full CRAG loop → real grade │                       │
│  │  stored with accurate TTL from the start.     │                       │
│  │                                               │                       │
│  │  If EMA shifts difficulty at runtime:         │                       │
│  │  → Cache miss for new difficulty level        │                       │
│  │  → RAG service CRAG retrieves at correct level│                       │
│  │  → New batch cached at runtime difficulty     │                       │
│  │                                               │                       │
│  │  Accepted tradeoff: occasional cache miss     │                       │
│  │  vs complexity of multi-difficulty pre-warm   │                       │
│  └──────────────────────────────────────────────┘                       │
│                          │                                               │
│                          ▼                                               │
│  Final Report                                                            │
│  ┌──────────────────────────────────────────────┐                       │
│  │  difficulty_history tracks ACTUAL difficulty   │                       │
│  │  used per question (not plan's suggestion).   │                       │
│  │  Weighted scoring uses actual difficulty.     │                       │
│  └──────────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## State Ownership Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STATE KEY OWNERSHIP                                    │
│                                                                          │
│  EVALUATOR AGENT (writes after evaluation):                             │
│  ├── current_evaluation          dict    Score, reasoning, key points   │
│  │   └── .is_fallback            bool    True if circuit breaker fired  │
│  │   └── .topic                  str     Injected from current_question │
│  │                                                                       │
│  FEEDBACK AGENT (writes after feedback generation):                     │
│  ├── current_feedback            str     User-facing feedback text      │
│  ├── previous_feedback_structures list   Last 5 structure templates     │
│  │                                                                       │
│  QUESTION SELECTOR AGENT (writes after question selection):             │
│  ├── current_question            dict    Next question to ask           │
│  │   └── .target_concepts        list    Dynamic rubric for follow-ups  │
│  ├── question_mode               str     "retrieve"|"follow_up"|"clarify"│
│  ├── follow_up_count             int     Follow-ups in current thread   │
│  ├── conversation_thread         list    Question IDs in current thread │
│  ├── topics_covered              list    All topics asked so far        │
│  │                                                                       │
│  SUPERVISOR AGENT (writes after fan-in):                                │
│  ├── stage                       str     Current interview stage        │
│  ├── should_continue             bool    Whether to ask next question   │
│  ├── end_reason                  str     Why interview ended            │
│  ├── question_count              int     INCREMENTED HERE (sole owner)  │
│  ├── difficulty_level            str     Current effective difficulty    │
│  ├── original_difficulty         str     Starting difficulty            │
│  ├── difficulty_reduced_due_to_performance  bool                        │
│  ├── performance_trajectory      list    Raw scores (excludes fallbacks)│
│  ├── ema_trajectory              list    Smoothed scores                │
│  ├── difficulty_history          list    Actual difficulty per question  │
│  ├── all_evaluations             list    Full evaluation history        │
│  │                                                                       │
│  CONVERSATION MANAGER (writes after summary check — graph node):        │
│  ├── conversation_summary        str     Compressed old turns           │
│  ├── summary_turn_count          int     Turns included in summary      │
│  │                                                                       │
│  API LAYER (writes on request receipt):                                 │
│  ├── candidate_response          str     User's answer                  │
│  ├── messages                    list    Full message history           │
│  │                                                                       │
│  IMMUTABLE AFTER /start:                                                │
│  ├── interview_id                str     Session identifier             │
│  ├── interview_plan              dict    Topic sequence, difficulty curve│
│  ├── interview_start_time        datetime                               │
│  ├── time_budget_minutes         int     Total time allowed             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State Structure

```python
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime
from langgraph.graph.message import add_messages
import operator


def last_value(existing, new):
    """Explicit last-write-wins reducer — makes the policy visible.
    Used for scalar fields owned by a single agent."""
    return new if new is not None else existing


class InterviewState(TypedDict):
    # ─── Conversation (managed by ConversationManager graph node) ───
    messages: Annotated[list, add_messages]
    conversation_summary: Annotated[Optional[str], last_value]
    summary_turn_count: Annotated[int, last_value]
    
    # ─── Current turn data (each owned by exactly one parallel branch) ───
    current_question: Annotated[Optional[dict], last_value]      # QS owns
    candidate_response: Annotated[Optional[str], last_value]     # API layer owns
    current_evaluation: Annotated[Optional[dict], last_value]    # Evaluator owns
    current_feedback: Annotated[Optional[str], last_value]       # Feedback owns
    
    # ─── Interview metadata (immutable after /start) ───
    interview_id: str
    interview_plan: Annotated[Optional[dict], last_value]
    interview_start_time: datetime
    time_budget_minutes: int
    
    # ─── Mutable metadata ───
    stage: Annotated[Literal["init", "planning", "questioning", "complete"], last_value]
    question_count: Annotated[int, last_value]  # Supervisor INCREMENTS after fan-in
    
    # ─── Difficulty tracking (Supervisor owns) ───
    difficulty_level: Annotated[Literal["easy", "medium", "hard"], last_value]
    original_difficulty: Annotated[Optional[str], last_value]
    difficulty_reduced_due_to_performance: Annotated[bool, last_value]
    
    # ─── Performance tracking (INTERNAL, Supervisor owns) ───
    # Lists use operator.add: agents return ONLY NEW items, reducer concatenates
    performance_trajectory: Annotated[list[float], operator.add]
    ema_trajectory: Annotated[list[float], last_value]  # Full recalc each turn, NOT incremental
    difficulty_history: Annotated[list[str], operator.add]
    all_evaluations: Annotated[list[dict], operator.add]
    
    # ─── Topic tracking (QS owns) ───
    topics_covered: Annotated[list[str], operator.add]
    
    # ─── Question flow (QS owns) ───
    question_mode: Annotated[Literal["retrieve", "follow_up", "clarify"], last_value]
    follow_up_count: Annotated[int, last_value]
    conversation_thread: Annotated[list[str], operator.add]
    
    # ─── Feedback variation tracking (Feedback Agent owns) ───
    previous_feedback_structures: Annotated[list[str], operator.add]
    
    # ─── Control flags (Supervisor owns) ───
    should_continue: Annotated[bool, last_value]
    needs_human_review: Annotated[bool, last_value]
    error_state: Annotated[Optional[dict], last_value]
    end_reason: Annotated[Optional[str], last_value]
```

**Key Reducer Rules:**
- `operator.add` lists: agents return **only new items** (e.g., `{"all_evaluations": [evaluation]}` not the full history)
- `last_value` scalars: agents return the new value directly
- `add_messages`: LangGraph's built-in message accumulator
- Every field has an explicit reducer — no implicit defaults

---

## Trend Analyzer (EMA Smoothing)

```python
class TrendAnalyzer:
    """
    EMA-smoothed trend detection to prevent difficulty oscillation.
    
    Why EMA:
    - Raw LLM scores are noisy (6.5 → 8.0 → 5.5 → 7.5)
    - Simple comparison causes false trend detection
    - EMA smooths noise while remaining responsive
    
    α = 0.3: Moderate smoothing (higher = more responsive, lower = smoother)
    
    Thresholds:
    - Increase: avg_ema >= 7.5 AND improving trend (lowered from 8.0;
      a candidate consistently at 7.5 is legitimately strong)
    - Decrease: avg_ema < 5.0 AND declining trend
    
    Fallback Protection:
    - Only operates on non-fallback scores
    - Fallback scores (is_fallback=True) are excluded BEFORE reaching this analyzer
    - Supervisor is responsible for filtering before calling TrendAnalyzer
    """
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
    
    def calculate_ema(self, trajectory: list[float]) -> list[float]:
        if not trajectory:
            return []
        ema = [trajectory[0]]
        for score in trajectory[1:]:
            ema.append(self.alpha * score + (1 - self.alpha) * ema[-1])
        return ema
    
    def get_trend(self, trajectory: list[float]) -> str:
        if len(trajectory) < 4:
            return "stable"
        ema = self.calculate_ema(trajectory)
        recent_ema = ema[-4:]
        change = recent_ema[-1] - recent_ema[0]
        if change > 0.8:
            return "improving"
        elif change < -0.8:
            return "declining"
        return "stable"
    
    def should_adjust_difficulty(
        self, trajectory: list[float]
    ) -> tuple[bool, str]:
        if len(trajectory) < 4:
            return False, "insufficient_data"
        
        ema = self.calculate_ema(trajectory)
        avg_ema = sum(ema[-4:]) / 4
        trend = self.get_trend(trajectory)
        
        # Lowered from 8.0 → 7.5: consistent 7.5 scorers deserve harder Qs
        if trend == "improving" and avg_ema >= 7.5:
            return True, "increase"
        
        if trend == "declining" and avg_ema < 5.0:
            return True, "decrease"
        
        return False, "stable"
    
    def get_current_ema(self, trajectory: list[float]) -> float:
        if not trajectory:
            return 5.0
        ema = self.calculate_ema(trajectory)
        return ema[-1]
```

---

## Conversation Manager (Graph Node, Sentence-Boundary Truncation)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
import re


SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are summarizing an AI interview for context continuity.\n"
        "Create a concise summary (~100 words) containing:\n"
        "- Topics covered\n"
        "- Key strengths demonstrated\n"
        "- Key weaknesses/gaps identified\n"
        "- Any misconceptions addressed\n\n"
        "Do NOT include full question/answer text, specific scores, "
        "or verbose descriptions."
    )),
    ("human", (
        "Existing summary:\n{existing_summary}\n\n"
        "New turns to incorporate:\n{new_turns}\n\n"
        "Updated summary:"
    ))
])


class ConversationManager:
    """
    Manages conversation context to prevent unbounded memory growth.
    Runs as a LangGraph node (not post-hoc outside the graph) so it
    is checkpointed like every other node.
    
    Strategy:
    - Keep last 3 turns in full detail (recent context)
    - Summarize older turns (compressed context)
    - Re-summarize every 3 new turns (batch efficiency)
    
    Memory Budget:
    - Summary: ~200 tokens (fixed)
    - Recent 3 turns: ~1500 tokens (variable)
    - Total: ~1700 tokens (bounded)
    """
    
    MAX_RECENT_TURNS = 3
    SUMMARIZE_EVERY_N_TURNS = 3
    
    def __init__(self, complex_llm: BaseChatModel):
        self.complex_llm = complex_llm
        self.summarize_chain = SUMMARIZATION_PROMPT | complex_llm
    
    async def maybe_update_summary(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        """
        LangGraph node: check if summary needs update, perform if needed.
        Returns partial state update (only changed keys).
        No-op most turns — returns empty dict when no update needed.
        """
        messages = state["messages"]
        # Count HumanMessage instances, not len//2 — robust against system msgs
        from langchain_core.messages import HumanMessage
        turn_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        summarized_count = state.get("summary_turn_count", 0)
        
        unsummarized = turn_count - summarized_count - self.MAX_RECENT_TURNS
        
        if unsummarized < self.SUMMARIZE_EVERY_N_TURNS:
            return {}  # No-op: reducer preserves existing values
        
        to_summarize_end = -(self.MAX_RECENT_TURNS * 2)
        to_summarize = messages[:to_summarize_end]
        
        new_summary = await self._create_summary(
            existing_summary=state.get("conversation_summary", ""),
            new_turns=to_summarize[-(self.SUMMARIZE_EVERY_N_TURNS * 2):],
            config=config
        )
        
        return {
            "conversation_summary": new_summary,
            "summary_turn_count": turn_count - self.MAX_RECENT_TURNS
        }
    
    async def get_context_for_agent(self, state: InterviewState) -> str:
        messages = state["messages"]
        turn_count = len(messages) // 2
        
        if turn_count <= self.MAX_RECENT_TURNS:
            return self._format_full(messages)
        
        recent_start = -(self.MAX_RECENT_TURNS * 2)
        recent = messages[recent_start:]
        
        summary = state.get("conversation_summary", "")
        context = ""
        if summary:
            context += f"Previous context:\n{summary}\n\n"
        context += f"Recent turns:\n{self._format_full(recent)}"
        return context
    
    async def _create_summary(
        self, existing_summary: str, new_turns: list, config: RunnableConfig
    ) -> str:
        response = await self.summarize_chain.ainvoke({
            "existing_summary": existing_summary or "No previous summary.",
            "new_turns": self._format_for_summary(new_turns)
        }, config=config)
        return response.content.strip()
    
    def _format_full(self, messages: list) -> str:
        formatted = []
        for i in range(0, len(messages), 2):
            q = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
            a = messages[i+1].content if i+1 < len(messages) else ""
            formatted.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(formatted)
    
    def _format_for_summary(self, messages: list) -> str:
        """Truncate at sentence boundaries, not mid-statement."""
        formatted = []
        for i in range(0, len(messages), 2):
            q = messages[i].content if hasattr(messages[i], 'content') else str(messages[i])
            a = messages[i+1].content if i+1 < len(messages) else ""
            formatted.append(
                f"Q: {self._truncate_at_sentence(q, 200)}\n"
                f"A: {self._truncate_at_sentence(a, 300)}"
            )
        return "\n".join(formatted)
    
    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int) -> str:
        """Truncate at the last sentence boundary within max_chars."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # Find last sentence-ending punctuation
        last_period = max(
            truncated.rfind('. '),
            truncated.rfind('? '),
            truncated.rfind('! '),
            truncated.rfind('.\n'),
        )
        if last_period > max_chars * 0.5:  # Only use if > 50% of budget
            return truncated[:last_period + 1]
        return truncated + "..."
```

---

## Session-Isolated Cache Store (Per-Session Locks, Atomic Select+Mark)

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import asyncio
from typing import Optional


@dataclass
class CacheEntry:
    value: object  # TopicQuestionCache or dict
    created_at: datetime
    ttl_seconds: int
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


@dataclass
class TopicQuestionCache:
    questions: list[dict]
    used_ids: set[str] = field(default_factory=set)
    crag_grade: str = "MEDIUM"
    
    @property
    def unused_questions(self) -> list[dict]:
        return [q for q in self.questions if q["id"] not in self.used_ids]
    
    @property
    def has_unused(self) -> bool:
        return len(self.unused_questions) > 0
    
    def mark_used(self, question_id: str):
        self.used_ids.add(question_id)


class InterviewCacheStore:
    """
    Singleton cache with SEPARATE POOLS per type and session lifecycle management.
    
    Why separate pools:
    - Topic caches are high-value (used for upcoming questions)
    - Concept caches are numerous but smaller
    - Without separation, concept flood can evict needed topic caches
    
    Pool limits:
    - Topics: 10 per session (fewer, more valuable)
    - Concepts: 30 per session (more numerous, smaller)
    
    Concurrency:
    - PER-SESSION locks (not a single global lock)
    - Under concurrent interviews, sessions don't block each other
    - Atomic select_and_mark eliminates TOCTOU race on question selection
    
    Session Lifecycle:
    - Sessions tracked with creation time
    - Periodic sweep clears sessions older than max_session_age
    - Explicit clear on /end
    - Prevents memory leak from abandoned sessions
    """
    
    _instance = None
    
    TTL_BY_GRADE = {
        "HIGH": 1800,    # 30 minutes
        "MEDIUM": 900,   # 15 minutes
        "LOW": 300       # 5 minutes
    }
    
    MAX_SESSION_AGE_MINUTES = 90
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._topic_cache: dict[str, OrderedDict] = {}
            cls._instance._concept_cache: dict[str, OrderedDict] = {}
            cls._instance._session_created_at: dict[str, datetime] = {}
            cls._instance._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
            cls._instance._global_lock = asyncio.Lock()  # Only for session creation/cleanup
            cls._instance._max_topics_per_session = 10
            cls._instance._max_concepts_per_session = 30
        return cls._instance
    
    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Per-session lock — sessions never block each other."""
        return self._session_locks[session_id]
    
    def _get_session_topic_cache(self, session_id: str) -> OrderedDict:
        if session_id not in self._topic_cache:
            self._topic_cache[session_id] = OrderedDict()
            self._session_created_at[session_id] = datetime.now()
        return self._topic_cache[session_id]
    
    def _get_session_concept_cache(self, session_id: str) -> OrderedDict:
        if session_id not in self._concept_cache:
            self._concept_cache[session_id] = OrderedDict()
            self._session_created_at[session_id] = datetime.now()
        return self._concept_cache[session_id]
    
    # ─────────────────────────────────────────────────────────────────
    # TOPIC QUESTION CACHE (Pool 1)
    # ─────────────────────────────────────────────────────────────────
    
    async def get_topic_questions(
        self, session_id: str, topic: str, difficulty: str
    ) -> Optional[TopicQuestionCache]:
        """Check if cached questions exist with unused items."""
        key = f"{topic}:{difficulty}"
        cache = self._get_session_topic_cache(session_id)
        lock = self._get_lock(session_id)
        
        async with lock:
            entry = cache.get(key)
            if entry is None or entry.is_expired:
                if entry:
                    del cache[key]
                return None
            cache.move_to_end(key)
            if not entry.value.has_unused:
                return None
            return entry.value
    
    async def select_and_mark(
        self, session_id: str, topic: str, difficulty: str,
        selector_fn
    ) -> Optional[dict]:
        """
        Atomic select + mark_used within a single lock acquisition.
        Eliminates TOCTOU race between get_topic_questions and mark_question_used.
        
        selector_fn: async callable that takes list[dict] candidates 
        and returns selected dict. Called INSIDE the lock.
        """
        key = f"{topic}:{difficulty}"
        cache = self._get_session_topic_cache(session_id)
        lock = self._get_lock(session_id)
        
        async with lock:
            entry = cache.get(key)
            if entry is None or entry.is_expired:
                if entry:
                    del cache[key]
                return None
            
            cache.move_to_end(key)
            topic_cache = entry.value
            
            if not topic_cache.has_unused:
                return None
            
            # selector_fn runs inside the lock — atomically selects and marks
            selected = await selector_fn(topic_cache.unused_questions)
            if selected:
                topic_cache.mark_used(selected["id"])
            return selected
    
    async def set_topic_questions(
        self, session_id: str, topic: str, difficulty: str,
        questions: list[dict], crag_grade: str = "MEDIUM"
    ) -> None:
        key = f"{topic}:{difficulty}"
        cache = self._get_session_topic_cache(session_id)
        ttl = self.TTL_BY_GRADE.get(crag_grade, 900)
        lock = self._get_lock(session_id)
        
        topic_cache = TopicQuestionCache(
            questions=questions, used_ids=set(), crag_grade=crag_grade
        )
        
        async with lock:
            cache[key] = CacheEntry(
                value=topic_cache, created_at=datetime.now(), ttl_seconds=ttl
            )
            cache.move_to_end(key)
            self._evict_if_needed(cache, self._max_topics_per_session)
    
    # ─────────────────────────────────────────────────────────────────
    # CONCEPT CACHE (Pool 2)
    # ─────────────────────────────────────────────────────────────────
    
    async def get_concept(self, session_id: str, concept_name: str) -> Optional[dict]:
        cache = self._get_session_concept_cache(session_id)
        lock = self._get_lock(session_id)
        
        async with lock:
            entry = cache.get(concept_name)
            if entry is None or entry.is_expired:
                if entry:
                    del cache[concept_name]
                return None
            cache.move_to_end(concept_name)
            return entry.value
    
    async def set_concept(
        self, session_id: str, concept_name: str, concept_data: dict
    ) -> None:
        cache = self._get_session_concept_cache(session_id)
        lock = self._get_lock(session_id)
        
        async with lock:
            cache[concept_name] = CacheEntry(
                value=concept_data, created_at=datetime.now(), ttl_seconds=3600
            )
            cache.move_to_end(concept_name)
            self._evict_if_needed(cache, self._max_concepts_per_session)
    
    # ─────────────────────────────────────────────────────────────────
    # CACHE MANAGEMENT + SESSION LIFECYCLE
    # ─────────────────────────────────────────────────────────────────
    
    def _evict_if_needed(self, cache: OrderedDict, max_entries: int) -> None:
        while len(cache) > max_entries:
            oldest_key = next(iter(cache))
            del cache[oldest_key]
    
    async def clear_session(self, session_id: str) -> None:
        async with self._global_lock:
            self._topic_cache.pop(session_id, None)
            self._concept_cache.pop(session_id, None)
            self._session_created_at.pop(session_id, None)
            self._session_locks.pop(session_id, None)
    
    async def cleanup_abandoned_sessions(self) -> int:
        cutoff = datetime.now() - timedelta(minutes=self.MAX_SESSION_AGE_MINUTES)
        cleaned = 0
        
        async with self._global_lock:
            expired_sessions = [
                sid for sid, created in self._session_created_at.items()
                if created < cutoff
            ]
            for sid in expired_sessions:
                self._topic_cache.pop(sid, None)
                self._concept_cache.pop(sid, None)
                self._session_created_at.pop(sid, None)
                self._session_locks.pop(sid, None)
                cleaned += 1
        
        return cleaned
    
    async def pre_warm_topics_background(
        self, session_id: str, rag_service: "AgenticRAGService",
        topics: list[str], difficulty: str
    ) -> None:
        tasks = [
            self._pre_warm_single(session_id, rag_service, topic, difficulty)
            for topic in topics
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _pre_warm_single(
        self, session_id: str, rag_service: "AgenticRAGService",
        topic: str, difficulty: str
    ) -> None:
        existing = await self.get_topic_questions(session_id, topic, difficulty)
        if existing:
            return
        result = await rag_service.retrieve_batch(
            topic=topic, difficulty=difficulty, n=5
        )
        await self.set_topic_questions(
            session_id, topic, difficulty, result.candidates, result.grade
        )
```

---

## Agentic RAG Service (CRAG Owner)

```python
from dataclasses import dataclass, field
from typing import List, Optional
from src.rag.models import RetrievalResult
from src.rag.cache import RelevanceGrade


@dataclass
class RAGResult:
    """Result from AgenticRAGService.retrieve_with_crag().
    Superset of original CRAGResult — adds observability fields.
    """
    candidates: List[RetrievalResult]
    grade: RelevanceGrade              # HIGH, MEDIUM, LOW
    attempts: int = 1
    refined_query: str | None = None
    served_from_cache: bool = False
    corrective_applied: bool = False
    queries_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_fallback: bool = False


class AgenticRAGService:
    """
    Owns the full CRAG retrieval loop via a compiled LangGraph StateGraph:
    retrieve → grade → (refine → retrieve) → package_results.
    
    Uses LangGraph subgraph for tracing, visualization, and composability.
    """
    
    MAX_CRAG_ATTEMPTS = 2
    
    def __init__(
        self,
        retriever: "VectorRetriever",
        grader: "DocumentGrader",
        query_refiner: "QueryRefiner",
        cache_store: Optional["InterviewCacheStore"] = None,
    ):
        self.retriever = retriever
        self.grader = grader
        self.query_refiner = query_refiner
        self.cache_store = cache_store or get_cache_store()
        self.crag_graph = build_crag_graph(retriever, grader, query_refiner)
    
    async def retrieve_with_crag(
        self,
        topic: str,
        difficulty: str,
        exclude_ids: list[str],
        remaining_time: float | None = None,
        n: int = 5,
        session_id: str | None = None,
    ) -> RAGResult:
        """Primary retrieval API — called by QuestionSelectorAgent.
        Tries cache → CRAG subgraph → fallback in order.
        """
        # ... cache check, CRAG ainvoke, cache store, fallback ...
        pass
    
    async def retrieve_batch(
        self, topic: str, difficulty: str, n: int = 5
    ) -> RAGResult:
        """Thin wrapper for pre-warming — no exclusions."""
        return await self.retrieve_with_crag(
            topic=topic, difficulty=difficulty,
            exclude_ids=[], n=n,
        )
    
    async def end_interview(self, session_id: str) -> int:
        """Clear all cached data for a finished session."""
        return await self.cache_store.clear_session(session_id)
```

---

## Document Grader (Standalone, Hybrid: Score Fast-Path + LLM)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal, List, Optional
from src.rag.cache import RelevanceGrade
from src.rag.models import RetrievalResult, RetrievalContext


class DocumentGrade(BaseModel):
    """LLM grading decision for borderline retrieval results."""
    grade: str       # HIGH, MEDIUM, or LOW
    feedback: str    # One sentence explaining the grade


class GradingResult:
    """Returned by DocumentGrader.grade() — never raises."""
    grade: RelevanceGrade
    feedback: str
    avg_score: float
    used_llm: bool = False
    penalised_score: Optional[float] = None


class DocumentGrader:
    """
    Hybrid grader: score-based fast paths, LLM only for borderline.
    Uses .with_structured_output() for reliable parsing.
    Independently testable without the interview graph.
    
    Fast paths (~70% of calls, no LLM):
      avg_score >= 0.75 → HIGH
      avg_score <= 0.45 → LOW
    Borderline (0.45 < score < 0.75) → 7B LLM grading
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        # Uses get_secondary_llm() by default (7B)
        self.chain = _build_grading_chain(llm or get_secondary_llm())
    
    async def grade(
        self,
        documents: List[RetrievalResult],
        context: RetrievalContext,
        topic_intent: str = ""
    ) -> GradingResult:
        """
        Grade a batch of retrieved documents.
        Returns GradingResult with grade, feedback, scores — never raises.
        """
        # 1. Compute avg relevance score with penalties
        # 2. Fast path for obvious HIGH/LOW
        # 3. LLM-grade only borderline cases
        pass
```

---

## Query Refiner (Loop-Safe, Strategy Rotation)

```python
from typing import List, Optional, Tuple
from enum import Enum


class QueryRefinementStrategy(str, Enum):
    LLM_REFINE = "llm_refine"
    TOPIC_PIVOT = "topic_pivot"
    SIMPLIFY = "simplify"
    FORCED = "forced"


class QueryRefiner:
    """
    Corrective action service for CRAG retry loop.
    Called by AgenticRAGService when grade == LOW.
    
    Full signature: includes feedback, seen_queries for anti-repeat,
    and covered_topics for topic-pivot strategy.
    """
    
    async def refine(
        self,
        original_query: str,
        feedback: str,
        difficulty: str,
        seen_queries: List[str],
        attempt: int,
        covered_topics: Optional[List[str]] = None,
    ) -> Tuple[str, QueryRefinementStrategy]:
        """
        Refine a query for retry after LOW grade.
        Returns (new_query, strategy_used).
        
        Strategies rotate by attempt:
        - 0: LLM_REFINE — rephrase using grading feedback
        - 1: TOPIC_PIVOT — shift to uncovered sub-topic
        - 2+: SIMPLIFY — strip qualifiers, broaden scope
        
        Anti-repeat: rejects queries with cosine > 0.85 to any seen_query.
        """
        pass
```

---

## Tools Layer (LangChain @tool Decorator)

```python
from langchain_core.tools import tool


@tool
async def rubric_lookup(question_id: str) -> dict:
    """Look up the scoring rubric for a given question ID.
    Returns key_points, scoring_criteria, and difficulty metadata."""
    # ... implementation ...


@tool
async def code_validator(code: str, language: str = "python") -> dict:
    """Validate code syntax using AST parsing.
    Returns is_valid, errors, and warnings."""
    # ... implementation ...


@tool
async def concept_lookup(concept_name: str) -> dict:
    """Look up an ML/AI concept explanation from the knowledge base.
    Returns explanation, simple_explanation, and examples."""
    # ... implementation ...
```

Using `@tool` gives automatic LangSmith tracing, input/output schema validation, and the option to bind tools to models via `llm.bind_tools([...])`.

---

## Supervisor Agent (Owns question_count, Fixed Difficulty Index)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFFICULTY_FROM_ORDER = {0: "easy", 1: "medium", 2: "hard"}

INTERVIEW_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI interview planner. Create a structured plan.\n"
        "Return valid JSON with: topic_sequence, difficulty_curve, "
        "time_allocation, focus_areas.\n"
        "difficulty_curve has ONE entry per topic (not per question)."
    )),
    ("human", (
        "Difficulty: {difficulty}\n"
        "Focus topics: {focus_topics}\n"
        "Time budget: {time_budget} minutes\n"
        "Target questions: {target_questions}"
    ))
])


class SupervisorAgent:
    """
    Orchestrates the interview flow.
    
    Key responsibilities:
    - Plan-and-Execute at /start (1 LLM call, 14B)
    - Rule-based OODA routing per turn (0 LLM calls)
    - EMA-smoothed trend detection for difficulty adaptation
    - Owns question_count: INCREMENTS in validate_and_decide after fan-in
    - Indexes difficulty_curve by TOPIC INDEX (len(topics_covered)), not question_count
    - NO early termination for performance
    
    Decoupled from RAG at /start:
    - Supervisor ONLY creates the plan
    - First topic retrieval is handled by QS in the start_graph's first_question node
    - Background pre-warming of remaining topics triggered by API layer
    """
    
    def __init__(
        self, complex_llm: BaseChatModel,
        trend_analyzer: TrendAnalyzer
    ):
        self.complex_llm = complex_llm
        self.trend_analyzer = trend_analyzer
        self.plan_chain = INTERVIEW_PLAN_PROMPT | complex_llm
        self.validation_gates = ValidationGateRegistry()
        self.circuit_breaker = CircuitBreaker(max_retries=1)
    
    # ─────────────────────────────────────────────────────────────────
    # PLAN CREATION (No RAG dependency — QS handles first question)
    # ─────────────────────────────────────────────────────────────────
    
    async def create_interview_plan(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        """
        Create plan only. First topic retrieval handled by QS.
        Background pre-warming triggered by API layer after graph completes.
        """
        target_questions = self._calculate_target_questions(state["time_budget_minutes"])
        
        response = await self.plan_chain.ainvoke({
            "difficulty": state["difficulty_level"],
            "focus_topics": state.get("focus_topics", []),
            "time_budget": state["time_budget_minutes"],
            "target_questions": target_questions,
        }, config=config)
        
        plan = self._parse_plan(response.content)
        first_difficulty = plan["difficulty_curve"][0]
        
        return {
            "interview_plan": plan,
            "original_difficulty": state["difficulty_level"],
            "difficulty_level": first_difficulty,
            "difficulty_reduced_due_to_performance": False,
            "stage": "questioning"
        }
    
    def _calculate_target_questions(self, time_budget: int) -> int:
        return max(5, min(12, time_budget // 4))
    
    def _parse_plan(self, content: str) -> dict:
        return parse_json_safely(content)
    
    # ─────────────────────────────────────────────────────────────────
    # OODA LOOP (question_count incremented here, topic-indexed difficulty)
    # ─────────────────────────────────────────────────────────────────
    
    async def validate_and_decide(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        """
        Rule-based OODA with EMA authority and fallback protection.
        Supervisor is the SOLE owner of question_count — increments here.
        """
        evaluation = state["current_evaluation"]
        is_fallback = evaluation.get("is_fallback", False)
        new_score = evaluation["overall_score"]
        
        # Reducer handles concatenation — return only NEW items
        new_all_evaluations = [evaluation]
        new_difficulty_history = [state["difficulty_level"]]
        
        if is_fallback:
            new_trajectory = []  # Empty — reducer.add won't change existing
        else:
            new_trajectory = [new_score]
        
        # Calculate EMA on FULL trajectory (existing + new)
        full_trajectory = state.get("performance_trajectory", []) + new_trajectory
        new_ema = self.trend_analyzer.calculate_ema(full_trajectory)
        
        observation = self._observe(state, new_ema)
        analysis = self._orient(observation, full_trajectory)
        should_continue, end_reason = self._decide_continuation(analysis, state)
        new_difficulty, reduced = self._resolve_difficulty(analysis, state)
        
        return {
            # Lists: return only NEW items (operator.add reducer)
            "performance_trajectory": new_trajectory,
            "ema_trajectory": new_ema,  # Full recalc — replaces via last_value
            "difficulty_history": new_difficulty_history,
            "all_evaluations": new_all_evaluations,
            
            # Scalars: return new value (last_value reducer)
            "should_continue": should_continue,
            "end_reason": end_reason,
            "difficulty_level": new_difficulty,
            "difficulty_reduced_due_to_performance": (
                state.get("difficulty_reduced_due_to_performance", False) or reduced
            ),
            "stage": "questioning",
            
            # INCREMENT question_count — Supervisor is sole owner
            "question_count": state["question_count"] + 1,
        }
    
    def _observe(self, state: InterviewState, ema: list[float]) -> "Observation":
        elapsed = self._get_elapsed_minutes(state)
        remaining = state["time_budget_minutes"] - elapsed
        return Observation(
            question_count=state["question_count"],
            topics_covered=state["topics_covered"],
            difficulty_level=state["difficulty_level"],
            elapsed_minutes=elapsed,
            remaining_minutes=remaining,
            current_ema=ema[-1] if ema else 5.0,
            target_questions=len(state.get("interview_plan", {}).get("topic_sequence", []))
        )
    
    def _orient(self, obs: "Observation", trajectory: list[float]) -> "Analysis":
        trend = self.trend_analyzer.get_trend(trajectory)
        should_adjust, direction = self.trend_analyzer.should_adjust_difficulty(trajectory)
        return Analysis(
            performance_trend=trend,
            avg_ema=obs.current_ema,
            should_adjust_difficulty=should_adjust,
            adjustment_direction=direction,
            time_pressure=obs.remaining_minutes < 5,
            time_critical=obs.remaining_minutes < 2,
            questions_remaining=obs.target_questions - obs.question_count
        )
    
    def _decide_continuation(
        self, analysis: "Analysis", state: InterviewState
    ) -> tuple[bool, Optional[str]]:
        if analysis.time_critical:
            return False, "time_up"
        target = len(state.get("interview_plan", {}).get("topic_sequence", [10]))
        if state["question_count"] + 1 >= target:  # +1 because we just incremented
            return False, "completed"
        return True, None
    
    def _resolve_difficulty(
        self, analysis: "Analysis", state: InterviewState
    ) -> tuple[str, bool]:
        current = state["difficulty_level"]
        if analysis.time_pressure:
            return current, False
        if state.get("question_mode") in ("follow_up", "clarify"):
            return current, False
        
        plan_difficulty = self._get_plan_difficulty_for_next_topic(state)
        
        if not analysis.should_adjust_difficulty:
            return plan_difficulty, False
        
        if analysis.adjustment_direction == "increase":
            ema_adjusted = {"easy": "medium", "medium": "hard", "hard": "hard"}[current]
            return self._harder_of(plan_difficulty, ema_adjusted), False
        elif analysis.adjustment_direction == "decrease":
            ema_adjusted = {"easy": "easy", "medium": "easy", "hard": "medium"}[current]
            return self._easier_of(plan_difficulty, ema_adjusted), True
        
        return plan_difficulty, False
    
    def _get_plan_difficulty_for_next_topic(self, state: InterviewState) -> str:
        """
        Index by TOPIC INDEX (len(topics_covered)), NOT question_count.
        difficulty_curve has one entry per topic. Follow-up questions
        don't advance the curve.
        """
        plan = state.get("interview_plan", {})
        curve = plan.get("difficulty_curve", [])
        topic_index = len(state.get("topics_covered", []))
        
        if topic_index < len(curve):
            return curve[topic_index]
        return state["difficulty_level"]
    
    def _harder_of(self, a: str, b: str) -> str:
        return DIFFICULTY_FROM_ORDER[max(DIFFICULTY_ORDER[a], DIFFICULTY_ORDER[b])]
    
    def _easier_of(self, a: str, b: str) -> str:
        return DIFFICULTY_FROM_ORDER[min(DIFFICULTY_ORDER[a], DIFFICULTY_ORDER[b])]
    
    def _get_elapsed_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return 0
        return (datetime.now() - state["interview_start_time"]).total_seconds() / 60
```

**Note on ema_trajectory:** Since EMA is a full recalculation (not incremental), it uses `last_value` reducer (replaces the whole list), while `performance_trajectory` uses `operator.add` (appends new scores). This is an intentional difference.

```python
# In InterviewState, ema_trajectory needs last_value, not operator.add:
ema_trajectory: Annotated[list[float], last_value]  # Full recalc each turn
```

---

## Validation Gates (Dynamic Rubric Support, Drift Detection)

```python
class ValidationGateRegistry:
    def __init__(self):
        self.gates = {
            "evaluator": EvaluatorValidationGate(),
            "feedback": FeedbackValidationGate(),
            "question_selector": QuestionSelectorValidationGate()
        }
    
    def get(self, agent_name: str) -> "ValidationGate":
        return self.gates[agent_name]


class EvaluatorValidationGate:
    """
    Validates Evaluator outputs with drift detection.
    
    Handles BOTH static and dynamic rubrics:
    - Static rubric: retrieved questions have key_points from DB
    - Dynamic rubric: follow-up/clarify Qs have target_concepts from QS
    
    Without dynamic rubric support, drift detection is silently disabled
    for ~30-40% of questions (all follow-ups and clarifications).
    """
    
    def validate(self, output: dict, question: dict) -> "ValidationResult":
        """
        Args:
            output: evaluation dict from Evaluator
            question: current_question dict (contains rubric OR target_concepts)
        """
        rubric_key_points = self._extract_key_points(question)
        
        checks = [
            self._scores_in_range(output),
            self._reasoning_provided(output),
            self._scores_consistent(output),
            self._required_fields_present(output),
            self._key_point_coverage_alignment(output, rubric_key_points)
        ]
        
        failed = [c for c in checks if not c.passed]
        return ValidationResult(
            is_valid=len(failed) == 0,
            failed_checks=failed,
            feedback=[c.message for c in failed]
        )
    
    def _extract_key_points(self, question: dict) -> list[str]:
        """
        Extract key points from EITHER static rubric or dynamic target_concepts.
        
        For retrieved questions: question["rubric"]["key_points"]
        For follow-up/clarify: question["target_concepts"] (set by QS)
        """
        # Static rubric (retrieved questions)
        rubric = question.get("rubric", {})
        static_points = rubric.get("criteria", {}).get(
            "technical_accuracy", {}
        ).get("key_points", [])
        
        if static_points:
            return static_points
        
        # Dynamic rubric (follow-up/clarification questions)
        target_concepts = question.get("target_concepts", [])
        if target_concepts:
            return target_concepts
        
        # Clarification: single misconception is the "key point"
        misconception = question.get("target_misconception")
        if misconception:
            return [f"Corrects misconception: {misconception}"]
        
        return []
    
    def _key_point_coverage_alignment(
        self, output: dict, key_points: list[str]
    ) -> "Check":
        if not key_points:
            # No key points available — skip alignment check but LOG it
            return Check(
                passed=True,
                message="No key points for alignment check (expected for some question types)"
            )
        
        covered = output.get("key_points_covered", [])
        score = output.get("overall_score", 5.0)
        coverage_ratio = len(covered) / len(key_points)
        
        if score >= 8.0 and coverage_ratio < 0.5:
            return Check(
                passed=False,
                message=f"Score {score} but only {coverage_ratio:.0%} key points covered"
            )
        if score <= 4.0 and coverage_ratio > 0.7:
            return Check(
                passed=False,
                message=f"Score {score} but {coverage_ratio:.0%} key points covered"
            )
        return Check(passed=True)
    
    def _scores_in_range(self, output: dict) -> "Check":
        score_fields = ["technical_accuracy", "completeness", "depth", "clarity", "overall_score"]
        for f in score_fields:
            val = output.get(f)
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None and not (0 <= float(score) <= 10):
                return Check(passed=False, message=f"{f}={score} outside 0-10")
        return Check(passed=True)
    
    def _reasoning_provided(self, output: dict) -> "Check":
        reasoning = output.get("reasoning", "")
        if len(reasoning) < 50:
            return Check(passed=False, message="Reasoning too short")
        return Check(passed=True)
    
    def _scores_consistent(self, output: dict) -> "Check":
        scores = []
        for f in ["technical_accuracy", "completeness", "depth", "clarity"]:
            val = output.get(f)
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None:
                scores.append(float(score))
        if scores and max(scores) - min(scores) > 5:
            return Check(passed=False, message="Score variance too high")
        return Check(passed=True)
    
    def _required_fields_present(self, output: dict) -> "Check":
        required = ["overall_score", "reasoning", "key_points_covered", "topic"]
        missing = [f for f in required if f not in output]
        if missing:
            return Check(passed=False, message=f"Missing: {missing}")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        return {
            "overall_score": 5.0,
            "technical_accuracy": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "completeness": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "depth": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "clarity": {"score": 5.0, "reasoning": "Unable to evaluate"},
            "reasoning": "Evaluation incomplete. Neutral scores assigned.",
            "needs_human_review": True,
            "is_fallback": True,
            "key_points_covered": [],
            "key_points_missed": [],
            "misconceptions": [],
            "topic": "unknown"  # Fallback must include topic field
        }


class FeedbackValidationGate:
    def validate(self, output: dict, evaluation_score: float) -> "ValidationResult":
        checks = [
            self._appropriate_length(output["feedback_text"]),
            self._no_forbidden_phrases(output["feedback_text"]),
            self._no_sycophancy_at_low_scores(output["feedback_text"], evaluation_score),
            self._no_score_leakage(output["feedback_text"])
        ]
        failed = [c for c in checks if not c.passed]
        return ValidationResult(is_valid=len(failed) == 0, failed_checks=failed)
    
    def _appropriate_length(self, text: str) -> "Check":
        words = len(text.split())
        if words < 20:
            return Check(passed=False, message="Feedback too short")
        if words > 200:
            return Check(passed=False, message="Feedback too long")
        return Check(passed=True)
    
    def _no_forbidden_phrases(self, text: str) -> "Check":
        forbidden = ["you failed", "wrong answer", "incorrect",
                     "you don't understand", "completely wrong"]
        text_lower = text.lower()
        for phrase in forbidden:
            if phrase in text_lower:
                return Check(passed=False, message=f"Forbidden: '{phrase}'")
        return Check(passed=True)
    
    def _no_sycophancy_at_low_scores(self, text: str, score: float) -> "Check":
        if score >= 7.0:
            return Check(passed=True)
        sycophantic = [
            "great job", "excellent", "perfect", "well done",
            "amazing", "fantastic", "wonderful", "brilliant",
            "impressive", "outstanding"
        ]
        opening = text.lower()[:150]
        for phrase in sycophantic:
            if phrase in opening:
                return Check(passed=False, message=f"Sycophantic '{phrase}' for score {score:.1f}")
        return Check(passed=True)
    
    def _no_score_leakage(self, text: str) -> "Check":
        import re
        patterns = [r'\b\d+\.?\d*/10\b', r'\bscored?\s+\d+',
                    r'\b\d+\.?\d*\s*out of', r'rating[:\s]+\d+']
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                return Check(passed=False, message="Score leaked")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        return {
            "feedback_text": "Thank you for your response. Let's continue.",
            "strength_acknowledgment": "",
            "gap_hint": "",
            "transition_phrase": "Moving on"
        }


class QuestionSelectorValidationGate:
    def validate(self, output: dict, remaining_minutes: float) -> "ValidationResult":
        checks = [
            self._question_present(output),
            self._valid_question_type(output),
            self._time_appropriate(output, remaining_minutes)
        ]
        failed = [c for c in checks if not c.passed]
        return ValidationResult(is_valid=len(failed) == 0, failed_checks=failed)
    
    def _question_present(self, output: dict) -> "Check":
        if not output.get("text"):
            return Check(passed=False, message="No question text")
        return Check(passed=True)
    
    def _valid_question_type(self, output: dict) -> "Check":
        valid = ["retrieved", "follow_up", "clarification"]
        if output.get("question_type") not in valid:
            return Check(passed=False, message="Invalid question type")
        return Check(passed=True)
    
    def _time_appropriate(self, output: dict, remaining_minutes: float) -> "Check":
        est_time = output.get("estimated_time_minutes", 5)
        if est_time > remaining_minutes + 2:
            return Check(passed=False, message="Question too long")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        return {
            "id": "fallback_001",
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4,
            "target_concepts": ["bias", "variance", "tradeoff"]
        }


class CircuitBreaker:
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self.retry_counts: dict[str, int] = {}
    
    def should_retry(self, agent_name: str) -> bool:
        count = self.retry_counts.get(agent_name, 0)
        if count < self.max_retries:
            self.retry_counts[agent_name] = count + 1
            return True
        return False
    
    def reset(self, agent_name: str = None):
        if agent_name:
            self.retry_counts[agent_name] = 0
        else:
            self.retry_counts = {}
```

---

## Resilience Patterns (LangChain 1.0+ Best Practices)

### Chain Retry Policy

All LCEL chains use `.with_retry()` — the standard LangChain mechanism for transient failure recovery:

```python
# Applied to every agent chain
chain = (
    prompt
    | llm.with_structured_output(OutputModel)
).with_retry(
    stop_after_attempt=2,
    wait_exponential_jitter=True,
)
```

### Structured Output Fallback

Critical chains (Evaluator, QS) use `.with_fallbacks()` for graceful degradation when `.with_structured_output()` fails (e.g., malformed JSON from Ollama):

```python
from langchain_core.output_parsers import JsonOutputParser

strict_chain = prompt | llm.with_structured_output(EvaluationOutput)
lenient_chain = prompt | llm | JsonOutputParser()  # regex-based parse

chain = strict_chain.with_fallbacks([lenient_chain])
```

### Model Routing via `configurable_alternatives`

Graceful degradation between 7B/14B uses LangChain's `configurable_alternatives()` — explicit, traceable, testable:

```python
from langchain_core.runnables import ConfigurableField

def get_configurable_llm(default_tier: str = "fast") -> BaseChatModel:
    fast = get_fast_llm()
    complex_model = get_complex_llm()
    
    if default_tier == "fast":
        return fast.configurable_alternatives(
            ConfigurableField(id="model_tier"),
            complex=complex_model,
        )
    else:
        return complex_model.configurable_alternatives(
            ConfigurableField(id="model_tier"),
            fast=fast,
        )

# At runtime, if health check detects 7B is down:
result = await chain.ainvoke(
    input, config={"configurable": {"model_tier": "complex"}}
)
```

### Agent Node Timeout Wrappers

Each agent graph node is wrapped with `asyncio.wait_for()` to prevent Ollama hangs (GPU memory pressure, long context):

```python
async def evaluator_with_timeout(state: InterviewState, config: RunnableConfig):
    return await asyncio.wait_for(
        agents.evaluator.execute(state, config),
        timeout=15.0  # seconds
    )

# In graph assembly:
graph.add_node("evaluator", evaluator_with_timeout)
```

TimeoutError is caught by the circuit breaker, which fires the fallback.

---

## Inter-Agent Contracts (Pydantic Models)

Agents communicate through `InterviewState`, but downstream agents depend on specific field shapes. Pydantic contracts catch schema violations at validation-gate time, not at downstream consumption time.

```python
from pydantic import BaseModel
from typing import Optional


class EvaluationOutput(BaseModel):
    """Contract: Evaluator → Supervisor + QS.
    QS reads overall_score, key_points_missed, misconceptions.
    Supervisor reads overall_score, is_fallback.
    """
    overall_score: float
    technical_accuracy: float
    completeness: float
    depth: float
    clarity: float
    evaluation_reasoning: str
    key_points_missed: list[str] = []
    misconceptions: list[str] = []
    is_fallback: bool = False
    topic: str = ""  # Injected from current_question


class FeedbackOutput(BaseModel):
    """Contract: Feedback Agent → API layer."""
    feedback_text: str
    strength_acknowledgment: str
    gap_hint: str
    transition_phrase: str
    structure_template: str  # For variation tracking


class QuestionOutput(BaseModel):
    """Contract: QS → Evaluator (next turn) + API layer."""
    id: str
    text: str
    question_type: str  # "retrieved" | "follow_up" | "clarification"
    topic: str
    difficulty: str
    target_concepts: list[str] = []  # Dynamic rubric for follow-ups
    estimated_time_minutes: float = 5.0
```

Validation gates use these contracts:
```python
class EvaluatorValidationGate:
    def validate(self, output: dict, question: dict) -> ValidationResult:
        # Parse with contract model first
        try:
            parsed = EvaluationOutput(**output)
        except ValidationError as e:
            return ValidationResult(is_valid=False, failed_checks=[...], feedback=[str(e)])
        # Then run domain checks...
```

---

## Feedback Agent (Structured Output, More Templates)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class FeedbackComponents(BaseModel):
    strength_acknowledgment: str = Field(
        description="1 sentence acknowledging what they did well. Empty if score < 5."
    )
    gap_hint: str = Field(
        description="Implicit hint about gaps WITHOUT stating 'you missed X'."
    )
    transition_phrase: str = Field(
        description="Natural transition. E.g., 'Building on that...'"
    )


FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an interview feedback generator. Generate constructive feedback.\n"
        "NEVER reveal scores, NEVER say 'you missed X', NEVER use forbidden phrases.\n"
        "Tone guidance: {tone_guidance}\n\n"
        "If concept context is provided, subtly weave it into your hint.\n"
        "Concept context: {concept_context}"
    )),
    ("human", (
        "Score band: {score_band}\n"
        "Question: {question}\n"
        "Response: {response}"
    ))
])


class FeedbackComposer:
    """
    Composes feedback with varied structures to avoid Mad Libs feel.
    """
    
    STRUCTURES = {
        "high": [
            "{strength}",
            "{strength} {transition}",
            "{transition}",                    # Skip strength sometimes
            "{strength}",                      # Added: avoids 2-template repetition
        ],
        "medium": [
            "{strength} {gap_hint}",
            "{gap_hint} {strength}",
            "{strength} {transition}",
            "{gap_hint} {transition}",
        ],
        "low": [
            "{gap_hint}",
            "{gap_hint} {transition}",
            "{transition} {gap_hint}",         # Added for variety
        ]
    }
    
    TRANSITIONS = [
        "Building on that...",
        "I'm curious...",
        "Let me ask you this...",
        "Thinking about that...",
        "Related to what you mentioned...",
        "On that note...",
        "Following up...",
        "",
    ]
    
    def compose(
        self, components: FeedbackComponents, score: float,
        turn_number: int, previous_structures: list[str]
    ) -> str:
        score_band = self._get_score_band(score)
        structures = self.STRUCTURES[score_band]
        structure = self._select_varied_structure(structures, turn_number, previous_structures)
        transition = self._select_transition(turn_number)
        
        result = structure.format(
            strength=components.strength_acknowledgment,
            gap_hint=components.gap_hint,
            transition=transition
        )
        return " ".join(result.split())
    
    def _get_score_band(self, score: float) -> str:
        if score >= 8.0:
            return "high"
        elif score >= 5.0:
            return "medium"
        return "low"
    
    def _select_varied_structure(
        self, structures: list[str], turn: int, previous: list[str]
    ) -> str:
        available = [s for s in structures if s not in previous[-2:]]
        if not available:
            available = structures
        return available[turn % len(available)]
    
    def _select_transition(self, turn_number: int) -> str:
        if turn_number % 3 == 0:
            return ""
        return self.TRANSITIONS[turn_number % len(self.TRANSITIONS)]


class FeedbackAgent:
    """
    Generates user-facing feedback with varied structures.
    Uses .with_structured_output() for reliable parsing.
    """
    
    def __init__(
        self, fast_llm: BaseChatModel, concept_tool,
        cache_store: InterviewCacheStore
    ):
        self.structured_chain = (
            FEEDBACK_PROMPT
            | fast_llm.with_structured_output(FeedbackComponents)
        )
        self.concept_tool = concept_tool
        self.cache_store = cache_store
        self.composer = FeedbackComposer()
    
    async def execute(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        score = eval_data["overall_score"]
        turn = state["question_count"]
        
        concept_context = await self._get_concept_context(state, eval_data, score)
        
        components = await self.structured_chain.ainvoke({
            "question": state["current_question"]["text"],
            "response": state["candidate_response"],
            "score_band": self.composer._get_score_band(score),
            "tone_guidance": self._get_tone_guidance(score),
            "concept_context": concept_context,
        }, config=config)
        
        previous_structures = state.get("previous_feedback_structures", [])
        feedback_text = self.composer.compose(components, score, turn, previous_structures)
        
        score_band = self.composer._get_score_band(score)
        structures = self.composer.STRUCTURES[score_band]
        used_structure = structures[turn % len(structures)]
        
        return {
            "current_feedback": feedback_text,
            # operator.add reducer: return only the NEW structure
            "previous_feedback_structures": [used_structure],
        }
    
    async def _get_concept_context(
        self, state: InterviewState, eval_data: dict, score: float
    ) -> str:
        if score >= 7.0:
            return ""
        missed = eval_data.get("key_points_missed", [])[:1]
        for concept in missed:
            cached = await self.cache_store.get_concept(state["interview_id"], concept)
            if cached:
                return cached.get("simple_explanation", "")
            data = await self.concept_tool.ainvoke(concept)
            if data:
                await self.cache_store.set_concept(state["interview_id"], concept, data)
                return data.get("simple_explanation", "")
        return ""
    
    def _get_tone_guidance(self, score: float) -> str:
        if score >= 8.0:
            return "Brief, genuine acknowledgment. No need to elaborate."
        elif score >= 6.0:
            return "Encouraging but direct. Hint at depth opportunity."
        elif score >= 4.0:
            return "Supportive. Focus on the attempt, guide gently."
        return "Patient. No praise openers. Direct but kind."
```

---

## Question Selector Agent (Topic Injection, Dynamic Rubric, Atomic Select)

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig


FOLLOW_UP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Generate a targeted follow-up question that probes the missed concepts."),
    ("human", (
        "Original question: {original_question}\n"
        "Candidate response: {candidate_response}\n"
        "Missed points: {missed_points}\n"
        "Topic: {topic}"
    ))
])

CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Generate a clarification question that helps the candidate "
     "recognize and correct their misconception."),
    ("human", (
        "Original question: {original_question}\n"
        "Candidate response: {candidate_response}\n"
        "Misconception: {misconception}"
    ))
])

REACT_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Select the best question for the interview context. "
     "Return JSON with selected_id."),
    ("human", (
        "Candidates: {candidates}\n"
        "Difficulty: {difficulty_level}\n"
        "Recent topics: {topics_covered}\n"
        "Performance trend: {performance_trend}"
    ))
])


class QuestionSelectorAgent:
    """
    Owns ALL question decisions and topic tracking.
    Delegates CRAG to AgenticRAGService.
    
    Key fixes:
    - Injects topic into every question dict (so Evaluator and _select_weakest_topic work)
    - Generates target_concepts alongside follow-up/clarify Qs (dynamic rubric)
    - Uses atomic select_and_mark via cache_store.select_and_mark()
    """
    
    MAX_FOLLOW_UPS = 2
    FUNDAMENTAL_TOPICS = ["ml_fundamentals", "statistics", "evaluation"]
    
    def __init__(
        self, rag_service: AgenticRAGService,
        fast_llm: BaseChatModel, complex_llm: BaseChatModel,
        cache_store: InterviewCacheStore
    ):
        self.rag = rag_service
        self.fast_llm = fast_llm
        self.complex_llm = complex_llm
        self.cache_store = cache_store
        self.followup_chain = FOLLOW_UP_PROMPT | complex_llm
        self.clarify_chain = CLARIFICATION_PROMPT | complex_llm
        self.select_chain = (
            REACT_SELECTION_PROMPT
            | fast_llm  # Returns JSON with selected_id
        )
    
    async def execute(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        remaining_time = self._get_remaining_minutes(state)
        mode = self._determine_question_mode(state, remaining_time)
        
        if mode == "retrieve":
            question, selected_topic = await self._retrieve_question(
                state, remaining_time, config
            )
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": 0,
                "conversation_thread": [question["id"]],  # operator.add: new items only
                "topics_covered": [selected_topic],        # operator.add: new items only
            }
        elif mode == "follow_up":
            question = await self._generate_follow_up(state, config)
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": state.get("follow_up_count", 0) + 1,
                "conversation_thread": [question["id"]],   # operator.add
                "topics_covered": [],                       # No new topic
            }
        elif mode == "clarify":
            question = await self._generate_clarification(state, config)
            return {
                "current_question": question,
                "question_mode": mode,
                "follow_up_count": state.get("follow_up_count", 0) + 1,
                "conversation_thread": [question["id"]],   # operator.add
                "topics_covered": [],                       # No new topic
            }
    
    def _get_remaining_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return state.get("time_budget_minutes", 30)
        elapsed = (datetime.now() - state["interview_start_time"]).total_seconds() / 60
        return max(0, state["time_budget_minutes"] - elapsed)
    
    # ─────────────────────────────────────────────────────────────────
    # MODE DETERMINATION
    # ─────────────────────────────────────────────────────────────────
    
    def _determine_question_mode(
        self, state: InterviewState, remaining_time: float
    ) -> str:
        if state["question_count"] == 0:
            return "retrieve"
        if remaining_time < 5:
            return "retrieve"
        
        eval_data = state["current_evaluation"]
        score = eval_data["overall_score"]
        missed = eval_data.get("key_points_missed", [])
        misconceptions = eval_data.get("misconceptions", [])
        follow_ups = state.get("follow_up_count", 0)
        
        if misconceptions and follow_ups < self.MAX_FOLLOW_UPS:
            return "clarify"
        if score < 7.0 and missed and follow_ups < self.MAX_FOLLOW_UPS:
            return "follow_up"
        if 7.0 <= score < 8.0 and missed and follow_ups < 1:
            return "follow_up"
        return "retrieve"
    
    # ─────────────────────────────────────────────────────────────────
    # RETRIEVE MODE (Atomic select+mark, delegates CRAG to RAG service)
    # ─────────────────────────────────────────────────────────────────
    
    async def _retrieve_question(
        self, state: InterviewState, remaining_time: float,
        config: RunnableConfig
    ) -> tuple[dict, str]:
        topic = self._get_next_topic_from_plan(state)
        difficulty = state["difficulty_level"]
        session_id = state["interview_id"]
        
        # Atomic select+mark: eliminates TOCTOU race
        async def _select_fn(candidates: list[dict]) -> dict:
            return await self._react_select(candidates, state, config)
        
        selected = await self.cache_store.select_and_mark(
            session_id=session_id,
            topic=topic,
            difficulty=difficulty,
            selector_fn=_select_fn
        )
        
        if selected is None:
            # Cache miss — retrieve via CRAG
            crag_result = await self.rag.retrieve_with_crag(
                topic=topic, difficulty=difficulty,
                exclude_ids=self._get_used_question_ids(state),
                remaining_time=remaining_time,
            )
            
            await self.cache_store.set_topic_questions(
                session_id=session_id, topic=topic,
                difficulty=difficulty, questions=crag_result.candidates,
                crag_grade=crag_result.grade,
            )
            
            # Now select from fresh cache (atomic)
            selected = await self.cache_store.select_and_mark(
                session_id=session_id, topic=topic,
                difficulty=difficulty, selector_fn=_select_fn
            )
            
            if selected is None and crag_result.candidates:
                selected = crag_result.candidates[0]
            elif selected is None:
                selected = self._get_fallback_question()
        
        selected["question_type"] = "retrieved"
        selected["topic"] = topic  # Ensure topic is always present
        return selected, topic
    
    async def _react_select(
        self, candidates: list[dict], state: InterviewState,
        config: RunnableConfig
    ) -> dict:
        if not candidates:
            return self._get_fallback_question()
        if len(candidates) == 1:
            return candidates[0]
        
        response = await self.select_chain.ainvoke({
            "candidates": [
                {"id": c["id"], "text": c["text"], "difficulty": c.get("difficulty", "")}
                for c in candidates[:5]
            ],
            "difficulty_level": state["difficulty_level"],
            "topics_covered": state["topics_covered"][-3:],
            "performance_trend": self._get_performance_trend(state),
        }, config=config)
        
        selected_id = parse_json_safely(response.content).get("selected_id", "")
        for c in candidates:
            if c["id"] == selected_id:
                return c
        return candidates[0]
    
    # ─────────────────────────────────────────────────────────────────
    # TOPIC SELECTION
    # ─────────────────────────────────────────────────────────────────
    
    def _get_next_topic_from_plan(self, state: InterviewState) -> str:
        plan = state.get("interview_plan", {})
        topic_sequence = plan.get("topic_sequence", [])
        topics_covered = state.get("topics_covered", [])
        
        remaining = [t for t in topic_sequence if t not in topics_covered]
        if not remaining:
            return self._select_weakest_topic(state)
        
        if state.get("difficulty_reduced_due_to_performance"):
            fundamentals = [t for t in remaining if t in self.FUNDAMENTAL_TOPICS]
            others = [t for t in remaining if t not in self.FUNDAMENTAL_TOPICS]
            remaining = fundamentals + others
        
        return remaining[0]
    
    def _select_weakest_topic(self, state: InterviewState) -> str:
        """
        Select topic where candidate struggled most.
        Uses evaluation["topic"] field — Evaluator injects this from current_question.
        """
        performance_by_topic = {}
        for ev in state.get("all_evaluations", []):
            topic = ev.get("topic", "general")
            if topic == "unknown":  # Skip fallback evaluations
                continue
            if topic not in performance_by_topic:
                performance_by_topic[topic] = []
            performance_by_topic[topic].append(ev["overall_score"])
        
        topic_avgs = {t: sum(s)/len(s) for t, s in performance_by_topic.items()}
        if topic_avgs:
            return min(topic_avgs.keys(), key=lambda t: topic_avgs[t])
        return "ml_fundamentals"
    
    # ─────────────────────────────────────────────────────────────────
    # FOLLOW_UP MODE (14B, with target_concepts for dynamic rubric)
    # ─────────────────────────────────────────────────────────────────
    
    async def _generate_follow_up(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        original = state["current_question"]
        missed = eval_data.get("key_points_missed", [])[:2]
        
        response = await self.followup_chain.ainvoke({
            "original_question": original["text"],
            "candidate_response": state["candidate_response"],
            "missed_points": missed,
            "topic": original.get("topic", "general"),
        }, config=config)
        
        return {
            "id": f"{original['id']}_followup_{state.get('follow_up_count', 0) + 1}",
            "text": response.content.strip(),
            "question_type": "follow_up",
            "topic": original.get("topic", "general"),  # Always inject topic
            "difficulty": original.get("difficulty", "medium"),
            "parent_question_id": original["id"],
            "target_concepts": missed,  # Dynamic rubric for drift detection
            "estimated_time_minutes": 3
        }
    
    # ─────────────────────────────────────────────────────────────────
    # CLARIFY MODE (14B, with target_misconception for dynamic rubric)
    # ─────────────────────────────────────────────────────────────────
    
    async def _generate_clarification(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        eval_data = state["current_evaluation"]
        misconception = eval_data["misconceptions"][0]
        original = state["current_question"]
        
        response = await self.clarify_chain.ainvoke({
            "original_question": original["text"],
            "candidate_response": state["candidate_response"],
            "misconception": misconception,
        }, config=config)
        
        return {
            "id": f"{original['id']}_clarify",
            "text": response.content.strip(),
            "question_type": "clarification",
            "topic": original.get("topic", "general"),  # Always inject topic
            "difficulty": original.get("difficulty", "medium"),
            "parent_question_id": original["id"],
            "target_misconception": misconception,
            "target_concepts": [misconception],  # Dynamic rubric
            "estimated_time_minutes": 3
        }
    
    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    
    def _get_used_question_ids(self, state: InterviewState) -> list[str]:
        return [
            ev.get("question_id", "")
            for ev in state.get("all_evaluations", [])
            if ev.get("question_id")
        ]
    
    def _get_performance_trend(self, state: InterviewState) -> str:
        trajectory = state.get("performance_trajectory", [])
        if len(trajectory) < 3:
            return "stable"
        ema = state.get("ema_trajectory", trajectory)
        if len(ema) >= 4 and ema[-1] - ema[-4] > 0.8:
            return "improving"
        elif len(ema) >= 4 and ema[-1] - ema[-4] < -0.8:
            return "declining"
        return "stable"
    
    def _get_fallback_question(self) -> dict:
        return {
            "id": "fallback_001",
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4,
            "target_concepts": ["bias", "variance", "tradeoff"]
        }
```

---

## Evaluator Agent (Topic Injection)

```python
class EvaluatorAgent:
    """
    Evaluates candidate responses with CoT + Reflection.
    
    Key fix: Injects topic from current_question into evaluation output.
    Without this, _select_weakest_topic and per-topic scoring silently fail.
    """
    
    def __init__(self, complex_llm: BaseChatModel, fast_llm: BaseChatModel):
        self.eval_chain = EVAL_COT_PROMPT | complex_llm.with_structured_output(EvaluationOutput)
        self.reflect_chain = REFLECTION_PROMPT | fast_llm
    
    async def execute(
        self, state: InterviewState, config: RunnableConfig
    ) -> dict:
        question = state["current_question"]
        
        # CoT evaluation
        evaluation = await self.eval_chain.ainvoke({
            "question": question["text"],
            "response": state["candidate_response"],
            "rubric": question.get("rubric", {}),
            "target_concepts": question.get("target_concepts", []),
        }, config=config)
        
        eval_dict = evaluation.model_dump()
        
        # INJECT TOPIC from current_question — critical for per-topic tracking
        eval_dict["topic"] = question.get("topic", "general")
        eval_dict["question_id"] = question.get("id", "")
        
        # Reflection step
        reflection = await self.reflect_chain.ainvoke({
            "evaluation": eval_dict,
            "question": question["text"],
            "response": state["candidate_response"],
        }, config=config)
        
        # Apply reflection adjustments if any
        eval_dict = self._apply_reflection(eval_dict, reflection.content)
        
        return {"current_evaluation": eval_dict}
    
    def _apply_reflection(self, eval_dict: dict, reflection: str) -> dict:
        # ... reflection adjustment logic ...
        return eval_dict
```

---

## LangGraph Workflow (Summarization as Node, Async Checkpointer)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # Production
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver    # Development


def build_interview_graph(
    agents: "AgentRegistry", checkpointer
) -> StateGraph:
    """
    Parallel execution with summarization as a checkpointed graph node.
    
    Flow: Evaluator → (Feedback || QS) → Supervisor → MaybeSummarize → END
    
    Key isolation ensures no state conflicts during parallel execution.
    Summarization runs after supervisor_check — if no update needed,
    returns empty dict (no-op).
    """
    graph = StateGraph(InterviewState)
    
    graph.add_node("evaluator", agents.evaluator.execute)
    graph.add_node("feedback", agents.feedback.execute)
    graph.add_node("question_selector", agents.question_selector.execute)
    graph.add_node("supervisor_check", agents.supervisor.validate_and_decide)
    graph.add_node("maybe_summarize", agents.conversation_manager.maybe_update_summary)
    
    graph.set_entry_point("evaluator")
    
    # Fan-out: Both depend on evaluator, run in parallel
    graph.add_edge("evaluator", "feedback")
    graph.add_edge("evaluator", "question_selector")
    
    # Fan-in: Supervisor waits for both
    graph.add_edge("feedback", "supervisor_check")
    graph.add_edge("question_selector", "supervisor_check")
    
    # Summarization after supervisor (checkpointed)
    graph.add_edge("supervisor_check", "maybe_summarize")
    graph.add_edge("maybe_summarize", END)
    
    return graph.compile(
        checkpointer=checkpointer,
        # interrupt_before=["supervisor_check"],  # Uncomment for admin review mode
    )


def _wrap_with_timeout(agent_fn, timeout_seconds: float = 15.0):
    """Wrap an agent's execute method with asyncio.wait_for() timeout.
    TimeoutError is caught by the circuit breaker → fallback."""
    async def wrapped(state: InterviewState, config: RunnableConfig):
        return await asyncio.wait_for(
            agent_fn(state, config), timeout=timeout_seconds
        )
    return wrapped


def build_start_graph(agents: "AgentRegistry") -> StateGraph:
    """
    /start: Plan creation → first question retrieval.
    Supervisor only creates plan (no RAG dependency).
    QS handles first topic retrieval via its normal path.
    Background pre-warming triggered by API layer.
    """
    graph = StateGraph(InterviewState)
    
    graph.add_node("create_plan", agents.supervisor.create_interview_plan)
    graph.add_node("first_question", agents.question_selector.execute)
    
    graph.set_entry_point("create_plan")
    graph.add_edge("create_plan", "first_question")
    graph.add_edge("first_question", END)
    
    return graph.compile()
```

---

## API Design (Lifespan, RunnableConfig, Streaming, Checkpointer)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from datetime import datetime
import asyncio
import json


# ─────────────────────────────────────────────────────────────────
# APPLICATION LIFECYCLE (lifespan, not deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Proper ASGI lifecycle:
    - Initialize async checkpointer
    - Compile graphs
    - Start periodic cleanup
    - Clean shutdown with task cancellation
    """
    # Production: AsyncPostgresSaver; Dev: AsyncSqliteSaver
    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        app.state.checkpointer = checkpointer
        app.state.interview_graph = build_interview_graph(agents, checkpointer)
        app.state.start_graph = build_start_graph(agents)
        
        # Periodic cleanup — raw asyncio.create_task is acceptable here
        # because lifespan context manager handles cancellation on shutdown
        cleanup_task = asyncio.create_task(_periodic_cleanup())
        
        yield
        
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


async def _periodic_cleanup():
    while True:
        await asyncio.sleep(900)
        cleaned = await cache_store.cleanup_abandoned_sessions()
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} abandoned session(s)")


app = FastAPI(lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    user_id: str
    difficulty: str = "medium"
    focus_topics: list[str] = []
    time_budget_minutes: int = 30

class StartResponse(BaseModel):
    session_id: str
    question: dict
    time_budget_minutes: int
    target_questions: int

class SubmitRequest(BaseModel):
    session_id: str
    response: str

class SubmitResponse(BaseModel):
    feedback: str
    next_question: dict | None
    progress: ProgressInfo
    continue_interview: bool

class ProgressInfo(BaseModel):
    questions_completed: int
    time_elapsed_minutes: float
    time_remaining_minutes: float

class FinalReport(BaseModel):
    overall_score: float
    adjusted_score: float
    questions_asked: int
    time_taken_minutes: float
    difficulty_progression: list[str]
    topic_scores: dict[str, float]
    strengths: list[str]
    areas_for_improvement: list[str]
    performance_notes: list[str]
    fallback_count: int
    detailed_evaluations: list[dict]


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/interview/start")
async def start_interview(
    request: StartRequest, background_tasks: BackgroundTasks
) -> StartResponse:
    state = initialize_state(request)
    
    config = RunnableConfig(
        configurable={"thread_id": state["interview_id"]},
        metadata={"user_id": request.user_id, "turn": 0},
        tags=["interview", "start"],
    )
    
    result = await app.state.start_graph.ainvoke(state, config=config)
    
    # Background pre-warming (RAG service runs full CRAG → real grades)
    plan = result["interview_plan"]
    remaining_topics = plan["topic_sequence"][1:5]
    if remaining_topics:
        background_tasks.add_task(
            cache_store.pre_warm_topics_background,
            session_id=result["interview_id"],
            rag_service=rag_service,
            topics=remaining_topics,
            difficulty=result["difficulty_level"]
        )
    
    return StartResponse(
        session_id=result["interview_id"],
        question={
            "text": result["current_question"]["text"],
            "topic": result["current_question"]["topic"],
            "estimated_time_minutes": result["current_question"]["estimated_time_minutes"]
        },
        time_budget_minutes=result["time_budget_minutes"],
        target_questions=len(plan["topic_sequence"])
    )


@app.post("/api/v1/interview/submit_response")
async def submit_response(request: SubmitRequest) -> SubmitResponse:
    """
    Non-streaming endpoint. Checkpointer handles state persistence
    automatically via thread_id — no manual save_state needed.
    """
    config = RunnableConfig(
        configurable={"thread_id": request.session_id},
        metadata={"session_id": request.session_id},
        tags=["interview", "submit"],
    )
    
    # Checkpointer loads last state automatically via thread_id
    # Only pass new input — graph resumes from checkpoint
    result = await app.state.interview_graph.ainvoke(
        {"candidate_response": request.response},
        config=config
    )
    
    elapsed = (datetime.now() - result["interview_start_time"]).total_seconds() / 60
    remaining = result["time_budget_minutes"] - elapsed
    
    return SubmitResponse(
        feedback=result["current_feedback"],
        next_question={
            "text": result["current_question"]["text"],
            "topic": result["current_question"]["topic"],
            "estimated_time_minutes": result["current_question"]["estimated_time_minutes"]
        } if result["should_continue"] else None,
        progress=ProgressInfo(
            questions_completed=result["question_count"],
            time_elapsed_minutes=round(elapsed, 1),
            time_remaining_minutes=round(max(0, remaining), 1)
        ),
        continue_interview=result["should_continue"]
    )


@app.post("/api/v1/interview/submit_response/stream")
async def submit_response_stream(request: SubmitRequest):
    """
    SSE streaming endpoint. Streams feedback tokens as they're generated,
    then sends final structured response.
    """
    config = RunnableConfig(
        configurable={"thread_id": request.session_id},
        metadata={"session_id": request.session_id},
        tags=["interview", "submit", "stream"],
    )
    
    async def event_generator():
        async for event in app.state.interview_graph.astream_events(
            {"candidate_response": request.response},
            config=config,
            version="v2"
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
            
            elif event["event"] == "on_chain_end" and event["name"] == "feedback":
                yield f"data: {json.dumps({'type': 'feedback_complete'})}\n\n"
            
            elif event["event"] == "on_chain_end" and event["name"] == "maybe_summarize":
                result = event["data"]["output"]
                yield f"data: {json.dumps({'type': 'turn_complete', 'data': _format_response(result)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/interview/end")
async def end_interview(session_id: str) -> dict:
    # Load final state from checkpointer
    config = RunnableConfig(configurable={"thread_id": session_id})
    state = await _get_state_from_checkpointer(session_id)
    
    report = generate_final_report(state)
    await cache_store.clear_session(session_id)
    
    return {"final_report": report}


def generate_final_report(state: InterviewState) -> FinalReport:
    all_evaluations = state.get("all_evaluations", [])
    difficulties = state.get("difficulty_history", [])
    
    real_evals = [(e, d) for e, d in zip(all_evaluations, difficulties)
                  if not e.get("is_fallback", False)]
    fallback_count = len(all_evaluations) - len(real_evals)
    
    raw_scores = [e["overall_score"] for e, d in real_evals]
    raw_avg = sum(raw_scores) / len(raw_scores) if raw_scores else 0
    
    weights = {"easy": 0.7, "medium": 1.0, "hard": 1.3}
    weighted_sum = sum(e["overall_score"] * weights.get(d, 1.0) for e, d in real_evals)
    max_possible = sum(10 * weights.get(d, 1.0) for e, d in real_evals)
    adjusted = (weighted_sum / max_possible) * 10 if max_possible > 0 else 0
    
    # Topic aggregation uses evaluation["topic"] field
    topic_scores = {}
    for e, d in real_evals:
        topic = e.get("topic", "general")
        if topic == "unknown":
            continue
        if topic not in topic_scores:
            topic_scores[topic] = []
        topic_scores[topic].append(e["overall_score"])
    topic_avgs = {t: sum(s)/len(s) for t, s in topic_scores.items()}
    
    notes = []
    if state.get("difficulty_reduced_due_to_performance"):
        original = state.get("original_difficulty", "medium")
        current = state["difficulty_level"]
        notes.append(f"Difficulty reduced from {original} to {current} due to performance")
    if fallback_count > 0:
        notes.append(f"{fallback_count} question(s) could not be evaluated (excluded from scoring)")
    
    elapsed = (datetime.now() - state["interview_start_time"]).total_seconds() / 60
    
    return FinalReport(
        overall_score=round(raw_avg, 1),
        adjusted_score=round(adjusted, 1),
        questions_asked=state["question_count"],
        time_taken_minutes=round(elapsed, 1),
        difficulty_progression=difficulties,
        topic_scores={t: round(s, 1) for t, s in topic_avgs.items()},
        strengths=[t for t, s in topic_avgs.items() if s >= 7.0],
        areas_for_improvement=[t for t, s in topic_avgs.items() if s < 6.0],
        performance_notes=notes,
        fallback_count=fallback_count,
        detailed_evaluations=all_evaluations
    )
```

---

## Follow-up Loop Rules

| Condition | Mode | Max | Exit |
|-----------|------|-----|------|
| First question | RETRIEVE | — | — |
| Time < 5 min | RETRIEVE | — | Skip follow-ups |
| Misconception detected | CLARIFY | 2 | Resolved or max |
| Score < 7.0 + gaps | FOLLOW_UP | 2 | Score ≥ 8 or max |
| Score 7.0-8.0 + gaps | FOLLOW_UP | 1 | Score ≥ 8 or max |
| Score ≥ 8.0 | RETRIEVE | — | Always new topic |

---

## Latency Summary

| Stage | Model | Calls | Latency |
|-------|-------|-------|---------|
| /start: Plan | 14B | 1 | ~2-3s |
| /start: First topic + Q (incl. CRAG) | 7B | 2 | ~2s |
| /start: Background pre-warm (incl. CRAG) | — | — | Non-blocking |
| **Total /start** | | 3 | **~4-5s** |
| | | | |
| Evaluator (CoT + Reflect) | 14B + 7B | 2 | ~2.5-3.5s |
| Feedback (parallel) | 7B | 1 | ~1-1.5s |
| Question Selector (parallel, cache hit) | 7B | 1 | ~1.5-2s |
| Question Selector (parallel, cache miss + CRAG) | 7B | 2-3 | ~2-2.5s |
| Summarization (every 3 turns, graph node) | 14B | 0.33 avg | ~0.7s avg |
| **Total /submit** | | 3-4 | **~3.5-5s** |

---

## Component Dependency Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT DEPENDENCIES                                │
│                                                                          │
│  AgenticRAGService                                                      │
│  ├── VectorRetriever          (ChromaDB queries)                        │
│  ├── DocumentGrader           (7B batch grading, .with_structured_output)│
│  └── QueryRefiner             (corrective query refinement)             │
│                                                                          │
│  QuestionSelectorAgent                                                  │
│  ├── AgenticRAGService        (retrieve_with_crag on cache miss)        │
│  ├── InterviewCacheStore      (atomic select_and_mark)                  │
│  ├── fast_llm (7B)            (ReAct selection, ChatPromptTemplate)     │
│  └── complex_llm (14B)        (follow-up/clarify, ChatPromptTemplate)   │
│                                                                          │
│  EvaluatorAgent                                                         │
│  ├── complex_llm (14B)        (.with_structured_output(EvaluationOutput))│
│  ├── fast_llm (7B)            (reflection)                              │
│  └── LangChain @tools         (rubric_lookup, code_validator)           │
│                                                                          │
│  FeedbackAgent                                                          │
│  ├── fast_llm (7B)            (.with_structured_output(FeedbackComponents))│
│  ├── concept_lookup (@tool)   (concept enrichment)                      │
│  └── InterviewCacheStore      (concept cache pool)                      │
│                                                                          │
│  SupervisorAgent (Decoupled from RAG)                                   │
│  ├── complex_llm (14B)        (plan creation, ChatPromptTemplate)       │
│  ├── TrendAnalyzer            (EMA smoothing, α=0.3, threshold 7.5)    │
│  ├── ValidationGateRegistry   (output validation, dynamic rubric aware) │
│  └── CircuitBreaker           (retry management)                        │
│                                                                          │
│  ConversationManager (Graph node, not post-hoc)                         │
│  └── complex_llm (14B)        (summarization, ChatPromptTemplate)       │
│                                                                          │
│  InterviewCacheStore                                                    │
│  └── AgenticRAGService        (pre-warm via retrieve_batch)             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Cache Strategy Summary

| Cache Type | Pool | Key Pattern | TTL | Max/Session | Used By |
|------------|------|-------------|-----|-------------|---------|
| Topic Questions | Topic Pool | `{topic}:{difficulty}` | Grade-based (5-30 min) | 10 | Question Selector |
| Concepts | Concept Pool | `{concept_name}` | 60 min (stable data) | 30 | Feedback Agent |

| Feature | Implementation |
|---------|----------------|
| Separate Pools | Topic and Concept caches evict independently (no cross-eviction) |
| Per-Session Locks | `asyncio.Lock` per session (not one global lock) |
| Atomic Select+Mark | `select_and_mark()` eliminates TOCTOU race on question selection |
| Real Grades | All entries get REAL CRAG grades from AgenticRAGService |
| Partial Reuse | `used_ids` tracks consumed questions from batch |
| Pool-Isolated LRU | Topics: 10 max, Concepts: 30 max (per session) |
| Background Pre-warming | Via FastAPI BackgroundTasks; runs full CRAG → accurate TTLs |
| Abandoned Session Cleanup | Periodic sweep every 15 min, removes sessions > 90 min old |

---

## Key Design Decisions Summary

| Decision | Approach | Rationale |
|----------|----------|-----------|
| LLM interface | `BaseChatModel` + `.ainvoke()` + `ChatPromptTemplate` + `with_structured_output()` throughout | Production LangChain pattern; enables tracing, composability, reliable parsing |
| State reducers | Explicit `operator.add` for lists, `last_value` for scalars, `add_messages` for messages | Prevents silent state overwrites during parallel fan-out; agents return only new items |
| Checkpointer | `AsyncPostgresSaver` (prod) / `AsyncSqliteSaver` (dev), loaded via lifespan | Async checkpointing; no event loop blocking; proper lifecycle management |
| Thread-based state | `thread_id = session_id` in `RunnableConfig`; checkpointer auto-saves/loads | Eliminates manual `save_state`/`load_state`; crash-safe state persistence |
| RunnableConfig propagation | Every graph invocation and agent call passes config with thread_id, metadata, tags | Full LangSmith/LangFuse trace visibility per session and turn |
| Streaming | `astream_events()` with SSE endpoint for real-time feedback | Frontend shows feedback streaming while QS works in parallel |
| Summarization | LangGraph node after supervisor_check (not post-hoc outside graph) | Checkpointed like all other nodes; crash-safe; no manual save_state gap |
| Tools | LangChain `@tool` decorator for rubric_lookup, code_validator, concept_lookup | Automatic tracing, schema validation, composable with `bind_tools()` |
| question_count ownership | Supervisor increments in `validate_and_decide` after fan-in | Single owner; prevents silent zero-forever bug |
| Difficulty curve indexing | `len(topics_covered)` not `question_count` | Curve has one entry per topic; follow-ups don't advance the index |
| CRAG ownership | AgenticRAGService owns full CRAG loop; QS calls `retrieve_with_crag()` | Clean separation: QS owns selection, RAG service owns retrieval quality |
| RAG service internals | LangGraph StateGraph subgraph with conditional edges | Tracing, visualization, composability; compiled once, concurrent-safe |
| Supervisor decoupled from RAG | Supervisor only creates plan; QS handles all retrieval including first topic | Cleaner separation; Supervisor has no RAG dependency |
| Per-session cache locks | `defaultdict(asyncio.Lock)` per session, not a single global lock | Concurrent interviews don't block each other |
| Atomic select+mark | `select_and_mark()` runs selector inside lock | Eliminates TOCTOU race between get and mark |
| Dynamic rubric | QS generates `target_concepts` alongside follow-up/clarify Qs | Drift detection works for all question types, not just retrieved |
| Topic in evaluation | Evaluator injects `topic` from `current_question` into output | Per-topic scoring and `_select_weakest_topic` actually work |
| EMA threshold | 7.5 (lowered from 8.0) for difficulty increase trigger | Consistent 7.5 scorers are legitimately strong; deserve harder questions |
| No early termination | Difficulty reduction + weighted final score | Early termination feels punitive |
| EMA fallback protection | Fallback scores flagged `is_fallback=True`, excluded from EMA | Prevents circuit breaker failures from tanking performance trends |
| Topic re-prioritization | Move fundamentals first when struggling (rule-based) | Foundational questions before advanced topics for struggling candidates |
| Rolling conversation window | 3 recent turns full + older turns summarized; sentence-boundary truncation | Bounds context to ~1700 tokens; clean truncation for code-heavy answers |
| Varied feedback | 4 templates per band (high increased from 2) + turn-based rotation | Avoids repetitive Mad Libs feel at all score bands |
| Query refiner signature | `refine(topic, attempt)` — dropped redundant `query` param | Query was always identical to topic in all call sites |
| Pre-warming grades | Pre-warming calls `retrieve_batch()` → full CRAG loop → real grades | Accurate TTLs from the start; no overwrite needed |
| Graceful degradation | `configurable_alternatives()` + `ConfigurableField(id="model_tier")` | LangChain-native model routing; traceable, testable, runtime-switchable |
| Chain retry policy | `.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)` on all chains | LangChain 1.0 standard; recovers from transient Ollama failures |
| Structured output fallback | `.with_fallbacks([lenient_parser])` on Evaluator and QS chains | Graceful degradation when JSON mode fails; partial recovery vs full rejection |
| Agent node timeouts | `asyncio.wait_for(timeout=15.0)` wrapper on each graph node | Prevents Ollama hangs from blocking interview indefinitely |
| Inter-agent contracts | Pydantic models (`EvaluationOutput`, `FeedbackOutput`, `QuestionOutput`) | Catches schema violations at gate time, not at downstream consumption |
| Turn counting | `sum(1 for m in messages if isinstance(m, HumanMessage))` | Robust against system messages, metadata messages in `add_messages` list |
| CRAG subgraph checkpointing | Invoked imperatively (not registered as subgraph node) | Accepted tradeoff: CRAG is fast (~2s) and idempotent; re-running on crash recovery is cheaper than subgraph state transformation complexity |
| Human-in-the-loop readiness | `interrupt_before=["supervisor_check"]` available but commented out | Architecture supports admin review; activate when needed without refactoring |