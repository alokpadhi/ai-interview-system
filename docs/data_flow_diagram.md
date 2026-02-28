# Data Flow Diagram — AI Interview System

> Source of truth: [architecture(v2).md](file:///home/alokpadhi/ai-interview-system/docs/architecture(v2).md)

---

## 1. High-Level Request Lifecycle

```
                        ┌─────────────────┐
                        │    Frontend /    │
                        │    API Client    │
                        └────────┬────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            │                    │                    │
            ▼                    ▼                    ▼
     POST /start        POST /submit_response    POST /end
            │                    │                    │
            ▼                    ▼                    ▼
   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
   │  Start Graph   │  │ Interview Graph│  │ Report Builder │
   │  (Plan + 1st Q)│  │  (Main Loop)  │  │  (Final Score) │
   └────────────────┘  └────────────────┘  └────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
     JSON Response       JSON (or SSE)         JSON Report
   { plan, question }  { feedback, question } { scores, topics }
```

---

## 2. POST /start — Interview Initialization

```
Client Request
│  { user_id }                                        ◄── required
│  { difficulty: "medium" }                           ◄── optional, default "medium"
│  { focus_topics: [] }                               ◄── optional, default [] (all topics)
│  { time_budget_minutes: 30 }                        ◄── optional, default 30
│
▼
┌──────────────────────────────────────────────────────────────┐
│  API LAYER (FastAPI)                                         │
│                                                              │
│  1. Create interview_id (UUID)                               │
│  2. Build RunnableConfig { thread_id: interview_id }         │
│  3. Initialize InterviewState                                │
│  4. Invoke start_graph                                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  START GRAPH (2 nodes, sequential)                           │
│                                                              │
│  ┌─────────────────────────┐                                 │
│  │  supervisor.create_     │   14B LLM call                  │
│  │  interview_plan()       │   ──────────────────────────►   │
│  │                         │   plan = {                      │
│  │  Input:                 │     topic_sequence,             │
│  │  • difficulty           │     difficulty_curve,           │
│  │  • focus_topics         │     target_questions            │
│  │  • time_budget          │   }                             │
│  └────────────┬────────────┘                                 │
│               │                                              │
│               │  State writes:                               │
│               │  ├─ interview_plan (dict)                     │
│               │  ├─ difficulty_level = curve[0]               │
│               │  ├─ original_difficulty                      │
│               │  └─ stage = "questioning"                    │
│               │                                              │
│               ▼                                              │
│  ┌─────────────────────────┐                                 │
│  │  question_selector.     │                                 │
│  │  execute()              │                                 │
│  │                         │   ┌───────────────────────────┐ │
│  │  Mode: RETRIEVE         │   │  Cache hit?               │ │
│  │  (always for 1st Q)     │──►│  No → CRAG subgraph       │ │
│  │                         │   │       (ChromaDB + 7B)     │ │
│  │                         │   │  Yes → atomic select_and_ │ │
│  │                         │   │        mark from cache    │ │
│  │                         │   └───────────────────────────┘ │
│  │                         │                                 │
│  │  State writes:          │   7B LLM call (selection)       │
│  │  ├─ current_question    │   ──────────────────────────►   │
│  │  ├─ question_mode       │   QuestionSelection {           │
│  │  ├─ follow_up_count = 0 │     selected_id, reasoning     │
│  │  ├─ topics_covered [+1] │   }                             │
│  │  └─ conversation_thread │                                 │
│  └─────────────────────────┘                                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  BACKGROUND TASK (FastAPI BackgroundTasks)                    │
│                                                              │
│  Pre-warm cache for remaining topics in plan:                │
│  for topic in plan.topic_sequence[1:]:                        │
│      cache_store.pre_warm(topic, difficulty)                  │
│          └─► AgenticRAGService.retrieve_batch()               │
│              └─► CRAG subgraph (full loop, real grades)       │
│                  └─► Cache stores with grade-based TTL        │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
Response to Client:
{
  interview_id, plan_summary,
  question: { id, text, topic, difficulty }
}
```

---

## 3. POST /submit_response — Main Interview Loop

```
Client Request
│  { interview_id, candidate_response }
│
▼
┌──────────────────────────────────────────────────────────────┐
│  API LAYER                                                   │
│                                                              │
│  1. Load state via checkpointer (thread_id = interview_id)   │
│  2. Set state["candidate_response"]                          │
│  3. Append HumanMessage to state["messages"]                 │
│  4. Invoke interview_graph                                   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  INTERVIEW GRAPH (LangGraph StateGraph)                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  NODE 1: EVALUATOR                                    14B + 7B    │  │
│  │                                                                    │  │
│  │  Input from state:                                                 │  │
│  │  ├─ current_question.text                                          │  │
│  │  ├─ candidate_response                                             │  │
│  │  ├─ current_question.rubric (or .target_concepts)                  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  Standard Mode (consistency_samples=1):                     │   │  │
│  │  │                                                             │   │  │
│  │  │  candidate_response ──► CoT Chain (14B) ──► EvaluationOutput│   │  │
│  │  │                                  │                          │   │  │
│  │  │                                  ▼                          │   │  │
│  │  │                         Reflection (7B)                     │   │  │
│  │  │                                  │                          │   │  │
│  │  │                                  ▼                          │   │  │
│  │  │                         _apply_reflection()                 │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │  Self-Consistency Mode (consistency_samples=2):             │   │  │
│  │  │                                                             │   │  │
│  │  │  ┌── CoT(14B) + Reflect(7B) ──► score_1 ──┐               │   │  │
│  │  │  │                                          ├─► median ──► │   │  │
│  │  │  └── CoT(14B) + Reflect(7B) ──► score_2 ──┘    ▲          │   │  │
│  │  │                                                  │          │   │  │
│  │  │                              if divergence > 2.0:           │   │  │
│  │  │                              set needs_human_review = True  │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                    │  │
│  │  Post-processing:                                                  │  │
│  │  ├─ Inject topic from current_question                             │  │
│  │  └─ Inject question_id                                             │  │
│  │                                                                    │  │
│  │  State writes:                                                     │  │
│  │  └─ current_evaluation { overall_score, reasoning,                 │  │
│  │       key_points_covered, key_points_missed, misconceptions,       │  │
│  │       topic, question_id, is_fallback, needs_human_review }        │  │
│  │                                                                    │  │
│  │  Validation Gate: EvaluatorValidationGate                          │  │
│  │  ├─ Scores 0-10  ├─ Reasoning >50 chars  ├─ Variance <5.0         │  │
│  │  ├─ Required fields  └─ Coverage-score alignment (drift check)     │  │
│  │  └─ On fail → CircuitBreaker → retry once → fallback (score 5.0)  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                           │                                              │
│              ┌────────────┴────────────┐                                 │
│              │       FAN-OUT           │    (parallel execution)          │
│              ▼                         ▼                                  │
│  ┌──────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │  NODE 2a: FEEDBACK   │  │  NODE 2b: QUESTION SELECTOR             │  │
│  │  (7B)                │  │  (7B + 14B)                             │  │
│  │                      │  │                                          │  │
│  │  Input from state:   │  │  Input from state:                      │  │
│  │  ├─ evaluation.score │  │  ├─ current_evaluation                  │  │
│  │  ├─ current_question │  │  ├─ difficulty_level                    │  │
│  │  ├─ candidate_resp   │  │  ├─ topics_covered                     │  │
│  │  ├─ previous_feed_   │  │  ├─ follow_up_count                    │  │
│  │  │  back_structures  │  │  ├─ time_budget_minutes                │  │
│  │  └─ recent_feedbacks │  │  └─ interview_plan                     │  │
│  │                      │  │                                          │  │
│  │  ┌────────────────┐  │  │  ┌──────────────────────────┐           │  │
│  │  │ score < 7.0?   │  │  │  │  _determine_question_    │           │  │
│  │  │ Yes: concept   │  │  │  │  mode()                  │           │  │
│  │  │   lookup tool  │  │  │  │  (rule-based, 0 LLM)     │           │  │
│  │  │   (cache-first)│  │  │  └──────────┬───────────────┘           │  │
│  │  │ No: skip       │  │  │             │                            │  │
│  │  └───────┬────────┘  │  │  ┌──────────┼───────────────────────┐   │  │
│  │          ▼           │  │  │          ▼                       │   │  │
│  │  ┌────────────────┐  │  │  │  RETRIEVE    FOLLOW_UP   CLARIFY│   │  │
│  │  │ Structured     │  │  │  │     │           │           │   │   │  │
│  │  │ Output (7B)    │  │  │  │     ▼           ▼           ▼   │   │  │
│  │  │ → Feedback     │  │  │  │  Cache/CRAG  14B gen     14B gen│   │  │
│  │  │   Components   │  │  │  │     │         + target    + mis-│   │  │
│  │  └───────┬────────┘  │  │  │     ▼         concepts   concep.│   │  │
│  │          │           │  │  │  Structured                     │   │  │
│  │          ▼           │  │  │  Selection                      │   │  │
│  │  ┌────────────────┐  │  │  │  (7B, .with_                   │   │  │
│  │  │ FeedbackComposer│  │  │  │  structured_                   │   │  │
│  │  │ (rule-based)   │  │  │  │  output)                       │   │  │
│  │  │ template       │  │  │  └─────────────────────────────────┘   │  │
│  │  │ rotation       │  │  │                                          │  │
│  │  └───────┬────────┘  │  │  State writes:                          │  │
│  │          │           │  │  ├─ current_question                    │  │
│  │          ▼           │  │  ├─ question_mode                       │  │
│  │  ┌────────────────┐  │  │  ├─ follow_up_count                    │  │
│  │  │ Semantic       │  │  │  ├─ conversation_thread [+1]           │  │
│  │  │ Repetition     │  │  │  └─ topics_covered [+topic]            │  │
│  │  │ Check (7B)     │  │  │                                          │  │
│  │  │ (after 2+      │  │  │  Validation Gate:                       │  │
│  │  │  turns)        │  │  │  QuestionSelectorValidationGate         │  │
│  │  │ Similar?       │  │  │  ├─ Question present                    │  │
│  │  │ → regenerate   │  │  │  ├─ Valid type                          │  │
│  │  └───────┬────────┘  │  │  └─ Time appropriate                    │  │
│  │          │           │  │                                          │  │
│  │  State writes:       │  │                                          │  │
│  │  ├─ current_feedback │  │                                          │  │
│  │  ├─ previous_feedback│  │                                          │  │
│  │  │  _structures [+1] │  │                                          │  │
│  │  └─ recent_feedbacks │  │                                          │  │
│  │    [+1]              │  │                                          │  │
│  │                      │  │                                          │  │
│  │  Validation Gate:    │  │                                          │  │
│  │  FeedbackValidation  │  │                                          │  │
│  │  ├─ 20-200 words     │  │                                          │  │
│  │  ├─ No forbidden     │  │                                          │  │
│  │  ├─ No sycophancy    │  │                                          │  │
│  │  └─ No score leakage │  │                                          │  │
│  └──────────────────────┘  └──────────────────────────────────────────┘  │
│              │                         │                                  │
│              └────────────┬────────────┘                                  │
│                           │  FAN-IN                                       │
│                           ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  NODE 3: SUPERVISOR CHECK (Rule-based, 0 LLM calls)              │  │
│  │                                                                    │  │
│  │  Input from state:                                                 │  │
│  │  ├─ current_evaluation (from Evaluator)                            │  │
│  │  ├─ performance_trajectory (existing)                              │  │
│  │  └─ interview_plan (difficulty_curve, target_questions)            │  │
│  │                                                                    │  │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │  │
│  │  │ OBSERVE  │──►│  ORIENT  │──►│  DECIDE  │──►│   ACT    │      │  │
│  │  │          │   │          │   │          │   │          │      │  │
│  │  │ Read EMA │   │ Trend    │   │ Continue │   │ Set new  │      │  │
│  │  │ + score  │   │ analysis │   │ or end?  │   │ diff +   │      │  │
│  │  │ + count  │   │ + plan   │   │ + adjust │   │ count++  │      │  │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │  │
│  │                                                                    │  │
│  │  EMA Calculation:                                                  │  │
│  │  full_trajectory = existing + [new_score]                          │  │
│  │  ema = TrendAnalyzer.calculate_ema(α=0.3)                          │  │
│  │                                                                    │  │
│  │  Difficulty Resolution:                                            │  │
│  │  ├─ avg_ema ≥ 7.5 + improving → increase difficulty               │  │
│  │  ├─ avg_ema < 5.0 + declining → decrease difficulty               │  │
│  │  └─ else → maintain current                                        │  │
│  │                                                                    │  │
│  │  State writes:                                                     │  │
│  │  ├─ question_count + 1             ├─ should_continue              │  │
│  │  ├─ performance_trajectory [+1]    ├─ end_reason                   │  │
│  │  ├─ ema_trajectory (full recalc)   ├─ difficulty_level             │  │
│  │  ├─ difficulty_history [+1]        ├─ difficulty_reduced_due_to_   │  │
│  │  └─ all_evaluations [+1]          │  performance                   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                           │                                              │
│                           ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  NODE 4: MAYBE SUMMARIZE (Conditional, 14B)                       │  │
│  │                                                                    │  │
│  │  Trigger: question_count % 3 == 0                                  │  │
│  │                                                                    │  │
│  │  ┌─ No ──► return {} (no-op, checkpointed)                        │  │
│  │  │                                                                 │  │
│  │  └─ Yes:                                                           │  │
│  │     Input: messages (turns older than last 3)                      │  │
│  │     Output: compressed summary (~200 tokens)                       │  │
│  │     Truncation: sentence-boundary (not char count)                 │  │
│  │                                                                    │  │
│  │     State writes:                                                  │  │
│  │     ├─ conversation_summary (str)                                  │  │
│  │     └─ summary_turn_count (int)                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                           │                                              │
│                           ▼                                              │
│                     ┌───────────┐                                         │
│                     │    END    │    (checkpointer auto-saves state)      │
│                     └───────────┘                                         │
└──────────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
Response to Client:
{
  feedback: "...",                    ◄── current_feedback (scores hidden)
  next_question: { id, text, topic}, ◄── current_question
  should_continue: true/false,       ◄── from Supervisor
  end_reason: null | "..."           ◄── from Supervisor
}
```

---

## 4. POST /end — Final Report

```
Client Request
│  { interview_id }
│
▼
┌──────────────────────────────────────────────────────────────┐
│  API LAYER                                                   │
│                                                              │
│  1. Load full state via checkpointer                         │
│  2. Build final report from state                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  REPORT BUILDER                                              │
│                                                              │
│  Input from state:                                           │
│  ├─ all_evaluations       (full history)                     │
│  ├─ difficulty_history     (actual difficulty per Q)          │
│  ├─ topics_covered         (topic sequence)                  │
│  ├─ performance_trajectory (raw scores)                      │
│  └─ ema_trajectory         (smoothed scores)                 │
│                                                              │
│  Processing:                                                 │
│  ├─ 1. Per-topic score aggregation                           │
│  ├─ 2. Difficulty-weighted overall score                     │
│  │     (hard Q worth more than easy Q)                       │
│  ├─ 3. Strength/weakness identification                      │
│  └─ 4. Misconception summary                                 │
│                                                              │
│  Output:                                                     │
│  ├─ overall_score (weighted)                                 │
│  ├─ per_topic_scores { topic: { avg, count, difficulty } }   │
│  ├─ strengths []                                             │
│  ├─ weaknesses []                                            │
│  ├─ misconceptions []                                        │
│  └─ difficulty_progression []                                │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. CRAG Subgraph (Internal to AgenticRAGService)

```
                  QS calls retrieve_with_crag(topic, difficulty, exclude_ids)
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  CRAG StateGraph (compiled once, concurrent-safe)                        │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │  RETRIEVE   │    │   GRADE     │    │         ROUTE               │  │
│  │             │    │             │    │                             │  │
│  │  ChromaDB   │───►│  DocumentGr │───►│  HIGH ──────► PACKAGE      │  │
│  │  .query()   │    │  ader (7B)  │    │  MEDIUM ────► PACKAGE      │  │
│  │             │    │             │    │  LOW ───────► REFINE        │  │
│  │  Filters:   │    │  Per-doc    │    │               │             │  │
│  │  • topic    │    │  relevance  │    │               ▼             │  │
│  │  • difficulty│   │  scoring    │    │  ┌────────────────────────┐ │  │
│  │  • time     │    │            │    │  │  REFINE + RE-RETRIEVE  │ │  │
│  │  • exclude  │    │            │    │  │                        │ │  │
│  │             │    │            │    │  │  QueryRefiner:         │ │  │
│  │  BGE-base-  │    │            │    │  │  • Strategy rotation   │ │  │
│  │  en-v1.5    │    │            │    │  │  • Broaden filters     │ │  │
│  │  768-dim    │    │            │    │  │  • Max 2 retries       │ │  │
│  │  cosine     │    │            │    │  │                        │ │  │
│  └─────────────┘    └─────────────┘    │  └────────────┬───────────┘ │  │
│                                        │               │             │  │
│                                        │               └──► RETRIEVE │  │
│                                        │                    (loop)   │  │
│                                        └─────────────────────────────┘  │
│                                                                          │
│  Output: CRAGResult { candidates[], grade, query_used }                  │
│  → Stored in cache with grade-based TTL                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Cache Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CACHE INTERACTIONS                                                      │
│                                                                          │
│  WRITE PATHS:                                                            │
│                                                                          │
│  Pre-warm (background)                                                   │
│  API BackgroundTask ──► cache_store.pre_warm(topic, difficulty)           │
│                         └──► AgenticRAGService.retrieve_batch()           │
│                              └──► CRAG subgraph (real grades)            │
│                                   └──► Topic Pool: {topic}:{difficulty}  │
│                                        TTL based on CRAG grade           │
│                                                                          │
│  Runtime (cache miss)                                                    │
│  QS._retrieve_question() ──► cache miss                                  │
│                              └──► AgenticRAGService.retrieve_with_crag() │
│                                   └──► cache_store.set_topic_questions()  │
│                                        └──► Topic Pool (same structure)  │
│                                                                          │
│  Concept (score < 7.0)                                                   │
│  FeedbackAgent._get_concept_context()                                    │
│  └──► cache_store.get_concept() ──► miss?                                │
│       └──► concept_lookup tool (@tool) ──► ChromaDB ml_concepts          │
│            └──► cache_store.set_concept()                                │
│                 └──► Concept Pool: {concept_name}, TTL 60min             │
│                                                                          │
│  READ PATHS:                                                             │
│                                                                          │
│  QS._retrieve_question()                                                 │
│  └──► cache_store.select_and_mark(session, topic, difficulty, fn)         │
│       └──► Atomic: selector_fn(candidates) → mark used_id → return       │
│            └──► Lock: per-session asyncio.Lock (not global)              │
│                                                                          │
│  EVICTION:                                                               │
│  ├─ Topic Pool:   LRU within pool, max 10/session                        │
│  ├─ Concept Pool: LRU within pool, max 30/session                        │
│  └─ No cross-pool eviction                                               │
│                                                                          │
│  CLEANUP:                                                                │
│  └─ Periodic sweep every 15 min, removes sessions > 90 min old           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 7. State Mutation Timeline (Per Turn)

```
TIME ──────────────────────────────────────────────────────────────────────►

API Layer          Evaluator         Feedback ∥ QS        Supervisor    Summarizer
   │                  │                  │        │            │             │
   │ candidate_       │                  │        │            │             │
   │ response ──►     │                  │        │            │             │
   │ messages [+1]    │                  │        │            │             │
   │                  │                  │        │            │             │
   │              current_              │        │            │             │
   │              evaluation ──►        │        │            │             │
   │                  │                  │        │            │             │
   │                  │           current_   current_         │             │
   │                  │           feedback   question         │             │
   │                  │           previous_  question_        │             │
   │                  │           feedback_  mode             │             │
   │                  │           structures follow_up_       │             │
   │                  │           recent_    count            │             │
   │                  │           feedbacks  conversation_    │             │
   │                  │              │       thread           │             │
   │                  │              │       topics_          │             │
   │                  │              │       covered ──►      │             │
   │                  │              │        │               │             │
   │                  │              │        │         question_count+1    │
   │                  │              │        │         performance_traj    │
   │                  │              │        │         ema_trajectory      │
   │                  │              │        │         difficulty_level    │
   │                  │              │        │         difficulty_history  │
   │                  │              │        │         all_evaluations     │
   │                  │              │        │         should_continue     │
   │                  │              │        │         end_reason ──►      │
   │                  │              │        │               │             │
   │                  │              │        │               │      conversation_
   │                  │              │        │               │      summary
   │                  │              │        │               │      summary_turn_
   │                  │              │        │               │      count
   │                  │              │        │               │             │
   ◄─────────────── Checkpointer saves full state ──────────────────────── │
```

---

## 8. Reducer Rules (Parallel Safety)

| State Key | Reducer | What Agents Return | Why |
|-----------|---------|-------------------|-----|
| `messages` | `add_messages` | New messages only | LangGraph built-in accumulator |
| `performance_trajectory` | `operator.add` | `[new_score]` | Supervisor appends one score |
| `all_evaluations` | `operator.add` | `[evaluation]` | Supervisor appends one eval |
| `topics_covered` | `operator.add` | `[topic]` or `[]` | QS appends on new topic only |
| `conversation_thread` | `operator.add` | `[question_id]` | QS appends one ID |
| `previous_feedback_structures` | `operator.add` | `[structure]` | Feedback appends one template |
| `recent_feedbacks` | `operator.add` | `[feedback_text]` | Feedback appends one text |
| `difficulty_history` | `operator.add` | `[difficulty]` | Supervisor appends one level |
| `current_evaluation` | `last_value` | Full dict | Evaluator replaces entirely |
| `current_feedback` | `last_value` | Full string | Feedback replaces entirely |
| `current_question` | `last_value` | Full dict | QS replaces entirely |
| `difficulty_level` | `last_value` | New level | Supervisor sets new value |
| `question_count` | `last_value` | `count + 1` | Supervisor increments |
| `should_continue` | `last_value` | `true/false` | Supervisor decides |
| `ema_trajectory` | `last_value` | Full recalc list | Supervisor replaces entirely |
| `conversation_summary` | `last_value` | New summary | ConversationManager replaces |

**Key invariant**: During fan-out (Feedback ∥ QS), each agent writes to **disjoint** state keys. No two parallel agents share a `last_value` key.
