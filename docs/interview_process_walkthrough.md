# Interview Process Walkthrough — Complete Example

> This document traces a **full 30-minute interview** from POST /start to POST /end,
> showing exactly what every agent, component, and data store does at each step.
>
> Source of truth: [architecture(v2).md](file:///home/alokpadhi/ai-interview-system/docs/architecture(v2).md)

---

## The Setup

A candidate wants to practice ML/AI interview questions. They hit the API.

---

## Phase 1: Starting the Interview (`POST /start`)

### Step 1.1 — Client sends request

```json
POST /api/v1/interview/start
{ "user_id": "candidate_42" }
```

All other fields use defaults: `difficulty = "medium"`, `focus_topics = []` (all topics), `time_budget_minutes = 30`.

### Step 1.2 — API Layer initializes state

The API layer creates a fresh `InterviewState`:

```python
state = {
    "interview_id": "a1b2c3d4-...",
    "interview_start_time": "2026-02-23T10:00:00",
    "time_budget_minutes": 30,
    "difficulty_level": "medium",
    "stage": "init",
    "question_count": 0,
    "messages": [],
    "performance_trajectory": [],
    "ema_trajectory": [],
    "topics_covered": [],
    "previous_feedback_structures": [],
    "recent_feedbacks": [],
    "should_continue": True,
    # ... all other fields initialized to defaults
}
```

A `RunnableConfig` is built for tracing:

```python
config = RunnableConfig(
    configurable={"thread_id": "a1b2c3d4-..."},
    metadata={"user_id": "candidate_42", "turn": 0},
    tags=["interview", "start"]
)
```

### Step 1.3 — Start Graph runs (2 nodes, sequential)

```
┌────────────────┐       ┌─────────────────────┐
│  create_plan   │──────►│   first_question    │──► END
│  (Supervisor)  │       │  (Question Selector) │
└────────────────┘       └─────────────────────┘
```

#### Node 1: Supervisor.create_interview_plan()

**What:** 14B model generates an interview plan (1 LLM call)

**Prompt sent:**
```
Create an interview plan for:
  Difficulty: medium
  Focus topics: (all — no filter)
  Time budget: 30 minutes
  Target questions: 7
```

**LLM returns:**
```json
{
  "topic_sequence": [
    "ml_fundamentals", "optimization", "regularization",
    "neural_networks", "evaluation_metrics", "model_selection", "deployment"
  ],
  "difficulty_curve": ["medium", "medium", "medium", "hard", "hard", "hard", "hard"]
}
```

**State mutations:**
| Key | Value | Why |
|-----|-------|-----|
| `interview_plan` | `{topic_sequence, difficulty_curve}` | Plan for the whole interview |
| `difficulty_level` | `"medium"` | First entry of difficulty_curve |
| `original_difficulty` | `"medium"` | Preserved for final report |
| `stage` | `"questioning"` | Interview is active |

#### Node 2: QuestionSelector.execute()

**What:** Picks the first question. Mode is always `RETRIEVE` for turn 0.

**Internal step-by-step:**

| Step | Component | Action |
|------|-----------|--------|
| 1 | `_determine_question_mode()` | `question_count == 0` → `"retrieve"` (rule-based, 0 LLM) |
| 2 | `_get_next_topic_from_plan()` | `plan.topic_sequence[0]` → `"ml_fundamentals"` (rule-based) |
| 3 | Cache lookup | Key `{session}:ml_fundamentals:medium` → **MISS** |
| 4 | `AgenticRAGService.retrieve_with_crag()` | Triggers CRAG subgraph |
| 4a | └─ ChromaDB query | `topic=ml_fundamentals, difficulty=medium` → 5 docs |
| 4b | └─ DocumentGrader (7B) | Grades each doc → all `HIGH` relevance |
| 4c | └─ Grade check | `HIGH` → skip refine, package results |
| 5 | `cache_store.set_topic_questions()` | Caches 5 questions, TTL=30min (HIGH grade) |
| 6 | `cache_store.select_and_mark()` | Atomic: lock → 7B selects best → marks used |
| 6a | └─ `_react_select()` (7B) | `.with_structured_output(QuestionSelection)` |

**7B selection output:**
```python
QuestionSelection(
    selected_id="q_bias_variance_01",
    reasoning="Good foundational opener for medium difficulty"
)
```

**State mutations:**
| Key | Value | Reducer |
|-----|-------|---------|
| `current_question` | `{id, text, topic, rubric, ...}` | `last_value` |
| `question_mode` | `"retrieve"` | `last_value` |
| `follow_up_count` | `0` | `last_value` |
| `topics_covered` | `["ml_fundamentals"]` | `operator.add` |
| `conversation_thread` | `["q_bias_variance_01"]` | `operator.add` |

### Step 1.4 — Background pre-warming (non-blocking)

While the response reaches the client, a FastAPI `BackgroundTask` fires:

```
Pre-warm pipeline (non-blocking):
  for each of ["optimization", "regularization", "neural_networks", "evaluation_metrics"]:
    → AgenticRAGService.retrieve_batch(topic, difficulty="medium")
      → Full CRAG loop (ChromaDB retrieval + 7B grading)
      → cache_store stores with real CRAG grades → accurate TTLs
```

### Step 1.5 — Response to client

```json
{
  "session_id": "a1b2c3d4-...",
  "question": {
    "text": "Can you explain the bias-variance tradeoff and why it matters?",
    "topic": "ml_fundamentals",
    "estimated_time_minutes": 4
  },
  "time_budget_minutes": 30,
  "target_questions": 7
}
```

> **Total:** ~4-5s latency, 3 LLM calls (14B plan, 7B grader, 7B selection)

---

## Phase 2: First Answer & Evaluation (`POST /submit_response` — Turn 1)

### Step 2.1 — Client submits a strong answer

```json
{
  "session_id": "a1b2c3d4-...",
  "response": "The bias-variance tradeoff is about balancing two sources of error.
    Bias is the error from overly simplistic assumptions — a linear model fitting
    nonlinear data has high bias. Variance is sensitivity to training data
    fluctuations — a deep decision tree memorizes noise, giving high variance.
    You want the sweet spot: complex enough to capture patterns, simple enough
    to generalize. Techniques like cross-validation help find this balance."
}
```

### Step 2.2 — Interview Graph fires (4 nodes)

```
Evaluator → (Feedback ∥ QS) → Supervisor → MaybeSummarize → END
```

---

#### NODE 1: Evaluator Agent

**Pattern:** Chain-of-Thought (14B) + Reflection (7B) + optional Self-Consistency

**LLM Call 1 — CoT Evaluation (14B), `.with_structured_output(EvaluationOutput)`:**

Prompt includes the question, candidate response, and rubric key points.

**Output:**
```json
{
  "overall_score": 7.5,
  "technical_accuracy": {"score": 8.0, "reasoning": "Correct definitions of bias and variance"},
  "completeness": {"score": 7.0, "reasoning": "Mentioned cross-validation but missed regularization"},
  "depth": {"score": 7.0, "reasoning": "Good intuition, could go deeper on math"},
  "clarity": {"score": 8.0, "reasoning": "Clear examples with decision tree and linear model"},
  "reasoning": "Solid conceptual understanding. Correctly identified the tradeoff...",
  "key_points_covered": ["bias definition", "variance definition", "tradeoff relationship"],
  "key_points_missed": ["impact on model complexity"],
  "misconceptions": []
}
```

**LLM Call 2 — Reflection (7B):**

```
"Review this evaluation. Is the score of 7.5 consistent with covering 3/5 key points?
 Any reasoning gaps?"
```
Reflection confirms: score is reasonable, no adjustment needed.

**Post-processing:**
- Injects `topic: "ml_fundamentals"` from `current_question`
- Injects `question_id: "q_bias_variance_01"`

**Validation Gate (`EvaluatorValidationGate`):**
- ✅ Scores in 0-10 range (all pass)
- ✅ Reasoning > 50 chars
- ✅ Score variance < 5.0 (max 8.0 - min 7.0 = 1.0)
- ✅ Required fields present
- ✅ Coverage-score alignment (score 7.5 with 60% coverage — reasonable)

**State writes:** `current_evaluation = {overall_score: 7.5, ...}`

---

#### NODE 2a: Feedback Agent (runs in parallel with QS)

**Pattern:** Structured Output (7B) + Conditional RAG + Semantic Repetition Reflection

**Step 1 — Concept lookup:**
- Score is 7.5 → `>= 7.0` → **skip concept lookup** (no RAG call)

**Step 2 — LLM Call (7B), `.with_structured_output(FeedbackComponents)`:**

```
System: You are an interview feedback generator. Generate constructive feedback.
        NEVER reveal scores, NEVER say 'you missed X'.
        Tone guidance: Brief, genuine acknowledgment. No need to elaborate.
```

**Output:**
```json
{
  "strength_acknowledgment": "Your decision tree and linear model examples effectively illustrated the core tension.",
  "gap_hint": "There's an interesting connection between this tradeoff and how we control model complexity...",
  "transition_phrase": "Building on that..."
}
```

**Step 3 — FeedbackComposer (rule-based):**
- Score band: `7.5 → "high"`
- Available structures for "high": `["{strength}", "{strength} {transition}", "{transition}", "{strength}"]`
- Turn 1, no previous structures → selects: `"{strength} {transition}"`
- Transition: `"Building on that..."`

**Composed feedback:**
```
"Your decision tree and linear model examples effectively illustrated the core tension. Building on that..."
```

**Step 4 — Semantic repetition check:**
- `len(recent_feedbacks) < 2` → **skipped** (first turn)

**Validation Gate (`FeedbackValidationGate`):**
- ✅ Length: 22 words (within 20-200)
- ✅ No forbidden phrases ("you failed", "wrong answer", etc.)
- ✅ No sycophancy check (score ≥ 7.0, skip)
- ✅ No score leakage (no "7.5/10" or "scored 7" in text)

**State writes:**
| Key | Value | Reducer |
|-----|-------|---------|
| `current_feedback` | `"Your decision tree and..."` | `last_value` |
| `previous_feedback_structures` | `["{strength} {transition}"]` | `operator.add` |
| `recent_feedbacks` | `["Your decision tree and..."]` | `operator.add` |

---

#### NODE 2b: Question Selector Agent (runs in parallel with Feedback)

**Pattern:** Rule-based mode determination + Delegated CRAG + Structured Selection

**Step 1 — Mode determination (rule-based):**
| Check | Value | Result |
|-------|-------|--------|
| `question_count == 0`? | No (it's turn 1 now) | Continue |
| `remaining_time < 5`? | No (28 min left) | Continue |
| Misconceptions detected? | `[]` — none | Skip clarify |
| Score < 7.0 + missed points? | 7.5 ≥ 7.0 | Skip follow_up |
| Score 7.0-8.0 + missed + follow_ups < 1? | **Yes** (7.5, 1 missed, 0 follow-ups) | **→ FOLLOW_UP** |

Mode = `"follow_up"` — candidate was decent but missed a key point.

**Step 2 — Generate follow-up (14B):**

```
System: Generate a targeted follow-up question that probes the missed concepts.

Original question: Can you explain the bias-variance tradeoff...
Candidate response: The bias-variance tradeoff is about balancing...
Missed points: ["impact on model complexity"]
Topic: ml_fundamentals
```

**LLM output:**
```
"You mentioned cross-validation helps find the balance. How does increasing
 model complexity — say, adding polynomial features — specifically affect
 bias and variance? Can you walk through a concrete example?"
```

**Step 3 — Generate target_concepts (dynamic rubric):**
```json
["polynomial feature impact", "bias reduction with complexity",
 "variance increase with complexity", "overfitting threshold"]
```

This dynamic rubric ensures the Evaluator can properly evaluate this follow-up question,
even though it has no static rubric from the database.

**State writes:**
| Key | Value | Reducer |
|-----|-------|---------|
| `current_question` | `{id: "q_bias_variance_01_followup_1", text: "You mentioned...", question_type: "follow_up", target_concepts: [...]}` | `last_value` |
| `question_mode` | `"follow_up"` | `last_value` |
| `follow_up_count` | `1` (was 0) | `last_value` |
| `conversation_thread` | `["q_bias_variance_01_followup_1"]` | `operator.add` |
| `topics_covered` | `[]` (no new topic) | `operator.add` |

---

#### NODE 3: Supervisor (rule-based, 0 LLM calls)

**Pattern:** OODA Loop

**Observe:**
```python
new_score = 7.5
full_trajectory = [] + [7.5]  # first score
new_ema = TrendAnalyzer.calculate_ema([7.5], alpha=0.3)  # → [7.5]
```

**Orient:**
```python
avg_ema = 7.5
trend = "insufficient_data"  # Only 1 data point, need ≥ 4 for trend
```

**Decide:**
```python
# Continuation check:
question_count (will be 1) < target_questions (7)  → continue
time_remaining (28 min) > 0  → continue
should_continue = True

# Difficulty check:
insufficient_data (< 4 questions) → use plan's difficulty_curve
difficulty_level stays "medium"
```

**Act:**

**State writes:**
| Key | Value | Reducer |
|-----|-------|---------|
| `question_count` | `1` (incremented from 0) | `last_value` |
| `performance_trajectory` | `[7.5]` | `operator.add` |
| `ema_trajectory` | `[7.5]` | `last_value` (full recalc) |
| `difficulty_history` | `["medium"]` | `operator.add` |
| `all_evaluations` | `[{...full eval...}]` | `operator.add` |
| `should_continue` | `true` | `last_value` |
| `difficulty_level` | `"medium"` | `last_value` (unchanged) |

---

#### NODE 4: MaybeSummarize (conditional)

```python
question_count (1) % 3 == 0?  → No → return {} (no-op)
```

No summarization this turn. Will trigger at turn 3.

---

### Step 2.3 — Response to client

```json
{
  "feedback": "Your decision tree and linear model examples effectively illustrated the core tension. Building on that...",
  "next_question": {
    "text": "You mentioned cross-validation helps find the balance. How does increasing model complexity — say, adding polynomial features — specifically affect bias and variance?",
    "topic": "ml_fundamentals"
  },
  "progress": {
    "questions_completed": 1,
    "time_elapsed_minutes": 4.2,
    "time_remaining_minutes": 25.8
  },
  "continue_interview": true
}
```

> Note: The **score (7.5) is never exposed** to the client. Feedback hints at gaps without revealing numbers.

> **Total:** ~3.5-5s latency, 4 LLM calls (14B eval, 7B reflection, 7B feedback, 14B follow-up)

---

## Phase 3: Follow-Up Answer (Turn 2 — Follow-Up)

### Step 3.1 — Client answers the follow-up

```json
{
  "session_id": "a1b2c3d4-...",
  "response": "Adding polynomial features increases model complexity. With degree 1 (linear), you have high bias but low variance — the model can't capture curves. At degree 5, bias drops because the model fits the training data closely, but variance increases because small changes in training data cause wild swings in predictions. At some extreme degree like 15, you're essentially memorizing — near-zero bias but massive variance. The optimal point depends on the problem, but you can use validation curves to find where test error is minimized."
}
```

This is a strong answer. Let's see what the system does.

### Step 3.2 — Graph execution

#### Evaluator

This time, the question has **no static rubric** (it's a follow-up). The Evaluator uses `target_concepts` as a dynamic rubric:

```python
rubric = question.get("rubric", {})          # → {} (empty for follow-ups)
target_concepts = question.get("target_concepts", [])
# → ["polynomial feature impact", "bias reduction with complexity",
#     "variance increase with complexity", "overfitting threshold"]
```

**CoT Evaluation (14B):**
```json
{
  "overall_score": 8.5,
  "key_points_covered": ["polynomial feature impact", "bias reduction with complexity",
                          "variance increase with complexity"],
  "key_points_missed": ["overfitting threshold"],
  "misconceptions": [],
  "reasoning": "Excellent concrete walkthrough with degree examples..."
}
```

**Validation Gate — Dynamic rubric path:**
- `_extract_key_points()` → finds `target_concepts` (not static rubric)
- Coverage: 3/4 key points = 75% → score 8.5 → alignment ✅

#### Feedback (parallel)

**Concept lookup:** Score 8.5 ≥ 7.0 → **skipped**

**Feedback generation (7B):**
```json
{
  "strength_acknowledgment": "Walking through the polynomial degrees progressively really clarified the mechanics.",
  "gap_hint": "",
  "transition_phrase": "I'm curious..."
}
```

**Composer:** Score band `"high"`, turn 2. Previous structure was `"{strength} {transition}"`.
Available structures: `["{strength}", "{strength} {transition}", "{transition}", "{strength}"]`.
Excludes last 2 → selects `"{transition}"`:

```
"I'm curious..."
```

**Semantic repetition check:**
- `len(recent_feedbacks) < 2` → skipped (only 1 previous)

#### Question Selector (parallel)

**Mode determination:**
- Score 8.5 ≥ 8.0 with no misconceptions → `"retrieve"` (move to new topic!)
- `follow_up_count` resets to 0

**Topic selection:**
- `topics_covered = ["ml_fundamentals"]`
- `plan.topic_sequence[1]` → `"optimization"` (next unvisited topic)

**Cache lookup:**
- Key `{session}:optimization:medium` → **HIT** (pre-warmed in background!)
- 5 cached questions available

**Atomic select-and-mark:**
- 7B selects best → `QuestionSelection(selected_id="q_gradient_descent_04", reasoning="Appropriate follow-on")`

**State writes:** New question about gradient descent, `topics_covered += ["optimization"]`.

#### Supervisor

```python
trajectory = [7.5] + [8.5]  # 2 scores
ema = calculate_ema([7.5, 8.5], alpha=0.3)
    # EMA[0] = 7.5
    # EMA[1] = 0.3 * 8.5 + 0.7 * 7.5 = 7.80
question_count = 2
```

Still insufficient data (< 4 questions) → difficulty stays `"medium"`.

---

## Phase 4: Turns 3-5 (Medium Difficulty, Summarization Triggers)

### Turn 3 — Summarization fires

After Supervisor increments `question_count` to 3:

```python
question_count (3) % 3 == 0?  → Yes → Summarize!
```

**ConversationManager (14B):**

Takes turns 1-3 (everything except the 3 most recent), compresses them:

```
"Covered ML fundamentals (bias-variance tradeoff) and optimization basics.
 Strong practical intuition with polynomial degree examples. Mentioned
 cross-validation and validation curves. Minor gap: didn't connect to
 regularization explicitly."
```

**State writes:**
- `conversation_summary` = compressed text (~200 tokens)
- `summary_turn_count` = 3

Now the context window stays bounded at ~1700 tokens regardless of interview length.

### Turn 4 — Difficulty increases!

After 4 questions, suppose these scores came in:

```
Trajectory: [7.5, 8.5, 7.0, 8.0]
EMA:        [7.5, 7.80, 7.56, 7.69]
avg_ema = 7.64
```

**Supervisor orient:**
```python
avg_ema (7.64) >= 7.5 threshold?  → Yes
Trend improving (7.69 > 7.56)?    → Yes
Data sufficient (4 questions)?     → Yes
→ INCREASE difficulty to "hard"
```

**State writes:** `difficulty_level = "hard"`, `difficulty_reduced_due_to_performance = False`

Now QS will retrieve `hard` questions. The pre-warmed cache was for `medium`, so this will be a **cache miss** — CRAG will retrieve at `hard` difficulty at runtime.

---

## Phase 5: Turn 5 — Weak Answer (Triggers Follow-Up + Clarification)

### Candidate struggles with a hard question

Score comes back as **4.5** with a misconception detected:

```json
{
  "overall_score": 4.5,
  "key_points_missed": ["L1 sparsity mechanism", "feature selection effect"],
  "misconceptions": ["Confused L1 and L2 regularization effects"]
}
```

#### Feedback Agent

**Concept lookup:** Score 4.5 < 7.0 → **triggers concept_lookup tool!**

```python
missed = ["L1 sparsity mechanism"]
cached = cache_store.get_concept("a1b2c3d4", "L1 sparsity mechanism")
# → MISS (first time seeing this concept)

data = concept_tool.ainvoke("L1 sparsity mechanism")
# → ChromaDB ml_concepts collection
# → Returns: {"simple_explanation": "L1 pushes weights to exactly zero, effectively removing features..."}

cache_store.set_concept("a1b2c3d4", "L1 sparsity mechanism", data)
# → Concept Pool, TTL 60min

concept_context = "L1 pushes weights to exactly zero, effectively removing features..."
```

**Feedback generated with concept context:**
```json
{
  "strength_acknowledgment": "",
  "gap_hint": "There's a subtle but important distinction in how different penalty terms affect the geometry of the solution space — particularly near the axes...",
  "transition_phrase": ""
}
```

Tone: `"Patient. No praise openers. Direct but kind."` (score < 4.0 threshold). No sycophancy.

**Semantic repetition check:**
- `len(recent_feedbacks) >= 2` → runs 7B check
- Compares with last 2 feedbacks → `"different"` → keeps as-is

#### Question Selector

**Mode determination:**
```python
misconceptions = ["Confused L1 and L2 regularization effects"]  # non-empty!
follow_up_count = 0
MAX_FOLLOW_UPS = 2
→ Mode = "clarify"  (misconception takes priority over follow_up)
```

**Clarification generation (14B):**
```
System: Generate a clarification question that helps the candidate
        recognize and correct their misconception.

Misconception: Confused L1 and L2 regularization effects
Original: "Explain L1 vs L2 regularization..."
Response: "L1 makes all weights small, L2 makes some weights zero..."  (reversed!)
```

**Output:**
```
"You mentioned that L1 makes weights small uniformly. Let me give you a hint —
 think about the shape of the constraint region. L1 uses a diamond shape and
 L2 uses a circle. Which shape is more likely to touch the axes? What does
 that mean for individual weights?"
```

**Dynamic rubric:** `target_concepts: ["L1 diamond constraint", "L2 circle constraint", "sparsity from corners"]`
**target_misconception:** `"Confused L1 and L2 regularization effects"`

#### Supervisor — EMA during struggle

```
Trajectory: [7.5, 8.5, 7.0, 8.0, 4.5]
EMA:        [7.5, 7.80, 7.56, 7.69, 6.73]
avg_ema = 7.46  →  < 7.5 → no increase
Trend: declining (6.73 < 7.69) but avg > 5.0 → no decrease
→ Difficulty stays "hard" (one bad score doesn't cause immediate change)
```

This is why EMA smoothing matters — the α=0.3 prevents oscillation from a single bad answer.

---

## Phase 6: Turn 6 — Candidate Corrects Misconception

Candidate answers the clarification correctly (score 7.0, misconception resolved).

**QS mode determination:**
```python
misconceptions = []  # resolved!
score = 7.0 < 8.0, missed_points exist, follow_ups = 1
follow_ups (1) < MAX_FOLLOW_UPS (2)
→ Mode = "follow_up" (one more chance to probe depth)
```

After the follow-up (Turn 7), suppose score is 6.5:

```
Trajectory: [7.5, 8.5, 7.0, 8.0, 4.5, 7.0, 6.5]
EMA:        [7.5, 7.80, 7.56, 7.69, 6.73, 6.81, 6.72]
avg_ema ≈ 7.26  →  still above 5.0, below 7.5
→ Difficulty stays "hard" (no change)
```

Eventually QS mode becomes `"retrieve"` (either score ≥ 8.0 or `follow_ups == MAX_FOLLOW_UPS`), and the interview moves to the next topic.

---

## Phase 7: Interview Ends

### Trigger: Supervisor decides to stop

After turn 7, Supervisor checks:

```python
question_count (7) >= target_questions (7)?  → Yes
→ should_continue = False
→ end_reason = "target_reached"
```

Alternative end reasons: `"time_expired"` (if time_remaining < 2 min), `"all_topics_covered"`.

### Response to client on final turn

```json
{
  "feedback": "That's an insightful observation about...",
  "next_question": null,
  "progress": {
    "questions_completed": 7,
    "time_elapsed_minutes": 28.3,
    "time_remaining_minutes": 1.7
  },
  "continue_interview": false
}
```

---

## Phase 8: Final Report (`POST /end`)

```json
POST /api/v1/interview/end
{ "session_id": "a1b2c3d4-..." }
```

### Report Builder reads full state

**Input:** All accumulated state from the checkpointer.

**Processing:**

```
all_evaluations:  [7.5, 8.5, 7.0, 8.0, 4.5, 7.0, 6.5]
difficulty_history: [med, med, med, hard, hard, hard, hard]

Difficulty weights: medium=1.0, hard=1.3

Weighted scores:
  7.5×1.0 + 8.5×1.0 + 7.0×1.0 + 8.0×1.3 + 4.5×1.3 + 7.0×1.3 + 6.5×1.3
  = 23.0 + 33.8
  = 56.8 / 7.9 (sum of weights)
  ≈ 7.19 adjusted score

Per-topic breakdown:
  ml_fundamentals: avg(7.5, 8.5) = 8.0   ← strength
  optimization:    avg(7.0) = 7.0
  regularization:  avg(8.0, 4.5, 7.0) = 6.5  ← area for improvement
  neural_networks: avg(6.5) = 6.5   ← area for improvement
```

### Response

```json
{
  "overall_score": 7.0,
  "adjusted_score": 7.2,
  "questions_asked": 7,
  "time_taken_minutes": 28.3,
  "difficulty_progression": ["medium", "medium", "medium", "hard", "hard", "hard", "hard"],
  "topic_scores": {
    "ml_fundamentals": 8.0,
    "optimization": 7.0,
    "regularization": 6.5,
    "neural_networks": 6.5
  },
  "strengths": ["ml_fundamentals", "optimization"],
  "areas_for_improvement": ["regularization", "neural_networks"],
  "performance_notes": [],
  "fallback_count": 0
}
```

---

## Summary: What Each Component Did

| Component | Role | LLM | Calls/Turn |
|-----------|------|-----|------------|
| **API Layer** | Routes requests, manages `RunnableConfig`, triggers background tasks | None | 0 |
| **Checkpointer** | Auto-saves/loads state via `thread_id` | None | 0 |
| **Supervisor** | Creates plan (once), OODA loop (every turn), owns `question_count` | 14B (plan only) | 0 per turn |
| **TrendAnalyzer** | EMA smoothing (α=0.3), difficulty thresholds | None | 0 |
| **ValidationGates** | 3 gates (Evaluator/Feedback/QS), checks output quality | None | 0 |
| **CircuitBreaker** | Max 1 retry per agent, then fallback | None | 0 |
| **Evaluator** | CoT evaluation + reflection + optional self-consistency | 14B + 7B | 2 (or 4 in 2x mode) |
| **Feedback** | Structured feedback + concept lookup + repetition reflection | 7B | 1-3 |
| **FeedbackComposer** | Template rotation, avoid repetitive structures | None | 0 |
| **QuestionSelector** | Mode determination → retrieve/follow-up/clarify | 7B + 14B | 1-2 |
| **ConversationManager** | Summarizes older turns every 3 questions | 14B | 0-1 |
| **AgenticRAGService** | Owns CRAG loop (retrieve → grade → refine) | 7B | 1-2 (on cache miss) |
| **InterviewCacheStore** | Topic pool + concept pool, atomic select-and-mark | None | 0 |
| **ChromaDB** | Vector search (BGE-base-en-v1.5, 768-dim, cosine) | None | 0 |
| **concept_lookup tool** | Retrieves concept explanations from `ml_concepts` collection | None | 0 |
| **rubric_checker tool** | Looks up rubric for retrieved questions | None | 0 |
| **code_validator tool** | AST-parses code answers for syntax validity | None | 0 |

---

## Agentic Patterns Used

| Pattern | Where | How |
|---------|-------|-----|
| **Chain-of-Thought** | Evaluator | 14B reasons through rubric points step-by-step |
| **Reflection** | Evaluator | 7B cross-checks if score matches reasoning |
| **Self-Consistency** | Evaluator (optional) | 2 parallel evals, median score, flag divergence |
| **Structured Output** | Evaluator, Feedback, QS selection | `.with_structured_output(PydanticModel)` everywhere |
| **Conditional RAG** | Feedback | concept_lookup only when score < 7.0 |
| **Semantic Repetition Reflection** | Feedback | 7B checks new vs last 2 feedbacks after turn 2 |
| **CRAG** | AgenticRAGService (delegated by QS) | Retrieve → grade → refine → re-retrieve loop |
| **ReAct-style Selection** | QS `_react_select()` | 7B reasons over candidates to pick best |
| **Rule-based Routing** | QS mode, Supervisor OODA, FeedbackComposer | Deterministic, 0 LLM calls, predictable |
| **Dynamic Rubric** | QS → Evaluator | `target_concepts` for follow-up/clarify questions |
| **Atomic Operations** | Cache `select_and_mark()` | Per-session lock, no TOCTOU race |
| **Graceful Degradation** | All chains | `.with_retry()` + `.with_fallbacks()` + `configurable_alternatives()` |
