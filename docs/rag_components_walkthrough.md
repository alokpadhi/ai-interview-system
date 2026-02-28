# RAG Components — Detailed Code Walkthrough

> Covers: [cache.py](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py), [query_refiner.py](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py), [grader.py](file:///home/alokpadhi/ai-interview-system/src/rag/grader.py), [agentic_rag.py](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py)

---

## How These Files Fit Together

```
QuestionSelectorAgent
        │
        ▼
AgenticRAGService (agentic_rag.py) ─── stateless facade
        │
        ├── InterviewCacheStore (cache.py) ─── check cache first
        │
        └── CRAG Subgraph (built from 3 nodes):
              │
              ├── retrieve_node ──► VectorRetriever (ChromaDB)
              ├── grade_node ──► DocumentGrader (grader.py)
              └── refine_query_node ──► QueryRefiner (query_refiner.py)
```

**Call order:** QS calls `AgenticRAGService.retrieve_with_crag()` → cache check → if miss, run CRAG graph → retrieve from ChromaDB → grade documents → if LOW, refine query → re-retrieve → package results → cache them → return to QS.

---

## 1. cache.py — Session-Isolated Dual-Pool Cache

### What It Does

Manages an in-memory cache for **all concurrent interviews**, with two separate pools:
- **Topic Pool**: Batches of retrieved questions per `topic:difficulty` — max 10 entries/session
- **Concept Pool**: Concept lookup results for feedback enrichment — max 30 entries/session

### External Concepts

#### OrderedDict (Python `collections`)

Python's `OrderedDict` remembers insertion order. When you do `pool.move_to_end(key)`, the most-recently-used entry goes to the end. When you evict with `pool.popitem(last=False)`, the **least-recently-used** entry gets removed from the front. This gives us an **LRU (Least Recently Used) cache** without any external library.

```python
# LRU in action:
pool = OrderedDict()
pool["A"] = data_a   # Order: A
pool["B"] = data_b   # Order: A, B
pool["C"] = data_c   # Order: A, B, C

pool.move_to_end("A")  # Order: B, C, A  (A is now "most recent")
pool.popitem(last=False)  # Removes B (least recent)
```

#### asyncio.Lock — Per-Session Concurrency

Each interview session gets its own `asyncio.Lock`. This means:
- Two different interviews can access the cache simultaneously (no blocking)
- Two coroutines within the **same** interview can't corrupt each other's data

```python
self._session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
```

`defaultdict(asyncio.Lock)` auto-creates a new lock the first time a session ID is used. No explicit initialization needed.

#### TOCTOU Race Condition

**Time-of-Check-to-Time-of-Use (TOCTOU)** is a concurrency bug:

```python
# DANGEROUS — TOCTOU race:
available = cache.get_available_questions()   # CHECK: 3 questions available
# ... another coroutine grabs question #1 here ...
selected = pick_best(available)               # USE: picks question #1
cache.mark_used(selected.id)                  # But #1 is already taken!
```

`select_and_mark()` solves this by running selection **inside** the lock:

```python
async def select_and_mark(self, session_id, topic, difficulty, selector_fn):
    async with self._session_locks[session_id]:    # Lock acquired
        entry = pool.get(key)
        available = entry.get_unused(set())
        selected = await selector_fn(available)     # Select INSIDE lock
        entry.mark_used([selected.id])              # Mark INSIDE lock
        return selected                             # Lock released
```

### Key Classes

#### [CacheEntry](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L88-L121) — One cached batch of questions

```python
@dataclass
class CacheEntry:
    documents: List[RetrievalResult]    # The cached questions
    grade: RelevanceGrade               # HIGH/MEDIUM/LOW — determines TTL
    created_at: float                   # For TTL expiration
    last_accessed_at: float             # For LRU ordering
    hit_count: int = 0                  # Observability
    used_ids: Set[str]                  # Tracks which questions already served
```

**TTL by grade:**
| Grade | TTL | Why |
|-------|-----|-----|
| HIGH | 30 min | High-quality results, safe to reuse |
| MEDIUM | 15 min | Decent but may need refresh sooner |
| LOW | Never cached | `set_topic_questions()` rejects LOW-grade batches entirely |

**Partial reuse** — `get_unused(exclude_ids)` returns only questions not yet served:
```python
def get_unused(self, exclude_ids: Set[str]) -> List[RetrievalResult]:
    combined = self.used_ids | exclude_ids
    return [d for d in self.documents if d.id not in combined]
```

This means a single CRAG retrieval of 5 questions can serve 5 different turns before exhausting.

#### [ConceptEntry](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L123-L137) — Cached concept lookup

Simpler — just `concept_name`, `data`, timestamps. Fixed 60-minute TTL (concepts don't change).

#### [CacheMetrics](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L143-L178) — Observability

Tracks hits, misses, and invalidation reasons. Exposes `hit_rate` property. Every cache operation logs structured data:

```python
logger.debug("CACHE_HIT | pool=%s", pool,
             extra={"interview_id": id, "key": key, "hit_rate": f"{self.hit_rate:.2%}"})
```

### Core Methods Walkthrough

#### [set_topic_questions()](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L216-L253) — Store a batch

```
1. Grade check: LOW? → reject immediately, return
2. Build key: "ml_fundamentals:medium"
3. Acquire session lock
4. Create CacheEntry with grade-based TTL
5. Insert into OrderedDict
6. While pool > 10 entries → pop oldest (LRU eviction)
7. Release lock
```

#### [get_topic_questions()](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L255-L301) — Try to serve from cache

```
1. Build key, acquire session lock
2. Entry not found? → MISS (not_found)
3. Grade is LOW? → remove entry, MISS (quality_too_low)
4. TTL expired? → remove entry, MISS (ttl_expired)
5. Get unused questions (excluding already-served IDs)
6. Less than 50% of requested count available? → remove entry, MISS (partial_exhausted)
7. All checks pass → HIT: move_to_end (LRU), touch(), return questions
```

The 50% threshold (`min_needed = max(1, int(n_results * 0.5))`) prevents serving a near-empty batch — better to CRAG again for fresh results.

#### [select_and_mark()](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L303-L333) — Atomic select + mark

The critical method. Called by QS to pick one question from cached candidates **without race conditions**. The `selector_fn` is the QS's `_react_select()` — the 7B LLM call that picks the best candidate. This LLM call happens **inside the lock**, which is fine because `asyncio.Lock` allows other coroutines to run during `await`.

#### [pre_warm_topics_background()](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L439-L475) — Background pre-warming

Called from FastAPI `BackgroundTasks` after `/start`. Iterates over upcoming topics from the interview plan and pre-populates the cache via the RAG service's `retrieve()` method (which runs the full CRAG loop).

```python
for topic in topics:
    result = await rag_service.retrieve(topic=topic, difficulty=difficulty, ...)
    if result.documents:
        await self.set_topic_questions(session_id, topic, difficulty, result.documents, result.grade)
```

Each pre-warmed topic gets a real CRAG grade → accurate TTL from the start.

#### [cleanup_abandoned_sessions()](file:///home/alokpadhi/ai-interview-system/src/rag/cache.py#L411-L437) — Periodic sweep

Called every 15 minutes by a background task. Any session older than 90 minutes gets fully cleared. Uses `_global_lock` (not per-session) since it modifies the session registry.

### Singleton Pattern

```python
_cache_store: Optional[InterviewCacheStore] = None

def get_cache_store() -> InterviewCacheStore:
    global _cache_store
    if _cache_store is None:
        _cache_store = InterviewCacheStore()
    return _cache_store
```

One cache store instance serves the entire process. All concurrent interviews share it, but each session's data is isolated via per-session locks and per-session OrderedDicts.

---

## 2. grader.py — Hybrid Relevance Grader

### What It Does

Decides whether ChromaDB's retrieved questions are good enough. Uses a **three-tier system** to minimize LLM calls:

```
avg_penalized_score >= 0.75  →  HIGH  (no LLM, instant)
avg_penalized_score <= 0.45  →  LOW   (no LLM, instant)
0.45 < score < 0.75          →  LLM   (borderline, needs judgment)
```

### External Concepts

#### Cosine Similarity Scores

ChromaDB returns a `relevance_score` (0.0–1.0) for each document. This is the cosine similarity between the query embedding and the document embedding. BGE-base-en-v1.5 produces normalized 768-dimensional vectors; cosine similarity of 0.85 means the vectors are very close in the embedding space.

#### LCEL (LangChain Expression Language)

The grading chain is built using LCEL's pipe (`|`) operator:

```python
primary_chain = (
    RunnablePassthrough()                         # Pass input dict unchanged
    | _GRADING_PROMPT                             # Format into ChatPromptTemplate
    | structured_llm.with_retry(                  # LLM call with retry
        stop_after_attempt=2,
        wait_exponential_jitter=False
    )
)
```

Each `|` connects the output of one step to the input of the next. The whole chain is a single `Runnable` object that supports `.ainvoke()`, `.with_retry()`, and `.with_fallbacks()`.

#### .with_structured_output() — Provider-Agnostic Typed Response

```python
structured_llm = llm.with_structured_output(DocumentGrade)
```

This tells the LLM to return a JSON object matching the `DocumentGrade` Pydantic model. Under the hood:
- **OpenAI**: Uses function calling / response_format
- **Anthropic**: Uses tool use
- **Ollama**: Uses JSON mode

The caller always gets a `DocumentGrade` Python object — no manual JSON parsing.

#### .with_retry() + .with_fallbacks() — Reliability Stack

```python
primary_chain = (prompt | structured_llm.with_retry(stop_after_attempt=2))
chain = primary_chain.with_fallbacks([RunnableLambda(lambda _: _FallbackGrade())])
```

- **Retry**: If the LLM call fails (timeout, malformed output), retry up to 2 times
- **Fallback**: If ALL retries fail, return a hardcoded `_FallbackGrade(grade="MEDIUM")`
- The system never crashes on a grader failure — it degrades to MEDIUM

#### RunnablePassthrough vs RunnableLambda

- `RunnablePassthrough()`: Passes input through unchanged. Used as the chain's entry point.
- `RunnableLambda(fn)`: Wraps an arbitrary Python function into a Runnable. Used for the fallback:
  ```python
  fallback_chain = RunnableLambda(lambda _: _FallbackGrade())
  ```

### Code Walkthrough

#### Context Penalties — [_apply_penalties()](file:///home/alokpadhi/ai-interview-system/src/rag/grader.py#L194-L221)

Before deciding the grade, raw cosine scores get penalized for mismatches:

```python
# If we asked for "hard" questions but got "medium" ones:
if doc_difficulty != context.difficulty_level:
    score -= 0.10  # DIFFICULTY_MISMATCH_PENALTY

# If we asked for "optimization" but got "regularization":
if doc_topic != topic_intent:
    score -= 0.15  # TOPIC_MISMATCH_PENALTY
```

**Critical design decision**: Penalties are applied to **copies** of scores, never to the original `RetrievalResult` objects. This means:
- The grader's penalties don't permanently corrupt document scores
- If the same batch is re-graded with different context, original scores are intact

#### The Grade Method — [grade()](file:///home/alokpadhi/ai-interview-system/src/rag/grader.py#L146-L192)

```
1. Empty documents? → LOW immediately
2. Apply penalties → get adjusted scores
3. avg_penalized >= 0.75? → HIGH (fast path, no LLM)
4. avg_penalized <= 0.45? → LOW (fast path, no LLM)
5. Borderline → invoke LLM chain
```

The fast paths handle ~70-80% of cases in practice, keeping the LLM to borderline situations only.

#### LLM Grading — [_llm_grade()](file:///home/alokpadhi/ai-interview-system/src/rag/grader.py#L223-L270)

For borderline cases, the 7B model sees a sample of retrieved documents and decides:

```
Prompt input:
  - Difficulty: "hard"
  - Topic intent: "regularization"
  - Stage: "questioning"
  - Doc sample: first 3 docs, truncated to 120 chars each
  - Average relevance score: 0.62

LLM returns: DocumentGrade(grade="MEDIUM", feedback="Mostly on-topic but...")
```

If the LLM returns an unrecognized grade string, `to_relevance_grade()` safely defaults to MEDIUM.

#### GradingResult — What Gets Returned

```python
@dataclass
class GradingResult:
    grade: RelevanceGrade         # HIGH, MEDIUM, or LOW
    feedback: str                 # Human-readable explanation
    avg_score: float              # Raw average (before penalties)
    used_llm: bool = False        # Was the LLM involved?
    penalised_score: float = None # Score after penalties
```

The `feedback` field is passed to the `QueryRefiner` if the grade is LOW — the refiner uses it to understand **why** retrieval failed.

> **Bug note**: There's a method name mismatch in the current code — `_apply_penalities()` (misspelled) is defined but `_apply_penalties()` (correct) is called. The method also doesn't return the `adjusted` list. These would need fixing during implementation.

---

## 3. query_refiner.py — Loop-Safe Query Refinement

### What It Does

When the CRAG grader says retrieval quality is LOW, the QueryRefiner produces a **meaningfully different** query for the next attempt. It uses a three-strategy rotation to avoid getting stuck:

```
Attempt 0 → LLM refine   (ask 7B to rephrase)
Attempt 1 → Topic pivot  (switch to a different ML topic)
Attempt 2 → Simplify     (strip to 2 core words)
```

### External Concepts

#### Strategy Pattern

The `_pick_strategy()` method selects which refinement approach to use based on the attempt number. This is the Strategy design pattern — the algorithm varies based on context, but the interface (`refine() → str`) stays the same.

#### SequenceMatcher (Python `difflib`)

After any refinement, a similarity check prevents the LLM from producing a query that's too similar to what was already tried:

```python
from difflib import SequenceMatcher

ratio = SequenceMatcher(None, "deep learning optimization", "optimizing deep learning").ratio()
# ratio ≈ 0.72 → different enough (threshold is 0.85)

ratio = SequenceMatcher(None, "deep learning optimization", "deep learning optimisation").ratio()
# ratio ≈ 0.96 → too similar! Rejected.
```

`SequenceMatcher` finds the longest contiguous matching subsequences and computes a ratio. Unlike cosine similarity on embeddings (which captures semantic meaning), this is a **character-level** comparison that catches trivial rephrasing.

#### LCEL Chain for LLM Strategy

```python
self._llm_refiner = _REFINE_PROMPT | llm.with_structured_output(RefinedQuery)
```

The LLM returns a structured `RefinedQuery(query="...", rationale="...")` — not raw text.

### Code Walkthrough

#### [refine()](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py#L234-L262) — Main entry point

```
1. Pick strategy based on attempt number
2. Execute the chosen strategy → get refined query
3. Is refined query too similar to any seen query? (difflib check)
   - Yes → force_different() (mechanical fallback)
   - No → return refined query
```

The similarity guard is the key innovation — even if the LLM rephrases instead of genuinely changing the query, the system catches it and falls back to a mechanical approach.

#### [_llm_refine()](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py#L293-L311) — Strategy 0

```
Prompt:
  "The previous query returned low-quality results."
  "Original query: deep learning optimization"
  "Grader feedback: Questions too advanced for easy difficulty"
  "Difficulty level: easy"
  "Queries already tried: deep learning optimization"

LLM returns: RefinedQuery(query="basic neural network training", rationale="Shifted to simpler framing")
```

On LLM failure → falls back to `_simplify()` (no crash).

#### [_topic_pivot()](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py#L313-L324) — Strategy 1

Completely changes topic. Picks a random uncovered topic from `ML_TOPICS`:

```python
uncovered = [t for t in ML_TOPICS if t not in covered_topics]
pivot_topic = random.choice(uncovered)
return f"{difficulty} {pivot_topic.replace('_', ' ')}"
# e.g., "easy feature engineering"
```

The `ML_TOPICS` list contains ~150 topics covering the full ML/AI curriculum. This ensures the system can always find something to ask about.

#### [_simplify()](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py#L326-L337) — Strategy 2

Last resort — strips the query down to its core:

```python
# "advanced deep learning optimization techniques for neural networks"
# After removing stop words: ["advanced", "deep", "learning", "optimization", "techniques", "neural", "networks"]
# Take first 2: "advanced deep"
# Result: "easy advanced deep"
```

#### [_force_different()](file:///home/alokpadhi/ai-interview-system/src/rag/query_refiner.py#L351-L367) — Mechanical guarantee

If everything else produces a query too similar (>85% match), this method **guarantees** a different query by combining original words with an uncovered topic:

```python
# original: "deep learning optimization"
# first uncovered topic: "feature_engineering"
# Result: "easy deep learning feature engineering"
```

This is deliberately not elegant — it's the emergency fallback that ensures the CRAG loop always has a genuinely different query to try.

---

## 4. agentic_rag.py — CRAG via LangGraph Subgraph

### What It Does

This is the **outer facade** that agents call. It:
1. Checks the cache first → return immediately if hit
2. Runs the CRAG LangGraph subgraph → retrieve, grade, possibly refine and re-retrieve
3. Caches results for future use
4. Falls back to hardcoded questions if everything fails

### External Concepts

#### CRAG (Corrective RAG)

CRAG is an agentic RAG pattern where retrieval quality is **evaluated** and **corrected** if insufficient. Unlike basic RAG (retrieve → use), CRAG adds a feedback loop:

```
Basic RAG:  query → retrieve → use results
CRAG:       query → retrieve → grade → 
                                  ├─ good? → use results
                                  └─ bad?  → refine query → retrieve again → grade → ...
```

The original CRAG paper (Yan et al., 2024) proposes three actions: Correct (use as-is), Incorrect (web search), Ambiguous (combine). Our implementation adapts this for an interview database:
- **HIGH/MEDIUM** → package and return (equivalent to Correct)
- **LOW** → refine query and re-retrieve (equivalent to Incorrect, but searches same DB)
- Maximum 2 correction attempts (prevents infinite loops)

#### LangGraph StateGraph

LangGraph models the CRAG flow as a **state machine**:

```python
class CRAGState(TypedDict):
    query: str                            # Current search query
    documents: List[RetrievalResult]      # Retrieved candidates
    grade: Optional[str]                  # Grading result
    grading_feedback: str                 # Why grade was given
    correction_count: int                 # How many refinements so far
    seen_queries: List[str]               # All queries tried (prevents loops)
    difficulty: str                       # Target difficulty
    topic_intent: str                     # What topic we're looking for
    n_results: int                        # How many results to fetch
    exclude_ids: List[str]                # Already-asked question IDs
    final_documents: List[RetrievalResult] # Output
```

Each node function receives the full state and returns a **partial update** — only the fields it changes. LangGraph handles merging.

#### functools.partial — Dependency Injection

Nodes are plain `async def` functions, not class methods. Dependencies (retriever, grader, refiner) are injected via `functools.partial`:

```python
workflow.add_node("retrieve", partial(retrieve_node, retriever=retriever))
workflow.add_node("grade", partial(grade_node, grader=grader))
workflow.add_node("refine_query", partial(refine_query_node, refiner=refiner))
```

`partial(retrieve_node, retriever=retriever)` creates a new function where `retriever` is pre-filled. LangGraph calls it with just `(state)`, and the retriever dependency is already bound.

#### Conditional Edges

```python
workflow.add_conditional_edges(
    "grade",                    # After grading...
    route_after_grade,          # ...call this function to decide where to go
    {
        "refine_query": "refine_query",    # If LOW → go to refine
        "package_results": "package_results" # If HIGH/MEDIUM → go to package
    },
)
```

`route_after_grade()` returns a string ("refine_query" or "package_results") based on the grade and correction count. LangGraph routes to the corresponding node.

#### asyncio.to_thread — Bridging Sync and Async

ChromaDB's Python client is synchronous. To avoid blocking the async event loop:

```python
docs = await asyncio.to_thread(
    retriever.retrieve_questions,
    query=state["query"],
    difficulty=state["difficulty"],
    ...
)
```

`asyncio.to_thread()` runs the sync function in a thread pool, making it `await`-able. The event loop continues handling other requests while ChromaDB does its work.

### Code Walkthrough

#### The CRAG Graph — [build_crag_graph()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L255-L286)

```
START → retrieve → grade → route_after_grade
                              ├── HIGH/MEDIUM → package_results → END
                              └── LOW (attempts < 2) → refine_query → retrieve (loop)
                              └── LOW (attempts >= 2) → package_results → END
```

The graph is compiled **once** at init and reused for every call. Each `ainvoke()` gets a fresh execution context — multiple CRAG loops can run concurrently without interfering.

#### Node 1: [retrieve_node()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L146-L166)

Queries ChromaDB with metadata filters:
- `topic=state["topic_intent"]` — filter by topic
- `difficulty=state["difficulty"]` — filter by difficulty
- `exclude_ids=set(state["exclude_ids"])` — skip already-asked questions

Returns the documents and appends the query to `seen_queries`.

#### Node 2: [grade_node()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L169-L187)

Calls `DocumentGrader.grade()` with the retrieved documents and a `RetrievalContext`. Returns the grade string and feedback.

#### Node 3: [refine_query_node()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L190-L212)

Only reached if grade is LOW and `correction_count < 2`. Calls `QueryRefiner.refine()` with the attempt number to select the right strategy. Increments `correction_count`.

#### Node 4: [package_results_node()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L215-L233)

Final node — prepares output based on grade:
```python
HIGH   → return all documents as-is
MEDIUM → filter by relevance_score >= 0.55, fallback to all
LOW    → sort by score descending (best available)
```

#### [route_after_grade()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L236-L252) — The routing logic

```python
def route_after_grade(state):
    if grade in ("HIGH", "MEDIUM"):
        return "package_results"          # Good enough, stop
    if attempts >= MAX_CORRECTION_ATTEMPTS:
        return "package_results"          # Exhausted, take what we have
    return "refine_query"                 # Try again
```

### The Facade: [AgenticRAGService](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L293-L453)

#### [retrieve_with_crag()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L325-L426)

The main entry point. Three-tier fallback:

```
1. CACHE CHECK
   └── session_id provided? → cache_store.get_topic_questions()
       └── HIT? → return RAGResult(served_from_cache=True)

2. CRAG SUBGRAPH
   └── Build initial CRAGState with query=topic, difficulty, exclude_ids
   └── crag_graph.ainvoke(state) → runs retrieve→grade→refine loop
   └── Got results? → cache them → return RAGResult

3. FALLBACK (last resort)
   └── Return FALLBACK_QUESTIONS (5 hardcoded ML questions)
   └── RAGResult(is_fallback=True)
   └── System never fails to return something
```

#### Example: Full CRAG flow with correction

```
Call: retrieve_with_crag(topic="regularization", difficulty="hard", exclude_ids=["q1","q2"])

1. Cache: MISS (no entry for regularization:hard)

2. CRAG iteration 1:
   retrieve_node → ChromaDB returns 5 docs (avg score 0.52)
   grade_node → penalty for 2 topic mismatches → avg_penalized 0.41 → LOW
   route → correction_count(0) < 2 → refine_query
   refine_query_node → strategy=LLM_REFINE → "L1 L2 penalty comparison hard"

3. CRAG iteration 2:
   retrieve_node → ChromaDB returns 5 docs (avg score 0.78)
   grade_node → avg_penalized 0.76 → HIGH (fast path, no LLM)
   route → HIGH → package_results
   package_results_node → return all 5 docs

4. Cache: store with grade=HIGH, TTL=30min

5. Return: RAGResult(candidates=[5 docs], grade=HIGH, attempts=2,
                     corrective_applied=True, refined_query="L1 L2 penalty...")
```

#### [RAGResult](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L109-L123) — What QS Receives

```python
@dataclass
class RAGResult:
    candidates: List[RetrievalResult]   # The questions
    grade: RelevanceGrade               # Cache uses this for TTL
    attempts: int = 1                   # How many CRAG iterations
    refined_query: str = None           # Final query used (if refined)
    served_from_cache: bool = False     # Was this a cache hit?
    corrective_applied: bool = False    # Did CRAG correct?
    queries_used: List[str] = []        # All queries tried (observability)
    latency_ms: float = 0.0            # End-to-end timing
    is_fallback: bool = False           # True = hardcoded questions
```

The `is_fallback` flag is critical — the Evaluator uses it to mark scores as unreliable, and the Supervisor excludes fallback scores from EMA calculations.

#### [retrieve_batch()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L428-L443) — Pre-warming API

Thin wrapper called by `InterviewCacheStore.pre_warm_topics_background()`:

```python
async def retrieve_batch(self, topic, difficulty, n_results=5):
    return await self.retrieve_with_crag(
        topic=topic, difficulty=difficulty,
        exclude_ids=[], n_results=n_results
    )
```

No `session_id` → no caching within this call (the caller handles caching).

#### [end_interview()](file:///home/alokpadhi/ai-interview-system/src/rag/agentic_rag.py#L445-L452) — Cleanup

```python
async def end_interview(self, session_id):
    removed = await self.cache_store.clear_session(session_id)
```

Called by the API layer on `POST /end`. Removes all cached data for the session.

---

## Component Interaction Summary

```
Turn N: QS needs a question on "optimization" at "hard" difficulty

QS → AgenticRAGService.retrieve_with_crag(topic, difficulty, exclude_ids, session_id)
  │
  ├─ Step 1: cache_store.get_topic_questions("optimization:hard")
  │    └─ HIT (pre-warmed) → return 4 unused questions → done (~1ms)
  │
  └─ Step 1: cache miss
       │
       ├─ Step 2: crag_graph.ainvoke(initial_state)
       │    │
       │    ├─ retrieve_node → ChromaDB.query(topic, difficulty)
       │    │    └─ asyncio.to_thread (sync→async bridge)
       │    │
       │    ├─ grade_node → DocumentGrader.grade(docs, context)
       │    │    ├─ _apply_penalties (copy scores, subtract mismatches)
       │    │    ├─ Fast path? (>0.75 or <0.45) → skip LLM
       │    │    └─ Borderline → llm.with_structured_output(DocumentGrade)
       │    │         └─ .with_retry(2) → .with_fallbacks(_FallbackGrade)
       │    │
       │    ├─ route_after_grade → HIGH/MED? → package_results → END
       │    │                    → LOW + attempts < 2? → refine_query → retrieve (loop)
       │    │
       │    └─ refine_query_node → QueryRefiner.refine(attempt)
       │         ├─ Strategy 0: LLM refine (7B, structured output)
       │         ├─ Strategy 1: topic_pivot (random uncovered topic)
       │         ├─ Strategy 2: simplify (strip to 2 words)
       │         └─ Similarity guard: >85% similar? → force_different()
       │
       ├─ Step 3: cache_store.set_topic_questions(results, grade)
       │    └─ LOW grade? → rejected, not cached
       │
       └─ Step 4 (if everything fails): return FALLBACK_QUESTIONS

QS ← RAGResult(candidates, grade, attempts, latency_ms, is_fallback)
  │
  └─ cache_store.select_and_mark(session_id, selector_fn=_react_select)
       └─ Atomic: lock → 7B picks best → marks used → unlock
```
