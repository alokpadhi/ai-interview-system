# Risk Analysis & Mitigation Strategies

## 🔴 High-Risk Issues

### Risk 1: LLM Output Parsing Failures
**Problem**: Your agents expect structured JSON from LLM responses. LLMs are unreliable at consistent formatting.

**Failure Mode**:
```python
# You expect:
{"score": 8, "reasoning": "..."}

# You might get:
"Here's my evaluation:\n\nScore: 8\nReasoning: ..."
# Or even worse:
{"score": 8, "reasoning": "...", }  # Trailing comma = invalid JSON
```

**Mitigation**:
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

class EvaluationOutput(BaseModel):
    technical_accuracy: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    depth: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    reasoning: str = Field(min_length=50)
    
    @validator('*', pre=True)
    def handle_string_numbers(cls, v):
        if isinstance(v, str) and v.replace('.', '').isdigit():
            return float(v)
        return v

# Use with retry logic
parser = PydanticOutputParser(pydantic_object=EvaluationOutput)

async def safe_parse(response: str, max_retries: int = 2) -> EvaluationOutput:
    for attempt in range(max_retries):
        try:
            return parser.parse(response)
        except ValidationError as e:
            if attempt < max_retries - 1:
                # Ask LLM to fix its output
                response = await llm.agenerate(
                    f"Fix this JSON to match the schema:\n{response}\n\nError: {e}"
                )
            else:
                return get_fallback_evaluation()
```

---

### Risk 2: ChromaDB Query Performance at Scale
**Problem**: 700 questions is fine, but ChromaDB's default HNSW index degrades with high-dimensional embeddings and complex filters.

**Failure Mode**:
- Queries take >2 seconds with multiple metadata filters
- Results become inconsistent with `where` clauses

**Mitigation**:
```python
# 1. Optimize collection creation
collection = client.create_collection(
    name="interview_questions",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 32,  # Increase connections (default 16)
        "hnsw:ef_construction": 200,  # Better index quality
        "hnsw:ef_search": 100  # Better search quality
    }
)

# 2. Use pre-filtering strategy (filter before vector search)
# Instead of:
results = collection.query(query_embeddings=[emb], where={"difficulty": "medium"})

# Do:
# First get IDs matching filter (fast metadata lookup)
filtered_ids = collection.get(where={"difficulty": "medium"})["ids"]
# Then vector search only within those
results = collection.query(
    query_embeddings=[emb],
    where={"id": {"$in": filtered_ids[:100]}}  # Limit search space
)

# 3. Add query caching
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve(query_hash: str, difficulty: str, topic: str):
    # Cache frequent query patterns
    pass
```

---

### Risk 3: Prompt Length Explosion
**Problem**: Your CoT evaluation prompt with full rubric can exceed context limits, especially for long candidate responses.

**Token Budget (GPT-4)**:
```
System prompt:        ~500 tokens
Question text:        ~100 tokens
Candidate response:   ~500-2000 tokens (varies!)
Full rubric JSON:     ~800 tokens
CoT instructions:     ~400 tokens
---------------------------------
Total:                2300-3800 tokens INPUT

With 8K context, you have headroom.
With complex coding responses, you don't.
```

**Mitigation**:
```python
def truncate_response(response: str, max_tokens: int = 1500) -> str:
    """Smart truncation preserving code blocks"""
    # Use tiktoken for accurate counting
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(response)
    
    if len(tokens) <= max_tokens:
        return response
    
    # Preserve code blocks if present
    if "```" in response:
        # Extract and prioritize code
        code_blocks = extract_code_blocks(response)
        prose = extract_prose(response)
        # Truncate prose first, keep code
        return reconstruct(code_blocks, truncate(prose, max_tokens - count(code_blocks)))
    
    return enc.decode(tokens[:max_tokens]) + "\n[Response truncated]"

def compress_rubric(rubric: dict, question_type: str) -> dict:
    """Send only relevant rubric sections"""
    relevant_criteria = CRITERIA_BY_TYPE[question_type]
    return {k: v for k, v in rubric["criteria"].items() if k in relevant_criteria}
```

---

### Risk 4: SQLite Write Contention
**Problem**: Your design has multiple concurrent writes per interview turn (conversation, evaluation, agent_traces, session_state).

**Failure Mode**:
```
sqlite3.OperationalError: database is locked
```

**Mitigation**:
```python
# 1. Enable WAL mode (do this ONCE at DB creation)
import sqlite3
conn = sqlite3.connect("interview.db")
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")  # Faster writes
conn.execute("PRAGMA busy_timeout=5000;")  # Wait 5s before failing

# 2. Use connection pooling with write queue
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "sqlite:///interview.db",
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    connect_args={"check_same_thread": False}
)

# 3. Batch writes within transaction
async def save_turn_results(interview_id: str, results: dict):
    async with engine.begin() as conn:
        # All writes in single transaction
        await conn.execute(insert_conversation, results["conversation"])
        await conn.execute(insert_evaluation, results["evaluation"])
        await conn.execute(update_session_state, results["state"])
        # Commit happens automatically
```

---

## 🟡 Medium-Risk Issues

### Risk 5: Embedding Model Cold Start
**Problem**: First request loads the embedding model into memory (~2-3 seconds).

**Mitigation**:
```python
# Preload at startup
class EmbeddingService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            # Warm up with dummy encoding
            cls._instance.model.encode("warmup", normalize_embeddings=True)
        return cls._instance

# In FastAPI startup
@app.on_event("startup")
async def startup():
    EmbeddingService()  # Preload model
```

---

### Risk 6: Self-RAG Decision Inconsistency
**Problem**: LLM might give different decisions for similar contexts, causing unpredictable retrieval patterns.

**Mitigation**:
```python
# Add deterministic guardrails around LLM decision
async def should_retrieve(self, state: InterviewState) -> bool:
    # Rule-based overrides (deterministic)
    if state["question_count"] == 0:
        return True
    if not self.cached_candidates:
        return True
    if state["question_count"] >= 5 and state["question_count"] % 3 == 0:
        return True  # Force fresh retrieval every 3rd question after 5
    
    # Only use LLM for edge cases
    last_topic = state["current_question"]["topic"]
    cached_topics = {q["topic"] for q in self.cached_candidates}
    
    if last_topic in cached_topics:
        # LLM decides: continue same topic or pivot?
        return await self._llm_should_retrieve(state)
    else:
        # Different topic needed, must retrieve
        return True
```

---

### Risk 7: Feedback Tone Drift
**Problem**: LLM might generate inconsistent tone—overly harsh or sycophantic—across different interviews.

**Mitigation**:
```python
# Add tone calibration in prompt
FEEDBACK_PROMPT = """
## Tone Guidelines
- Score >= 8: Congratulatory but brief. Focus on what they got right.
- Score 6-7: Balanced. Acknowledge strengths, gently note improvements.
- Score 4-5: Supportive. Emphasize this is a learning opportunity.
- Score < 4: Encouraging. Focus on effort, provide clear path forward.

NEVER use these phrases:
- "Unfortunately..." (sounds negative)
- "You failed to..." (sounds accusatory)  
- "Perfect!" (sounds sycophantic)
- "Great job!" without specifics (empty praise)

ALWAYS:
- Start with something specific they did well
- Frame gaps as "areas to explore further"
- End with actionable next step
"""

# Add post-generation validation
async def validate_tone(feedback: str, score: float) -> str:
    banned_phrases = ["unfortunately", "failed to", "wrong", "incorrect"]
    
    for phrase in banned_phrases:
        if phrase in feedback.lower():
            # Regenerate with stricter prompt
            return await regenerate_feedback(feedback, score)
    
    return feedback
```

---

### Risk 8: CRAG Infinite Refinement
**Problem**: Query refinement might not improve results, leading to wasted LLM calls.

**Mitigation**:
```python
async def retrieve_with_correction(self, query: str, ...) -> List[dict]:
    seen_queries = {query}  # Track to avoid loops
    
    for attempt in range(self.max_correction_attempts + 1):
        results = await self._retrieve(query, ...)
        grading = await self._grade_documents(results, query)
        
        if grading.quality in ["HIGH", "MEDIUM"]:
            return self._filter_results(results, grading)
        
        if attempt < self.max_correction_attempts:
            refined = await self._refine_query(query, grading.feedback)
            
            # Check if refinement is actually different
            if refined in seen_queries or self._similarity(query, refined) > 0.9:
                # Refinement not helping, try broadening instead
                query = self._broaden_query(query)
            else:
                query = refined
                seen_queries.add(query)
        else:
            return results[:3]  # Best effort
```

---

## 🟢 Low-Risk (But Worth Noting)

### Risk 9: Rubric Coverage Gaps
**Problem**: Automated rubric generation might miss edge cases for certain question types.

**Mitigation**: Flag questions without high-confidence rubrics for manual review.

### Risk 10: Code Validator False Positives
**Problem**: AST validation catches syntax errors but not semantic issues.

**Mitigation**: Clearly scope as "syntax check only" in evaluation output. Full execution is Phase 2.

---

## Pre-Launch Checklist

### Before Demo
- [ ] Run 10 complete interviews end-to-end
- [ ] Test with adversarial inputs (empty responses, very long responses, off-topic)
- [ ] Verify LangSmith traces are capturing all agent reasoning
- [ ] Check ChromaDB index is persisted correctly
- [ ] Confirm SQLite has WAL mode enabled
- [ ] Test circuit breakers trigger correctly
- [ ] Validate all fallback responses are sensible

### Interview Prep
- [ ] Have 3 pre-recorded demo runs ready (in case live demo fails)
- [ ] Prepare LangSmith trace walkthrough showing Self-RAG decisions
- [ ] Document one CRAG correction example with before/after queries
- [ ] Be ready to explain: "Why not just use a simple RAG pipeline?"
- [ ] Know your latency numbers: avg per agent, p95, total turn time