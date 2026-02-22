# AI Interview System - Final Architecture (V1)

## Design Principles

1. **Natural Interview Flow**: Hide internal scoring from user; feedback + question delivered conversationally
2. **Latency Optimization**: Dual-model strategy + parallel execution + topic-aware caching
3. **Production-Grade**: Validation gates, circuit breakers, graceful degradation
4. **Time-Bounded**: Real interviews have time limits; system respects them

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  POST /start          → Creates plan, pre-caches topics, returns Q1  │   │
│  │  POST /submit_response → Returns feedback + next question (BUNDLED)  │   │
│  │                         Scores HIDDEN from user                      │   │
│  │  POST /end            → Returns final report WITH all scores         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DUAL MODEL CONFIGURATION                             │
│                                                                              │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │   Qwen2.5-14B (5-bit)       │  │   Qwen2.5-7B (5-bit)                │  │
│  │   "Complex Model"           │  │   "Fast Model"                      │  │
│  │                             │  │                                     │  │
│  │   Used for:                 │  │   Used for:                         │  │
│  │   • Interview Plan (start)  │  │   • CRAG grading                    │  │
│  │   • CoT Evaluation          │  │   • ReAct selection                 │  │
│  │   • Follow-up generation    │  │   • Reflection steps                │  │
│  │   • Clarification generation│  │   • Feedback generation             │  │
│  │                             │  │                                     │  │
│  │   Latency: ~2-3s per call   │  │   Latency: ~1-1.5s per call         │  │
│  └─────────────────────────────┘  └─────────────────────────────────────┘  │
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
│  │  • Pre-caches topic question batches at /start                       │   │
│  │  • Rule-based routing thereafter (0 LLM calls)                       │   │
│  │  • Time constraint enforcement                                        │   │
│  │  • Difficulty adaptation                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Validation Gates                              │   │
│  │                                                                       │   │
│  │  Evaluator Gate:           Feedback Gate:         Question Gate:     │   │
│  │  • Scores in 0-10 range    • Length 50-500 words  • Question present │   │
│  │  • Reasoning >50 chars     • No forbidden phrases • Valid type       │   │
│  │  • Variance <3.0           • No sycophancy at     • Estimated time   │   │
│  │  • Required fields           low scores             fits budget      │   │
│  │                            • No score leakage                        │   │
│  │                                                                       │   │
│  │  Circuit Breaker: Max 1 retry per agent │ Fallback on failure        │   │
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
│  │             └──────────┬───────────────┘                             │   │
│  │                        │  FAN-IN                                     │   │
│  │                        ▼                                             │   │
│  │               ┌──────────────┐                                       │   │
│  │               │  Supervisor  │                                       │   │
│  │               │    Check     │                                       │   │
│  │               │ (Rule-based) │                                       │   │
│  │               └──────────────┘                                       │   │
│  │                                                                       │   │
│  │  Latency: Evaluator(2-3s) + max(Feedback, QS)(1.5-2s) = 3.5-5s      │   │
│  │  vs Sequential: 5-7s (30-40% improvement)                            │   │
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
│  │                                                                       │   │
│  │  Rubric Handling:          Tools:               Output (INTERNAL):   │   │
│  │  • Static (DB lookup)      • rubric_checker     • overall_score      │   │
│  │  • Dynamic (for follow-ups)• code_validator     • key_points_missed  │   │
│  │                              (AST syntax)       • misconceptions     │   │
│  │                                                 • reasoning          │   │
│  │  Does NOT:                                                           │   │
│  │  • Generate feedback        LLM Calls: 2 (CoT + Reflection)          │   │
│  │  • Generate questions       Latency: ~2.5-3.5s                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Feedback Agent                                │   │
│  │                                                                       │   │
│  │  Responsibility:           Patterns:            Model Assignment:    │   │
│  │  • Generate constructive   • Structured Output  • Feedback: 7B      │   │
│  │    feedback (no scores!)   • Anti-sycophancy                         │   │
│  │  • Acknowledge strengths     validation                              │   │
│  │  • Hint at gaps naturally                                            │   │
│  │                                                                       │   │
│  │  Structured Output:        Caching:             Does NOT:            │   │
│  │  • strength_acknowledgment • Concept cache      • Expose scores      │   │
│  │  • gap_hint (implicit)       (session-isolated) • Generate questions │   │
│  │  • transition_phrase       • Avoids repeated    • Say "you missed X" │   │
│  │                              concept lookups                         │   │
│  │  Anti-Sycophancy:                                                    │   │
│  │  • Score-appropriate tone  LLM Calls: 1                              │   │
│  │  • No "Great job!" at <6   Latency: ~1-1.5s                          │   │
│  │  • Direct correction style                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Question Selector Agent                         │   │
│  │                                                                       │   │
│  │  Responsibility:           Operating Modes:     Model Assignment:    │   │
│  │  • ALL question decisions  • RETRIEVE: Cache    • CRAG grade: 7B    │   │
│  │  • Mode determination        + RAG fallback     • ReAct select: 7B  │   │
│  │  • Time budget enforcement • FOLLOW_UP: LLM gen • Follow-up gen: 14B │   │
│  │                            • CLARIFY: LLM gen   • Clarify gen: 14B   │   │
│  │                                                                       │   │
│  │  Caching Strategy:         Time-Aware:          Does NOT:            │   │
│  │  • Topic-aware batches     • Filter by          • Evaluate answers   │   │
│  │  • Composite key:            estimated_time     • Generate feedback  │   │
│  │    {session}:topic:        • Skip long Qs if                         │   │
│  │    {topic}:{difficulty}      time running out                        │   │
│  │  • Partial reuse tracking                                            │   │
│  │  • TTL based on CRAG grade                                           │   │
│  │  • LRU eviction                                                      │   │
│  │                                                                       │   │
│  │  RETRIEVE Mode Flow:       LLM Calls: 1-2 (RETRIEVE) or 2 (GENERATE) │   │
│  │  Cache hit → select        Latency: ~1.5-2s (cache hit)              │   │
│  │  Cache miss → CRAG → cache          ~2-2.5s (cache miss)             │   │
│  │                                     ~2.5-3s (generate)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Per-Turn Summary:                                                          │
│  • Total LLM Calls: 3-4 (down from 5-7)                                    │
│  • Total Latency: 3.5-5s (down from 6-8s)                                  │
│  • 14B Calls: 1-2 (heavy lifting)                                          │
│  • 7B Calls: 2-3 (fast decisions)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SESSION-ISOLATED CACHE LAYER                         │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    InterviewCacheStore (Singleton)                   │   │
│  │                                                                       │   │
│  │  Cache Types:                                                        │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────┐   │   │
│  │  │  Topic Question Cache   │  │     Concept Cache               │   │   │
│  │  │                         │  │                                 │   │   │
│  │  │  Key Pattern:           │  │  Key Pattern:                   │   │   │
│  │  │  {session}:topic:       │  │  {session}:concept:             │   │   │
│  │  │  {topic}:{difficulty}   │  │  {concept_name}                 │   │   │
│  │  │                         │  │                                 │   │   │
│  │  │  Value:                 │  │  Value:                         │   │   │
│  │  │  • questions: List[dict]│  │  • explanation: str             │   │   │
│  │  │  • used_ids: Set[str]   │  │  • simple_explanation: str      │   │   │
│  │  │  • crag_grade: str      │  │  • examples: List[str]          │   │   │
│  │  │                         │  │                                 │   │   │
│  │  │  Used by:               │  │  Used by:                       │   │   │
│  │  │  Question Selector      │  │  Feedback Agent                 │   │   │
│  │  └─────────────────────────┘  └─────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Cache Features:                                                     │   │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐    │   │
│  │  │  Composite Key  │ │  Grade-based    │ │  Partial Reuse      │    │   │
│  │  │                 │ │  TTL            │ │  Tracking           │    │   │
│  │  │  Session +      │ │                 │ │                     │    │   │
│  │  │  Type +         │ │  HIGH: 30 min   │ │  used_ids tracks    │    │   │
│  │  │  Identifier     │ │  MEDIUM: 15 min │ │  consumed questions │    │   │
│  │  │                 │ │  LOW: 5 min     │ │  from batch         │    │   │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────────┘    │   │
│  │  ┌─────────────────┐ ┌─────────────────┐                            │   │
│  │  │  LRU Eviction   │ │  Thread Safety  │                            │   │
│  │  │                 │ │                 │                            │   │
│  │  │  Max 50 entries │ │  asyncio.Lock   │                            │   │
│  │  │  per session    │ │  for writes     │                            │   │
│  │  └─────────────────┘ └─────────────────┘                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC RAG SERVICE                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      CRAG (Corrective RAG)                           │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                    RETRIEVE Mode Flow                        │   │   │
│  │   │                                                               │   │   │
│  │   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │   │   │
│  │   │  │ Topic Cache │    │  Cache Miss │    │    CRAG     │      │   │   │
│  │   │  │   Lookup    │ →  │  ChromaDB   │ →  │   Grading   │      │   │   │
│  │   │  │             │    │  Retrieval  │    │    (7B)     │      │   │   │
│  │   │  │ Cache hit?  │    │             │    │             │      │   │   │
│  │   │  │ Has unused? │    │ Time-aware  │    │ HIGH: Cache │      │   │   │
│  │   │  │             │    │ filtering   │    │ MED: Filter │      │   │   │
│  │   │  │   ↓ YES     │    │             │    │ LOW: Retry  │      │   │   │
│  │   │  │ Skip to     │    │             │    │             │      │   │   │
│  │   │  │ Selection   │    │             │    │             │      │   │   │
│  │   │  └─────────────┘    └─────────────┘    └──────┬──────┘      │   │   │
│  │   │                                                │              │   │   │
│  │   │                                   ┌────────────┴────────────┐│   │   │
│  │   │                                   │   Corrective Actions    ││   │   │
│  │   │                                   │  • Query Refinement     ││   │   │
│  │   │                                   │  • Query Decomposition  ││   │   │
│  │   │                                   │  • Fallback Broadening  ││   │   │
│  │   │                                   │  • Max 2 retry attempts ││   │   │
│  │   │                                   └─────────────────────────┘│   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │                   ReAct Selection (7B)                       │   │   │
│  │   │                                                               │   │   │
│  │   │  Input: Candidate questions (from cache or fresh retrieval)  │   │   │
│  │   │  Process: Reason about best fit for interview context        │   │   │
│  │   │  Output: Selected question + mark as used in cache           │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Time-Aware Filtering:                                               │   │
│  │  • estimated_time_minutes <= remaining_time                          │   │
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
│  │    questions     │  │  • conversations │  │    all_rubrics   │          │
│  │  • ml_concepts   │  │  • evaluations   │  │    .json         │          │
│  │  • code_solutions│  │  • session_state │  │  • prompts/      │          │
│  │    (Phase 2)     │  │  • agent_traces  │  │    *.yaml        │          │
│  │                  │  │                  │  │  • logs/         │          │
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
│  │    chains        │  │  • Throughput    │  │  • Correlation   │          │
│  │                  │  │  • Time budget   │  │    IDs           │          │
│  │                  │  │    utilization   │  │                  │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State Structure

```python
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime
from langgraph.graph.message import add_messages

class InterviewState(TypedDict):
    # Conversation history
    messages: Annotated[list, add_messages]
    
    # Current turn data
    current_question: Optional[dict]
    candidate_response: Optional[str]
    current_evaluation: Optional[dict]  # INTERNAL ONLY - never exposed to user
    current_feedback: Optional[str]     # User-facing feedback (no scores)
    
    # Interview metadata
    interview_id: str
    stage: Literal["init", "planning", "questioning", "evaluating", "feedback", "complete"]
    question_count: int
    
    # Interview plan (from Plan-and-Execute)
    interview_plan: Optional[dict]  # Topic sequence, difficulty curve
    
    # Time tracking
    interview_start_time: datetime
    time_budget_minutes: int  # e.g., 30 or 45 minutes
    
    # Adaptive context
    topics_covered: list[str]
    difficulty_level: Literal["easy", "medium", "hard"]
    performance_trajectory: list[float]  # INTERNAL - for difficulty adaptation
    
    # Question flow tracking
    question_mode: Literal["retrieve", "follow_up", "clarify"]
    follow_up_count: int
    conversation_thread: list[str]
    
    # Control flags
    should_continue: bool
    needs_human_review: bool
    error_state: Optional[dict]
    end_reason: Optional[str]  # "completed", "time_up", "max_questions", "user_ended"
```

---

## Session-Isolated Cache Store

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from collections import OrderedDict
import asyncio

@dataclass
class CacheEntry:
    value: dict
    created_at: datetime
    ttl_seconds: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


@dataclass
class TopicQuestionCache:
    """Cached question batch for a topic"""
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
    Singleton cache store with session isolation.
    
    Features:
    - Composite keys: {session_id}:{type}:{identifier}
    - Grade-based TTL: HIGH=30min, MEDIUM=15min, LOW=5min
    - Partial reuse tracking for question batches
    - LRU eviction (max 50 entries per session)
    - Thread-safe async operations
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: OrderedDict[str, CacheEntry] = OrderedDict()
            cls._instance._max_entries_per_session = 50
        return cls._instance
    
    # TTL configuration by CRAG grade
    TTL_BY_GRADE = {
        "HIGH": 1800,    # 30 minutes
        "MEDIUM": 900,   # 15 minutes
        "LOW": 300       # 5 minutes
    }
    
    # ─────────────────────────────────────────────────────────────────
    # TOPIC QUESTION CACHE
    # ─────────────────────────────────────────────────────────────────
    
    def _topic_key(self, session_id: str, topic: str, difficulty: str) -> str:
        return f"{session_id}:topic:{topic}:{difficulty}"
    
    async def get_topic_questions(
        self, 
        session_id: str, 
        topic: str, 
        difficulty: str
    ) -> Optional[TopicQuestionCache]:
        """Get cached questions for topic, if available and not exhausted"""
        key = self._topic_key(session_id, topic, difficulty)
        
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                return None
            
            if entry.is_expired:
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            
            topic_cache = entry.value
            if not topic_cache.has_unused:
                return None  # All questions used
            
            return topic_cache
    
    async def set_topic_questions(
        self,
        session_id: str,
        topic: str,
        difficulty: str,
        questions: list[dict],
        crag_grade: str = "MEDIUM"
    ) -> None:
        """Cache question batch for topic"""
        key = self._topic_key(session_id, topic, difficulty)
        ttl = self.TTL_BY_GRADE.get(crag_grade, 900)
        
        topic_cache = TopicQuestionCache(
            questions=questions,
            used_ids=set(),
            crag_grade=crag_grade
        )
        
        async with self._lock:
            self._cache[key] = CacheEntry(
                value=topic_cache,
                created_at=datetime.now(),
                ttl_seconds=ttl,
                metadata={"type": "topic_questions", "session_id": session_id}
            )
            self._cache.move_to_end(key)
            await self._evict_if_needed(session_id)
    
    async def mark_question_used(
        self,
        session_id: str,
        topic: str,
        difficulty: str,
        question_id: str
    ) -> None:
        """Mark question as used (partial reuse tracking)"""
        key = self._topic_key(session_id, topic, difficulty)
        
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                entry.value.mark_used(question_id)
    
    # ─────────────────────────────────────────────────────────────────
    # CONCEPT CACHE (for Feedback Agent)
    # ─────────────────────────────────────────────────────────────────
    
    def _concept_key(self, session_id: str, concept_name: str) -> str:
        return f"{session_id}:concept:{concept_name}"
    
    async def get_concept(
        self, 
        session_id: str, 
        concept_name: str
    ) -> Optional[dict]:
        """Get cached concept explanation"""
        key = self._concept_key(session_id, concept_name)
        
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired:
                if entry:
                    del self._cache[key]
                return None
            
            self._cache.move_to_end(key)
            return entry.value
    
    async def set_concept(
        self,
        session_id: str,
        concept_name: str,
        concept_data: dict
    ) -> None:
        """Cache concept explanation (long TTL - concepts are stable)"""
        key = self._concept_key(session_id, concept_name)
        
        async with self._lock:
            self._cache[key] = CacheEntry(
                value=concept_data,
                created_at=datetime.now(),
                ttl_seconds=3600,  # 1 hour - concepts don't change
                metadata={"type": "concept", "session_id": session_id}
            )
            self._cache.move_to_end(key)
            await self._evict_if_needed(session_id)
    
    # ─────────────────────────────────────────────────────────────────
    # CACHE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    
    async def _evict_if_needed(self, session_id: str) -> None:
        """LRU eviction for session entries"""
        session_keys = [
            k for k in self._cache.keys() 
            if k.startswith(f"{session_id}:")
        ]
        
        while len(session_keys) > self._max_entries_per_session:
            oldest_key = session_keys.pop(0)
            del self._cache[oldest_key]
    
    async def clear_session(self, session_id: str) -> None:
        """Clear all cache entries for a session"""
        async with self._lock:
            keys_to_delete = [
                k for k in self._cache.keys() 
                if k.startswith(f"{session_id}:")
            ]
            for key in keys_to_delete:
                del self._cache[key]
    
    async def pre_warm_topics(
        self,
        session_id: str,
        rag_service: "AgenticRAGService",
        interview_plan: dict
    ) -> None:
        """
        Pre-fetch and cache questions for planned topics.
        Called during /start after plan creation.
        """
        topic_sequence = interview_plan.get("topic_sequence", [])
        difficulty_curve = interview_plan.get("difficulty_curve", [])
        
        for topic, difficulty in zip(topic_sequence[:5], difficulty_curve[:5]):
            # Check if already cached
            existing = await self.get_topic_questions(session_id, topic, difficulty)
            if existing:
                continue
            
            # Retrieve and cache
            questions, grade = await rag_service.retrieve_batch(
                topic=topic,
                difficulty=difficulty,
                n=5
            )
            
            await self.set_topic_questions(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                questions=questions,
                crag_grade=grade
            )
```

---

## Supervisor Agent (Plan-and-Execute + Rule-Based Routing)

```python
class SupervisorAgent:
    """
    - Plan-and-Execute: Creates interview plan at /start (1 LLM call, 14B)
    - Pre-warms topic caches after plan creation
    - OODA Loop: Rule-based routing and decisions (0 LLM calls per turn)
    - Time enforcement: Ends interview when time budget exhausted
    """
    
    def __init__(
        self, 
        complex_llm: BaseLLM,
        cache_store: InterviewCacheStore,
        rag_service: AgenticRAGService
    ):
        self.complex_llm = complex_llm  # Qwen2.5-14B
        self.cache_store = cache_store
        self.rag_service = rag_service
        self.validation_gates = ValidationGateRegistry()
        self.circuit_breaker = CircuitBreaker(max_retries=1)
    
    # ─────────────────────────────────────────────────────────────────
    # PLAN-AND-EXECUTE (Called once at /start)
    # ─────────────────────────────────────────────────────────────────
    
    async def create_interview_plan(self, state: InterviewState) -> InterviewState:
        """
        Creates structured interview plan and pre-warms cache.
        Uses 14B model for quality planning.
        """
        target_questions = self._calculate_target_questions(state["time_budget_minutes"])
        
        prompt = INTERVIEW_PLAN_PROMPT.format(
            difficulty=state["difficulty_level"],
            focus_topics=state.get("focus_topics", []),
            time_budget=state["time_budget_minutes"],
            target_questions=target_questions
        )
        
        response = await self.complex_llm.agenerate(prompt)
        plan = self._parse_plan(response.content)
        
        # Pre-warm cache with first 5 topics
        await self.cache_store.pre_warm_topics(
            session_id=state["interview_id"],
            rag_service=self.rag_service,
            interview_plan=plan
        )
        
        return {
            **state,
            "interview_plan": plan,
            "stage": "questioning"
        }
    
    def _calculate_target_questions(self, time_budget: int) -> int:
        """~4-5 minutes per question including response time"""
        return max(5, min(12, time_budget // 4))
    
    def _parse_plan(self, content: str) -> dict:
        """
        Parse LLM output into structured plan.
        
        Expected structure:
        {
          "topic_sequence": ["ml_fundamentals", "optimization", ...],
          "difficulty_curve": ["medium", "medium", "hard", ...],
          "time_allocation": [5, 5, 4, ...],
          "focus_areas": ["gradient_descent", "regularization", ...]
        }
        """
        return parse_json_safely(content)
    
    # ─────────────────────────────────────────────────────────────────
    # OODA LOOP (Rule-based, called every turn)
    # ─────────────────────────────────────────────────────────────────
    
    async def validate_and_decide(self, state: InterviewState) -> InterviewState:
        """
        Rule-based OODA loop. No LLM calls.
        Called after parallel Feedback + Question Selector execution.
        """
        # OBSERVE
        observation = self._observe(state)
        
        # ORIENT
        analysis = self._orient(observation)
        
        # DECIDE
        should_continue, end_reason = self._decide_continuation(analysis, state)
        new_difficulty = self._maybe_adjust_difficulty(analysis, state)
        
        # ACT (update state)
        return {
            **state,
            "should_continue": should_continue,
            "end_reason": end_reason,
            "difficulty_level": new_difficulty,
            "performance_trajectory": state["performance_trajectory"] + [
                state["current_evaluation"]["overall_score"]
            ]
        }
    
    def _observe(self, state: InterviewState) -> Observation:
        elapsed = self._get_elapsed_minutes(state)
        remaining = state["time_budget_minutes"] - elapsed
        
        return Observation(
            current_stage=state["stage"],
            question_count=state["question_count"],
            question_mode=state.get("question_mode", "retrieve"),
            follow_up_count=state.get("follow_up_count", 0),
            performance_trajectory=state["performance_trajectory"],
            last_score=state.get("current_evaluation", {}).get("overall_score"),
            topics_covered=state["topics_covered"],
            difficulty_level=state["difficulty_level"],
            elapsed_minutes=elapsed,
            remaining_minutes=remaining,
            target_questions=len(state.get("interview_plan", {}).get("topic_sequence", []))
        )
    
    def _orient(self, obs: Observation) -> Analysis:
        trajectory = obs.performance_trajectory
        return Analysis(
            performance_trend=self._calculate_trend(trajectory),
            avg_score=sum(trajectory) / len(trajectory) if trajectory else 0,
            is_struggling=obs.last_score and obs.last_score < 4.0,
            is_excelling=obs.last_score and obs.last_score >= 9.0,
            time_pressure=obs.remaining_minutes < 5,
            time_critical=obs.remaining_minutes < 2,
            questions_remaining=obs.target_questions - obs.question_count,
            coverage_gaps=self._identify_topic_gaps(obs.topics_covered)
        )
    
    def _decide_continuation(
        self, analysis: Analysis, state: InterviewState
    ) -> tuple[bool, Optional[str]]:
        """Rule-based continuation decision"""
        
        # Time-based termination (highest priority)
        if analysis.time_critical:
            return False, "time_up"
        
        # Question count termination
        target = len(state.get("interview_plan", {}).get("topic_sequence", [10]))
        if state["question_count"] >= target:
            return False, "completed"
        
        # Performance-based early termination
        if analysis.avg_score < 3.0 and state["question_count"] >= 5:
            return False, "performance_threshold"
        
        return True, None
    
    def _maybe_adjust_difficulty(
        self, analysis: Analysis, state: InterviewState
    ) -> str:
        """Adaptive difficulty based on performance trend"""
        current = state["difficulty_level"]
        
        # Don't adjust if time pressure
        if analysis.time_pressure:
            return current
        
        # Need at least 3 questions to assess trend
        if state["question_count"] < 3:
            return current
        
        if analysis.performance_trend == "improving" and analysis.avg_score >= 8.0:
            return {"easy": "medium", "medium": "hard", "hard": "hard"}[current]
        elif analysis.performance_trend == "declining" and analysis.avg_score < 5.0:
            return {"easy": "easy", "medium": "easy", "hard": "medium"}[current]
        
        return current
    
    def _calculate_trend(self, trajectory: list[float]) -> str:
        if len(trajectory) < 3:
            return "stable"
        recent = trajectory[-3:]
        if recent[-1] > recent[0] + 1.0:
            return "improving"
        elif recent[-1] < recent[0] - 1.0:
            return "declining"
        return "stable"
    
    def _get_elapsed_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return 0
        elapsed = datetime.now() - state["interview_start_time"]
        return elapsed.total_seconds() / 60
    
    def _identify_topic_gaps(self, topics_covered: list[str]) -> list[str]:
        all_topics = {"ml_fundamentals", "optimization", "deep_learning", 
                      "regularization", "evaluation", "statistics"}
        return list(all_topics - set(topics_covered))
```

---

## Validation Gates

```python
class ValidationGateRegistry:
    """
    Each agent has a validation gate checking output quality.
    Failed validation → retry (max 1) → fallback on second failure.
    """
    
    def __init__(self):
        self.gates = {
            "evaluator": EvaluatorValidationGate(),
            "feedback": FeedbackValidationGate(),
            "question_selector": QuestionSelectorValidationGate()
        }
    
    def get(self, agent_name: str) -> ValidationGate:
        return self.gates[agent_name]


class EvaluatorValidationGate:
    """Validates Evaluator Agent outputs"""
    
    def validate(self, output: dict) -> ValidationResult:
        checks = [
            self._scores_in_range(output),
            self._reasoning_provided(output),
            self._scores_consistent(output),
            self._required_fields_present(output)
        ]
        
        failed = [c for c in checks if not c.passed]
        return ValidationResult(
            is_valid=len(failed) == 0,
            failed_checks=failed,
            feedback=[c.message for c in failed]
        )
    
    def _scores_in_range(self, output: dict) -> Check:
        score_fields = ["technical_accuracy", "completeness", "depth", "clarity", "overall_score"]
        for field in score_fields:
            val = output.get(field)
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None and not (0 <= float(score) <= 10):
                return Check(passed=False, message=f"{field}={score} outside 0-10")
        return Check(passed=True)
    
    def _reasoning_provided(self, output: dict) -> Check:
        reasoning = output.get("reasoning", "")
        if len(reasoning) < 50:
            return Check(passed=False, message="Reasoning too short")
        return Check(passed=True)
    
    def _scores_consistent(self, output: dict) -> Check:
        scores = []
        for field in ["technical_accuracy", "completeness", "depth", "clarity"]:
            val = output.get(field)
            score = val.get("score") if isinstance(val, dict) else val
            if score is not None:
                scores.append(float(score))
        if scores and max(scores) - min(scores) > 5:
            return Check(passed=False, message="Score variance too high")
        return Check(passed=True)
    
    def _required_fields_present(self, output: dict) -> Check:
        required = ["overall_score", "reasoning"]
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
            "key_points_missed": [],
            "misconceptions": []
        }


class FeedbackValidationGate:
    """Validates Feedback Agent outputs with anti-sycophancy checks"""
    
    def validate(self, output: dict, evaluation_score: float) -> ValidationResult:
        checks = [
            self._appropriate_length(output["feedback_text"]),
            self._no_forbidden_phrases(output["feedback_text"]),
            self._no_sycophancy_at_low_scores(output["feedback_text"], evaluation_score),
            self._no_score_leakage(output["feedback_text"])
        ]
        
        failed = [c for c in checks if not c.passed]
        return ValidationResult(is_valid=len(failed) == 0, failed_checks=failed)
    
    def _appropriate_length(self, text: str) -> Check:
        words = len(text.split())
        if words < 20:
            return Check(passed=False, message="Feedback too short")
        if words > 200:
            return Check(passed=False, message="Feedback too long")
        return Check(passed=True)
    
    def _no_forbidden_phrases(self, text: str) -> Check:
        forbidden = [
            "you failed", "wrong answer", "incorrect", 
            "you don't understand", "completely wrong"
        ]
        text_lower = text.lower()
        for phrase in forbidden:
            if phrase in text_lower:
                return Check(passed=False, message=f"Forbidden: '{phrase}'")
        return Check(passed=True)
    
    def _no_sycophancy_at_low_scores(self, text: str, score: float) -> Check:
        """Prevent excessive praise when score is low"""
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
                return Check(
                    passed=False, 
                    message=f"Sycophantic '{phrase}' inappropriate for score {score:.1f}"
                )
        return Check(passed=True)
    
    def _no_score_leakage(self, text: str) -> Check:
        """Ensure no numeric scores appear in feedback"""
        import re
        score_patterns = [
            r'\b\d+\.?\d*/10\b',
            r'\bscored?\s+\d+',
            r'\b\d+\.?\d*\s*out of',
            r'rating[:\s]+\d+',
        ]
        for pattern in score_patterns:
            if re.search(pattern, text.lower()):
                return Check(passed=False, message="Score leaked in feedback")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        return {
            "feedback_text": "Thank you for your response. Let's continue.",
            "strength_acknowledgment": "",
            "gap_hint": "",
            "transition_phrase": "Moving on"
        }


class QuestionSelectorValidationGate:
    """Validates Question Selector outputs"""
    
    def validate(self, output: dict, remaining_minutes: float) -> ValidationResult:
        checks = [
            self._question_present(output),
            self._valid_question_type(output),
            self._time_appropriate(output, remaining_minutes)
        ]
        
        failed = [c for c in checks if not c.passed]
        return ValidationResult(is_valid=len(failed) == 0, failed_checks=failed)
    
    def _question_present(self, output: dict) -> Check:
        if not output.get("text"):
            return Check(passed=False, message="No question text")
        return Check(passed=True)
    
    def _valid_question_type(self, output: dict) -> Check:
        valid_types = ["retrieved", "follow_up", "clarification"]
        if output.get("question_type") not in valid_types:
            return Check(passed=False, message="Invalid question type")
        return Check(passed=True)
    
    def _time_appropriate(self, output: dict, remaining_minutes: float) -> Check:
        est_time = output.get("estimated_time_minutes", 5)
        if est_time > remaining_minutes + 2:
            return Check(passed=False, message="Question too long for remaining time")
        return Check(passed=True)
    
    def get_fallback(self) -> dict:
        return {
            "id": "fallback_001",
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4
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

## Feedback Agent (Structured Output + Anti-Sycophancy + Concept Caching)

```python
class FeedbackComponents(BaseModel):
    """Structured output to prevent sycophancy and control tone"""
    strength_acknowledgment: str = Field(
        description="1 sentence acknowledging what they did well. Empty if score < 5."
    )
    gap_hint: str = Field(
        description="Implicit hint about gaps WITHOUT stating 'you missed X'. "
                    "Frame as curiosity or natural follow-up context."
    )
    transition_phrase: str = Field(
        description="Natural transition to next question. "
                    "E.g., 'Building on that...', 'Now I'm curious about...'"
    )


class FeedbackAgent:
    """
    Generates user-facing feedback with NO scores or explicit gap statements.
    Uses structured output to prevent LLM sycophancy.
    Uses concept cache to avoid repeated lookups.
    """
    
    def __init__(
        self, 
        fast_llm: BaseLLM, 
        concept_tool: ConceptLookupTool,
        cache_store: InterviewCacheStore
    ):
        self.fast_llm = fast_llm  # Qwen2.5-7B
        self.concept_tool = concept_tool
        self.cache_store = cache_store
        self.output_parser = PydanticOutputParser(pydantic_object=FeedbackComponents)
    
    async def execute(self, state: InterviewState) -> InterviewState:
        eval_data = state["current_evaluation"]  # INTERNAL - never exposed
        score = eval_data["overall_score"]
        q_type = state["current_question"].get("question_type", "standard")
        
        # Get concept context for low scores (with caching)
        concept_context = ""
        if q_type == "standard" and score < 7.0:
            missed = eval_data.get("key_points_missed", [])[:1]
            for concept in missed:
                concept_data = await self._get_concept_cached(
                    session_id=state["interview_id"],
                    concept_name=concept
                )
                if concept_data:
                    concept_context = concept_data.get("simple_explanation", "")
        
        # Generate structured components
        components = await self._generate_components(state, score, concept_context)
        
        # Compose final feedback (programmatic, not LLM)
        feedback_text = self._compose_feedback(components, score)
        
        return {
            **state,
            "current_feedback": feedback_text,
            "stage": "feedback"
        }
    
    async def _get_concept_cached(
        self, 
        session_id: str, 
        concept_name: str
    ) -> Optional[dict]:
        """Get concept with cache lookup first"""
        # Check cache
        cached = await self.cache_store.get_concept(session_id, concept_name)
        if cached:
            return cached
        
        # Cache miss - fetch from RAG
        concept_data = await self.concept_tool.lookup(concept_name)
        
        if concept_data:
            # Cache for future use
            await self.cache_store.set_concept(
                session_id=session_id,
                concept_name=concept_name,
                concept_data=concept_data
            )
        
        return concept_data
    
    async def _generate_components(
        self, state: InterviewState, score: float, concept_context: str
    ) -> FeedbackComponents:
        """Generate structured feedback components using 7B model"""
        
        tone_guidance = self._get_tone_guidance(score)
        
        prompt = STRUCTURED_FEEDBACK_PROMPT.format(
            question=state["current_question"]["text"],
            response=state["candidate_response"],
            score_band=self._get_score_band(score),
            tone_guidance=tone_guidance,
            concept_context=concept_context,
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        response = await self.fast_llm.agenerate(prompt)
        return self.output_parser.parse(response.content)
    
    def _compose_feedback(self, components: FeedbackComponents, score: float) -> str:
        """Programmatic composition - removes LLM freedom to be sycophantic"""
        
        parts = []
        
        # Only include strength if score warrants it
        if score >= 5.0 and components.strength_acknowledgment:
            parts.append(components.strength_acknowledgment)
        
        # Always include gap hint (implicit, not "you missed X")
        if components.gap_hint:
            parts.append(components.gap_hint)
        
        # Transition phrase
        if components.transition_phrase:
            parts.append(components.transition_phrase)
        
        return " ".join(parts)
    
    def _get_tone_guidance(self, score: float) -> str:
        if score >= 8.0:
            return "Genuinely impressed. Brief acknowledgment, then move on."
        elif score >= 6.0:
            return "Encouraging but direct. Note the good parts, hint at depth opportunity."
        elif score >= 4.0:
            return "Supportive and constructive. Focus on the attempt, guide toward better."
        else:
            return "Patient and helpful. No praise openers. Direct but kind guidance."
    
    def _get_score_band(self, score: float) -> str:
        if score >= 8.0:
            return "high"
        elif score >= 5.0:
            return "medium"
        else:
            return "low"
```

---

## Question Selector Agent (Topic-Aware Caching, No Self-RAG)

```python
class QuestionSelectorAgent:
    """
    Owns ALL question decisions. Uses topic-aware caching for efficiency.
    No Self-RAG - cache lookup by topic replaces the decision.
    
    RETRIEVE mode flow:
    1. Cache lookup by {session}:topic:{topic}:{difficulty}
    2. Cache hit with unused questions → select from cache
    3. Cache miss or exhausted → CRAG retrieval → cache result
    4. ReAct selection from candidates
    """
    
    def __init__(
        self, 
        rag_service: AgenticRAGService, 
        fast_llm: BaseLLM,
        complex_llm: BaseLLM,
        cache_store: InterviewCacheStore
    ):
        self.rag = rag_service
        self.fast_llm = fast_llm      # Qwen2.5-7B
        self.complex_llm = complex_llm # Qwen2.5-14B
        self.cache_store = cache_store
    
    async def execute(self, state: InterviewState) -> InterviewState:
        remaining_time = self._get_remaining_minutes(state)
        
        # Mode determination (rule-based, no LLM)
        mode = self._determine_question_mode(state, remaining_time)
        
        # Execute mode
        if mode == "retrieve":
            question = await self._retrieve_question(state, remaining_time)
            new_follow_up_count = 0
            new_thread = [question["id"]]
        elif mode == "follow_up":
            question = await self._generate_follow_up(state)
            new_follow_up_count = state.get("follow_up_count", 0) + 1
            new_thread = state.get("conversation_thread", []) + [question["id"]]
        elif mode == "clarify":
            question = await self._generate_clarification(state)
            new_follow_up_count = state.get("follow_up_count", 0) + 1
            new_thread = state.get("conversation_thread", []) + [question["id"]]
        
        return {
            **state,
            "current_question": question,
            "question_mode": mode,
            "follow_up_count": new_follow_up_count,
            "conversation_thread": new_thread,
            "stage": "questioning"
        }
    
    def _get_remaining_minutes(self, state: InterviewState) -> float:
        if not state.get("interview_start_time"):
            return state.get("time_budget_minutes", 30)
        elapsed = (datetime.now() - state["interview_start_time"]).total_seconds() / 60
        return max(0, state["time_budget_minutes"] - elapsed)
    
    def _determine_question_mode(
        self, state: InterviewState, remaining_time: float
    ) -> str:
        """Rule-based mode determination"""
        
        # First question
        if state["question_count"] == 0:
            return "retrieve"
        
        # Time pressure: skip follow-ups, move to new topics
        if remaining_time < 5:
            return "retrieve"
        
        eval_data = state["current_evaluation"]
        score = eval_data["overall_score"]
        missed = eval_data.get("key_points_missed", [])
        misconceptions = eval_data.get("misconceptions", [])
        follow_ups = state.get("follow_up_count", 0)
        
        # Address misconceptions (priority)
        if misconceptions and follow_ups < 2:
            return "clarify"
        
        # Probe gaps for weak answers
        if score < 7.0 and missed and follow_ups < 2:
            return "follow_up"
        
        # One follow-up for decent but incomplete
        if 7.0 <= score < 8.0 and missed and follow_ups < 1:
            return "follow_up"
        
        return "retrieve"
    
    # ─────────────────────────────────────────────────────────────────
    # RETRIEVE MODE (Cache-first, CRAG fallback)
    # ─────────────────────────────────────────────────────────────────
    
    async def _retrieve_question(
        self, state: InterviewState, remaining_time: float
    ) -> dict:
        """
        Topic-aware caching replaces Self-RAG.
        
        Flow:
        1. Get next topic from plan
        2. Check cache for {session}:topic:{topic}:{difficulty}
        3. Cache hit with unused → select from cache
        4. Cache miss → CRAG retrieval → cache → select
        """
        topic = self._get_next_topic_from_plan(state)
        difficulty = state["difficulty_level"]
        session_id = state["interview_id"]
        
        # Step 1: Check cache
        cached = await self.cache_store.get_topic_questions(
            session_id=session_id,
            topic=topic,
            difficulty=difficulty
        )
        
        if cached and cached.has_unused:
            # Cache hit - select from unused questions
            candidates = cached.unused_questions
            crag_grade = cached.crag_grade
        else:
            # Cache miss - CRAG retrieval
            candidates, crag_grade = await self._crag_retrieve(
                topic=topic,
                difficulty=difficulty,
                remaining_time=remaining_time,
                exclude_ids=state.get("topics_covered", [])
            )
            
            # Cache the results
            await self.cache_store.set_topic_questions(
                session_id=session_id,
                topic=topic,
                difficulty=difficulty,
                questions=candidates,
                crag_grade=crag_grade
            )
        
        # Step 2: ReAct selection (7B)
        selected = await self._react_select(candidates, state)
        
        # Step 3: Mark as used for partial reuse
        await self.cache_store.mark_question_used(
            session_id=session_id,
            topic=topic,
            difficulty=difficulty,
            question_id=selected["id"]
        )
        
        # Add metadata
        selected["question_type"] = "retrieved"
        return selected
    
    async def _crag_retrieve(
        self,
        topic: str,
        difficulty: str,
        remaining_time: float,
        exclude_ids: list[str]
    ) -> tuple[list[dict], str]:
        """CRAG retrieval with grading"""
        
        # Time-aware filter
        filters = {
            "topic": topic,
            "difficulty": difficulty,
            "estimated_time_minutes": {"$lte": remaining_time}
        }
        
        # Initial retrieval
        candidates = await self.rag.retrieve(
            filters=filters,
            exclude_ids=exclude_ids,
            n=5
        )
        
        # CRAG grading (7B)
        grade = await self._grade_results(candidates, topic)
        
        if grade == "LOW" and len(candidates) > 0:
            # Retry with broader filters
            broader_filters = {"topic": topic}  # Remove difficulty constraint
            candidates = await self.rag.retrieve(
                filters=broader_filters,
                exclude_ids=exclude_ids,
                n=5
            )
            grade = await self._grade_results(candidates, topic)
        
        return candidates, grade
    
    async def _grade_results(self, candidates: list[dict], topic: str) -> str:
        """CRAG grading using 7B model"""
        if not candidates:
            return "LOW"
        
        prompt = CRAG_GRADING_PROMPT.format(
            topic=topic,
            candidates=[{"id": c["id"], "text": c["text"][:200]} for c in candidates]
        )
        
        response = await self.fast_llm.agenerate(prompt)
        grade = response.content.strip().upper()
        
        if grade not in ["HIGH", "MEDIUM", "LOW"]:
            return "MEDIUM"
        return grade
    
    async def _react_select(
        self, candidates: list[dict], state: InterviewState
    ) -> dict:
        """ReAct selection using 7B model"""
        prompt = REACT_SELECTION_PROMPT.format(
            candidates=[{
                "id": c["id"], 
                "text": c["text"], 
                "difficulty": c["difficulty"]
            } for c in candidates[:5]],
            difficulty_level=state["difficulty_level"],
            topics_covered=state["topics_covered"][-3:],
            performance_trend=self._get_performance_trend(state)
        )
        
        response = await self.fast_llm.agenerate(prompt)
        selected_id = self._parse_selection(response.content)
        
        # Find selected question
        for c in candidates:
            if c["id"] == selected_id:
                return c
        
        # Fallback to first candidate
        return candidates[0] if candidates else self._get_fallback_question()
    
    def _get_next_topic_from_plan(self, state: InterviewState) -> str:
        """Get next topic from interview plan"""
        plan = state.get("interview_plan", {})
        topic_sequence = plan.get("topic_sequence", [])
        topics_covered = state.get("topics_covered", [])
        
        # Find first uncovered topic
        for topic in topic_sequence:
            if topic not in topics_covered:
                return topic
        
        # All covered - cycle back
        return topic_sequence[0] if topic_sequence else "ml_fundamentals"
    
    # ─────────────────────────────────────────────────────────────────
    # FOLLOW_UP MODE (LLM Generated)
    # ─────────────────────────────────────────────────────────────────
    
    async def _generate_follow_up(self, state: InterviewState) -> dict:
        """Generate targeted follow-up using 14B model"""
        eval_data = state["current_evaluation"]
        original = state["current_question"]
        missed = eval_data.get("key_points_missed", [])[:2]
        
        prompt = FOLLOW_UP_GENERATION_PROMPT.format(
            original_question=original["text"],
            candidate_response=state["candidate_response"],
            missed_points=missed,
            topic=original["topic"]
        )
        
        response = await self.complex_llm.agenerate(prompt)  # 14B
        
        return {
            "id": f"{original['id']}_followup_{state.get('follow_up_count', 0) + 1}",
            "text": response.content.strip(),
            "question_type": "follow_up",
            "topic": original["topic"],
            "difficulty": original["difficulty"],
            "parent_question_id": original["id"],
            "target_concepts": missed,
            "estimated_time_minutes": 3
        }
    
    # ─────────────────────────────────────────────────────────────────
    # CLARIFY MODE (LLM Generated)
    # ─────────────────────────────────────────────────────────────────
    
    async def _generate_clarification(self, state: InterviewState) -> dict:
        """Generate clarification using 14B model"""
        eval_data = state["current_evaluation"]
        misconception = eval_data["misconceptions"][0]
        original = state["current_question"]
        
        prompt = CLARIFICATION_PROMPT.format(
            original_question=original["text"],
            candidate_response=state["candidate_response"],
            misconception=misconception
        )
        
        response = await self.complex_llm.agenerate(prompt)  # 14B
        
        return {
            "id": f"{original['id']}_clarify",
            "text": response.content.strip(),
            "question_type": "clarification",
            "topic": original["topic"],
            "difficulty": original["difficulty"],
            "parent_question_id": original["id"],
            "target_misconception": misconception,
            "estimated_time_minutes": 3
        }
    
    def _get_fallback_question(self) -> dict:
        return {
            "id": "fallback_001",
            "text": "Can you explain the bias-variance tradeoff?",
            "question_type": "retrieved",
            "topic": "ml_fundamentals",
            "difficulty": "medium",
            "estimated_time_minutes": 4
        }
```

---

## LangGraph Workflow (Parallel Execution)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_interview_graph(agents: AgentRegistry) -> StateGraph:
    """
    Parallel execution: Feedback and Question Selector run concurrently
    after Evaluator completes. Fan-out/Fan-in pattern.
    """
    graph = StateGraph(InterviewState)
    
    # Nodes
    graph.add_node("evaluator", agents.evaluator.execute)
    graph.add_node("feedback", agents.feedback.execute)
    graph.add_node("question_selector", agents.question_selector.execute)
    graph.add_node("supervisor_check", agents.supervisor.validate_and_decide)
    
    # Entry point
    graph.set_entry_point("evaluator")
    
    # Fan-out: Both depend on evaluator, run in parallel
    graph.add_edge("evaluator", "feedback")
    graph.add_edge("evaluator", "question_selector")
    
    # Fan-in: Supervisor waits for both
    graph.add_edge("feedback", "supervisor_check")
    graph.add_edge("question_selector", "supervisor_check")
    
    # End
    graph.add_edge("supervisor_check", END)
    
    return graph.compile(checkpointer=SqliteSaver.from_conn_string("interview.db"))


def build_start_graph(agents: AgentRegistry) -> StateGraph:
    """Graph for /start: Plan creation + first question"""
    graph = StateGraph(InterviewState)
    
    graph.add_node("create_plan", agents.supervisor.create_interview_plan)
    graph.add_node("first_question", agents.question_selector.execute)
    
    graph.set_entry_point("create_plan")
    graph.add_edge("create_plan", "first_question")
    graph.add_edge("first_question", END)
    
    return graph.compile()
```

---

## API Design (Scores Hidden from User)

```python
from pydantic import BaseModel
from datetime import datetime

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
    """User-facing response - NO scores, NO key_points_missed"""
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
    questions_asked: int
    time_taken_minutes: float
    topic_scores: dict[str, float]
    strengths: list[str]
    areas_for_improvement: list[str]
    detailed_evaluations: list[dict]


@app.post("/api/v1/interview/start")
async def start_interview(request: StartRequest) -> StartResponse:
    state = initialize_state(request)
    result = await start_graph.ainvoke(state)
    await save_state(result)
    
    return StartResponse(
        session_id=result["interview_id"],
        question={
            "text": result["current_question"]["text"],
            "topic": result["current_question"]["topic"],
            "estimated_time_minutes": result["current_question"]["estimated_time_minutes"]
        },
        time_budget_minutes=result["time_budget_minutes"],
        target_questions=len(result["interview_plan"]["topic_sequence"])
    )


@app.post("/api/v1/interview/submit_response")
async def submit_response(request: SubmitRequest) -> SubmitResponse:
    state = await load_state(request.session_id)
    state["candidate_response"] = request.response
    
    result = await interview_graph.ainvoke(state)
    await save_state(result)
    
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


@app.post("/api/v1/interview/end")
async def end_interview(session_id: str) -> dict:
    state = await load_state(session_id)
    report = generate_final_report(state)
    
    # Clear session cache
    await cache_store.clear_session(session_id)
    await mark_complete(session_id)
    
    return {"final_report": report}
```

---

## Latency Summary

| Stage | Model | Calls | Cache | Latency |
|-------|-------|-------|-------|---------|
| /start: Plan | 14B | 1 | — | ~2-3s |
| /start: Pre-warm | — | — | Write | ~1-2s |
| /start: First Q | 7B | 1-2 | Hit | ~1.5-2s |
| **Total /start** | | 2-3 | | **~4.5-7s** |
| | | | | |
| Evaluator (CoT + Reflect) | 14B + 7B | 2 | — | ~2.5-3.5s |
| Feedback (parallel) | 7B | 1 | Concept | ~1-1.5s |
| Question Selector (parallel) | 7B | 1-2 | Topic | ~1.5-2s |
| **Total /submit (parallel)** | | 3-4 | | **~3.5-5s** |
| | | | | |
| vs Previous (with Self-RAG) | | 4-5 | | ~4-5.5s |
| vs Sequential | | 4-5 | | ~5-7s |
| **Improvement** | | | | **30-50%** |

---

## Follow-up Loop Rules

| Condition | Mode | Max | Exit |
|-----------|------|-----|------|
| First question | RETRIEVE | — | — |
| Time < 5 min | RETRIEVE | — | Skip follow-ups |
| Misconception | CLARIFY | 2 | Resolved or max |
| Score < 7 + gaps | FOLLOW_UP | 2 | Score ≥ 8 or max |
| Score 7-8 + gaps | FOLLOW_UP | 1 | Score ≥ 8 or max |
| Score ≥ 8 | RETRIEVE | — | Always new topic |

---

## Cache Strategy Summary

| Cache Type | Key Pattern | TTL | Used By |
|------------|-------------|-----|---------|
| Topic Questions | `{session}:topic:{topic}:{difficulty}` | Grade-based (5-30 min) | Question Selector |
| Concepts | `{session}:concept:{name}` | 60 min (stable data) | Feedback Agent |

| Feature | Implementation |
|---------|----------------|
| Composite Key | Session-isolated, type-specific |
| Grade-based TTL | HIGH=30min, MEDIUM=15min, LOW=5min |
| Partial Reuse | `used_ids` tracks consumed questions |
| LRU Eviction | Max 50 entries per session |
| Thread Safety | `asyncio.Lock` for writes |