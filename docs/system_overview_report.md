# AI Interview System - Comprehensive Overview

Based on the architectural blueprints (`architecture(v2).md`), the current implementation status (`implementation_status.md`), and the source code structure, this document provides a deep-dive overview of the AI Interview System.

---

## 1. What We Have Built So Far (Sessions 1–7)

We have successfully established the foundational infrastructure, the data persistence layers, the advanced RAG (Retrieval-Augmented Generation) pipeline, the core orchestration primitives, and the first major intelligence node—the **EvaluatorAgent**.

### 1.1 Core Infrastructure
- **LLM Factory & Dual-Model Strategy:** (`src/utils/llm_factory.py`)
  - Integrated `qwen2.5:14b-instruct-q5_K_M` as the "Complex Model" for heavy lifting (CoT Evaluation, Interview Planning, Clarification generation).
  - Integrated `qwen2.5:7b-instruct-q5_K_M` as the "Fast Model" for rapid tasks (CRAG grading, feedback generation, ReAct selection, Reflective steps).
  - All LLMs are robustly wrapped using Langchain's `BaseChatModel` along with `with_structured_output()` and `with_retry()` for structural integrity.

### 1.2 Data & Persistence Layer
- **Vector DB & Embeddings:** (`src/data/vector_store.py`, `src/data/embeddings.py`)
  - **ChromaDB** with 3 distinct collections (`interview_questions`, `ml_concepts`, `code_solutions`).
  - Powered by the `BAAI/bge-base-en-v1.5` embeddings (768-dimensional, normalized for cosine distance).
- **Relational DB:** (`src/data/database.py`)
  - **SQLite** running in WAL (Write-Ahead Logging) mode with a 5-connection pool and `busy_timeout` of 5000ms.
  - Manages **4 types of memory** (`short-term buffer`, `episodic` in SQLite, `semantic` in ChromaDB, and `working` in LangGraph state). Supports tables for interviews, conversations, evaluations, session_state, and agent_traces.

### 1.3 Advanced Agentic RAG Pipeline
- **AgenticRAGService (Facade):** (`src/rag/agentic_rag.py`)
  - Implements the **Corrective RAG (CRAG)** subgraph using LangGraph's `StateGraph`. The flow is structured as: `retrieve → grade → (refine → retrieve)* → package_results`.
- **Hybrid DocumentGrader:** (`src/rag/grader.py`)
  - Evaluates retrieval relevance. Employs a heuristic + LLM hybrid approach. If relevance >= 0.75, it's HIGH. If <= 0.45, it's LOW. Borderline cases are deflected to the 7B LLM for a granular grade.
- **QueryRefiner:** (`src/rag/query_refiner.py`)
  - Modifies queries dynamically using three strategies (LLM Refinement, Topic Pivot, Structure Simplification) with anti-repeat safeguards (cosine matching > 0.85 rejected).
- **Session-Isolated CacheStore:** (`src/rag/cache.py`)
  - Dual-pool cache (Pool for topics: 10/session, Pool for concepts: 30/session) using per-session `asyncio.Lock`. Employs an atomic `select_and_mark()` to prevent race conditions during follow-up cache hits.

### 1.4 Orchestration & State Management
- **LangGraph State (`InterviewState`):** (`src/graph/state.py`)
  - Defined rigid boundaries using `TypedDict` and specific Graph Reducers (e.g., `last_value` for scalar replacements and `operator.add` for list appends).
- **Inter-Agent Contracts:** (`src/agents/contracts.py`)
  - Strict Pydantic models (e.g., `EvaluationOutput`, `FeedbackOutput`, `QuestionOutput`) ensuring zero malformed context propagation between isolated graph nodes.
- **Validation Gates & CircuitBreakers:** (`src/services/validation.py`)
  - Evaluates outputs pre-orchestration (e.g., bounds checking `overall_score`, minimum string lengths for `evaluation_reasoning`, constraint validation on `question_type`). Failing nodes are handled gracefully by standard graph reroutes instead of throwing terminal exceptions.

### 1.5 Intelligent Agents (Focus: EvaluatorAgent)
- **EvaluatorAgent:** (`src/agents/evaluator.py`)
  - Employs **Chain-of-Thought (CoT)** via the 14B model to rigorously score answers.
  - Passes its output to a **Reflection step** via the 7B model for score clamping, misconception detection, and logic checks.
  - Built-in support for **Self-Consistency** (running multiple samples and picking median evaluations if divergence is low).
- **Function/Tool Implementations:**
  - `rubric_tool.py`: An asynchronous `@tool` caching rubrics from a flattened JSON file (avoiding I/O per call).
  - `code_validator.py`: A Langchain `@tool` performing tiered validation (Regex detection -> Syntax markers -> AST parsing for valid Python syntax), shielding the Evaluator from executing malformed code.

---

## 2. How Everything Connects (Data Flow & Architecture)

The system is structured via **LangGraph** orchestrating deterministic rules combined with autonomous agent branches. The core cycle runs through a **SupervisorAgent (OODA Loop)**.

1. **Start Phase:** The API Layer generates an Initial Interview Plan, seeds an initial `difficulty_curve` per topic, and kicks off pre-warming as a FastAPI background task which populates the `InterviewCacheStore`.
2. **Observe & Orient (Supervisor):** When the User submits an answer, the `SupervisorAgent` observes the LangGraph state. It calculates recent performance strings via the `TrendAnalyzer`. The **EMA (Exponential Moving Average, α=0.3)** smooths noisy LLM scores and calculates a stable trait metric to ascertain difficulty shift (increase, decrease, stable).
3. **Act (Fan-Out Execution):**
   - The user's answer triggers parallel branching in LangGraph to avoid sequential latency penalties.
   - **EvaluatorAgent (14B)**: Scores the current response, interacting with the database dynamically via `rubric_tool`, and stores data to `current_evaluation`.
   - **QuestionSelectorAgent (7B/14B - Next Up)**: Looks at `question_mode`, uses `retrieve` to pull the next RAG-validated context from `InterviewCacheStore`, or branches to 14B to generate a dynamic `follow_up`/`clarification`.
   - **FeedbackAgent (7B - Next Up)**: Synthesizes the Evaluator's scores into non-sycophantic, dynamic feedback templates (protecting internal scores from user view).
4. **Conclusion (Fan-In):** The Parallel streams synchronize back at the Supervisor. The Supervisor updates standard graph properties: `question_count`, `difficulty_history`, `ema_trajectory`. Finally, the API bundles the Feedback and new Question.
5. **Context Checkpointing:** Once every three turns, a `ConversationManager` LangGraph node truncates the oldest context, firing an LLM to generate a dense semantic summary (combating context starvation).

---

## 3. Concepts Covered

The development of this system touches upon deep layers of multiple technical domains:

### 3.1 AI Engineering
- **LangChain & LangGraph Orchestration:** Constructing cyclical, non-DAG agent loops (StateGraph), enforcing type contracts via `.with_structured_output()`, and manipulating sub-chains dynamically via `.with_fallbacks()` and declarative `.with_retry()` decorators.
- **Agentic RAG (CRAG):** Moving away from naive retrieval. Integrating **Corrective RAG** to evaluate raw embeddings, reject unhelpful context iteratively via LLM critics (QueryRefiner), and dynamically pivoting topics during a retrieval impasse.
- **Reflection & Self-Consistency:** Avoiding straightforward LLM hallucination limits through runtime critics (the 7B model reviewing the 14B model's math and scoring logic) and variance reduction via multi-sample voting schemas.
- **Dynamic Context Summarization:** Overcoming context window saturation using dynamic boundary-truncation and rolling semantic LLM summarization.

### 3.2 Machine Learning
- **EMA (Exponential Moving Average) Smoothing:** Applied via the `TrendAnalyzer`. Instead of naive rule-based difficulty jumps which lead to erratic user experiences, setting $\alpha=0.3$ stabilizes noise in generated LLM evaluations.
- **Semantic Operations:** Deep application of Vector Embeddings (BGE-base 768d dimensions). Processing Cosine Distances natively (`>0.85` stringency filters) to deduplicate generated question semantics and validate retrieval quality.

### 3.3 Software Engineering
- **Dual-Model Fallback Pattern (Graceful Degradation):** Engineering a system around dynamic API failures using `configurable_alternatives` where expensive tasks conditionally drop to faster/less capable hardware under duress without crashing the state machine.
- **Concurrency & Transaction Integrity:** 
  - Using dual-pool `asyncio.Lock` caching schemas with non-blocking atomic selects (`select_and_mark()`) to eliminate TOCTOU (Time-of-check to time-of-use) race conditions when concurrent parallel nodes read from shared question pools.
  - Employing SQLite natively in **WAL mode** with busy timeouts to handle fast concurrent episodic logging by parallel agent nodes.
- **Circuit Breakers & Validation Gates:** Implementing enterprise-level domain boundaries. Agent nodes do not crash the interview. Failure at validation triggers a circuit switch (maximum 1 retry), emitting predictable fallback logic. 
- **Type-Driven Design:** Leveraging highly strict `TypedDict` and `Pydantic` schemas throughout the graph, abstracting away stringly-typed dictionary errors into explicit compilation-time constraints.

---

## 4. What is Next (Sessions 8+)

With all foundational RAG semantics, memory systems, data pipelines, and the Evaluator completed, the next immediate phase (Session 8) entails building the **FeedbackAgent** (`src/agents/feedback.py`) and the **QuestionSelectorAgent** (`src/agents/question_selector.py`) and coupling them via the central Supervisor graph.
