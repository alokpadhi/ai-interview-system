```mermaid
flowchart TD
    %% ─────────────────────────────────────────────
    %% LAYER 1 — CLIENT
    %% ─────────────────────────────────────────────
    subgraph CLIENT["🖥 LAYER 1 — CLIENT"]
        StreamlitUI["🖥 Streamlit UI"]
        RESTClient["⬛ REST Client"]
        SSEConsumer["📡 SSE Consumer"]
    end

    %% ─────────────────────────────────────────────
    %% LAYER 2 — API LAYER
    %% ─────────────────────────────────────────────
    subgraph API["⚡ LAYER 2 — API LAYER (FastAPI)"]
        subgraph Endpoints["Endpoints"]
            EP["POST /start
            POST /submit_response
            POST /submit_response/stream (SSE)
            DELETE /end
            GET /topics"]
        end

        subgraph Lifespan["Lifespan Manager"]
            LC["LLMs → Model Prewarm
            → VectorStore + VectorRetriever
            → initialize_concept_lookup()
            → get_available_topics()
            → RAG Infrastructure
            → AgentRegistry (DI container)
            → async with AsyncSqliteSaver
            → Compile Graphs
            → Start Periodic Cleanup Task"]
        end

        subgraph SessionStore["SessionStore (in-memory)"]
            SS["SessionMeta:
            user_id: str (server-generated UUID)
            turn_number: int
            first_turn_done: bool
            start_result: dict | None
            ⚠ Single-process only
            Use Redis for multi-worker"]
        end

        subgraph RConfig["RunnableConfig"]
            RC["thread_id = session_id
            metadata: user_id, turn_number
            tags: interview, start|submit
            BackgroundTasks: pre-warming
            AsyncSqliteSaver (dev)
            AsyncPostgresSaver (prod)"]
        end
    end

    %% ─────────────────────────────────────────────
    %% LAYER 3 — GRAPH LAYER
    %% ─────────────────────────────────────────────
    subgraph GRAPHS["🔀 LAYER 3 — GRAPH LAYER"]
        subgraph StartGraph["START GRAPH (plan + first question)"]
            SG["create_plan → first_question → END
            Seeds checkpoint on first /submit_response
            No checkpointer — output lives in memory
            until first interview_graph ainvoke()"]
        end

        subgraph InterviewGraph["INTERVIEW GRAPH (per-turn pipeline)"]
            IGStart(["START"])
            IGEval["Evaluator"]
            IGFeedback["Feedback Agent"]
            IGQS["Question Selector Agent"]
            IGSupervisor["Supervisor (OODA + EMA)"]
            IGSummarize["Maybe Summarize (every 3 turns)"]
            IGEnd(["END"])

            IGStart --> IGEval
            IGEval -->|fan-out| IGFeedback
            IGEval -->|fan-out| IGQS
            IGFeedback -->|fan-in| IGSupervisor
            IGQS -->|fan-in| IGSupervisor
            IGSupervisor --> IGSummarize
            IGSummarize --> IGEnd
        end
    end

    %% ─────────────────────────────────────────────
    %% LAYER 4 — AGENT LAYER
    %% ─────────────────────────────────────────────
    subgraph AGENTS["🤖 LAYER 4 — AGENT LAYER"]
        subgraph EvalAgent["⚖ EvaluatorAgent"]
            EA["• CoT evaluation (14B)
            • Reflection (7B)
            • Self-consistency (configurable N)
            • Sub-score normalization:
              Standard: MAX_SPREAD=6.0
              Low-score: sub_avg&lt;2.0 → MAX_SPREAD=1.0
              Overall alignment: drift ≤ 1.5
            • Drift detection
            • Dynamic rubric (static + target_concepts)
            • Injects topic from current_question
            • is_fallback=True excluded from EMA"]
        end

        subgraph FeedAgent["💬 FeedbackAgent"]
            FA["• Concept context (skip if score≥7.0)
            • Structured gen via FeedbackComposer (7B)
            • 4 templates per score band
            • Semantic repetition check (7B)
            • Validation gate enforcement
            • Anti-sycophancy at low scores
            • Off-topic path: candidate response only
              Q text never seen by LLM"]
        end

        subgraph QSAgent["❓ QuestionSelectorAgent"]
            QS["• Modes: retrieve / follow_up / clarify
            • Clarify fires before follow_up
            • ReAct selection (7B)
            • Atomic select_and_mark() — TOCTOU-safe
            • Dynamic rubric gen (target_concepts)
            • Smarter topic cycling (least-served)
            • 5-tier fallback chain:
              1. cache hit (0ms)
              2. CRAG retrieval (~2s)
              3. alt topic cache — escape hatch (0ms)
              4. LLM-generated 7B context-aware (~1.5s)
              5. hardcoded (last resort, 0ms)"]
        end

        subgraph SupAgent["🎯 SupervisorAgent"]
            SA["• Plan-and-Execute at /start (14B)
            • OODA loop per turn (0 LLM calls)
            • EMA difficulty authority (α=0.3):
              Path 1: improving + avg_ema ≥ 7.5
              Path 2: stable + avg_ema ≥ 8.0
              Decrease: declining + avg_ema &lt; 5.0
              Min 4 scores required
            • Sole owner of question_count
            • topic_sequence is guide not hard cap
            • difficulty_curve indexed by topics_covered
            • MAX_QUESTIONS_HARD_CEILING = 15
            • focus_topics validated post-LLM"]
        end

        subgraph ConvManager["📝 ConversationManager"]
            CM["• Rolling window: last 3 turns verbatim
            • Batch summarize every 3 turns (14B)
            • Sentence-boundary truncation
            • Memory budget: ~1700 tokens bounded
            • LangGraph node (checkpointed)
            • Returns {} no-op most turns
            • get_context_for_agent() is sync"]
        end

        subgraph VGates["🔒 Validation Gates + CircuitBreaker"]
            VG["EvaluatorGate: score range 0-10 | reasoning >50 chars | variance ≤6.0 | coverage-score alignment
            FeedbackGate: length 15-200 words | no forbidden phrases | no sycophancy at score&lt;7.0 | no score leakage | no (?)
            QuestionGate: question present | valid type | estimated_time_minutes ≤ remaining_time
            CircuitBreaker: max 1 retry → fallback on 2nd failure | is_fallback=True | excluded from EMA"]
        end
    end

    %% ─────────────────────────────────────────────
    %% LAYER 5 — TOOLS LAYER
    %% ─────────────────────────────────────────────
    subgraph TOOLS["🛠 LAYER 5 — TOOLS LAYER (@tool)"]
        subgraph RubricTool["📋 rubric_lookup"]
            RT["• Module-level JSON cache (zero I/O)
            • Loaded at import time
            • Returns: key_points, scoring_criteria,
              common_mistakes
            • found: False = graceful miss
            • Flattens ALL criteria key_points"]
        end

        subgraph CodeTool["✅ code_validator"]
            CT["• 3-tier code detection
            • _extract_code() → AST _validate_syntax()
            • Returns: code_detected, is_valid,
              errors, validation_scope, language
            • code_detected: False → is_valid: None"]
        end

        subgraph ConceptTool["🧠 concept_lookup"]
            CLT["• Module-level singleton (_retriever)
            • initialize_concept_lookup() at startup
            • ChromaDB via VectorRetriever
            • Returns: explanation, simple_explanation,
              examples, related_concepts
            • Tool is stateless — agent owns cache"]
        end
    end

    %% ─────────────────────────────────────────────
    %% LAYER 6 — AGENTIC RAG SERVICE
    %% ─────────────────────────────────────────────
    subgraph RAG["🔍 LAYER 6 — AGENTIC RAG SERVICE (LangGraph StateGraph Subgraph)"]
        RAGEntry1["retrieve_with_crag()
        called by QS on cache miss"]
        RAGEntry2["retrieve_batch()
        called by pre-warming"]

        ChromaRetrieval["ChromaDB Retrieval
        Metadata filtering:
        topic / difficulty / time"]

        DocGrader["DocumentGrader
        score ≥ 0.75 → HIGH (no LLM)
        score ≤ 0.45 → LOW (no LLM)
        0.45-0.75 → 7B LLM grading"]

        GradeCheck{"Grade Check"}

        QueryRef["QueryRefiner
        Attempt 0: LLM refine
        Attempt 1: topic pivot
        Attempt 2: simplify
        Anti-repeat: cosine > 0.85 rejected
        MAX_CRAG_ATTEMPTS = 2"]

        PackageResults["package_results"]

        RAGEntry1 --> ChromaRetrieval
        RAGEntry2 --> ChromaRetrieval
        ChromaRetrieval --> DocGrader
        DocGrader --> GradeCheck
        GradeCheck -->|HIGH / MEDIUM| PackageResults
        GradeCheck -->|LOW| QueryRef
        QueryRef -->|max 2 retry attempts| ChromaRetrieval
    end

    %% ─────────────────────────────────────────────
    %% LAYER 7 — CACHE + LLM LAYER
    %% ─────────────────────────────────────────────
    subgraph CACHE_LLM["⚙ LAYER 7 — CACHE + LLM LAYER"]
        subgraph CacheLayer["SESSION-ISOLATED CACHE LAYER (InterviewCacheStore Singleton)"]
            subgraph TopicPool["Topic Question Pool"]
                TP["Key: {topic}:{difficulty}
                Max: 10 entries/session
                TTL: HIGH=30min | MEDIUM=15min | LOW=never
                LRU eviction (pool-isolated)
                Partial reuse via used_ids Set
                Atomic select_and_mark()"]
            end

            subgraph ConceptPool["Concept Pool"]
                CP["Key: {concept_name}
                Max: 30 entries/session
                TTL: 60 minutes
                LRU eviction (pool-isolated)"]
            end

            CacheNotes["• Per-session asyncio.Lock (NOT global)
            • Atomic select_and_mark() — TOCTOU-safe
            • Abandoned sweep: 15min / 90min threshold
            • BackgroundTasks pre-warming via retrieve_batch()
            • Real CRAG grades → accurate TTLs"]
        end

        subgraph LLMLayer["LLM LAYER"]
            subgraph ComplexModel["Complex Model (14B)"]
                CM14["Roles: interview planning,
                CoT evaluation,
                follow-up/clarify generation,
                conversation summarization
                Latency: ~2-3s per call"]
            end

            subgraph FastModel["Fast Model (7B)"]
                FM7["Roles: CRAG grading,
                ReAct selection, reflection,
                feedback generation,
                repetition check, fallback gen
                Latency: ~1-1.5s per call"]
            end

            LLMNotes["Provider: Ollama (default) / OpenAI / Anthropic → llm_factory
            BaseChatModel throughout | .with_structured_output() on all chains
            .with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
            .with_fallbacks([lenient_parser]) on critical chains
            configurable_alternatives() for graceful degradation
            asyncio.wait_for(timeout=15.0) on every graph node"]
        end
    end

    %% ─────────────────────────────────────────────
    %% LAYER 8 — DATA LAYER
    %% ─────────────────────────────────────────────
    subgraph DATA["💾 LAYER 8 — DATA LAYER"]
        subgraph ChromaDB["🟢 ChromaDB"]
            CDB["interview_questions: 700
            ml_concepts: 125
            code_solutions: 50
            BGE-base-en-v1.5 (768d)
            Cosine distance, normalized
            PersistentClient (local disk)"]
        end

        subgraph SQLiteDB["🔵 SQLite"]
            SDB["Tables: interviews, conversations,
            evaluations, session_state, agent_traces
            WAL mode | 5 connections
            busy_timeout: 5000ms
            Checkpoints: AsyncSqliteSaver
            Production: AsyncPostgresSaver"]
        end

        subgraph FileSystem["📁 File System"]
            FS["data/rubrics/ (JSON)
            data/prompts/ (YAML)
            data/questions/ (JSON)
            data/concepts/ (JSON)"]
        end
    end

    %% ─────────────────────────────────────────────
    %% INTER-LAYER CONNECTIONS
    %% ─────────────────────────────────────────────

    %% L1 → L2
    CLIENT -->|HTTP / SSE| API

    %% L2 → L3
    Lifespan -->|start_graph no checkpointer| StartGraph
    Lifespan -->|interview_graph checkpointed| InterviewGraph

    %% L2 → L7 (pre-warming)
    RC -.->|BackgroundTasks pre-warming| TopicPool

    %% L3 → L4 (graph nodes map to agents)
    IGEval -.->|implements| EvalAgent
    IGFeedback -.->|implements| FeedAgent
    IGQS -.->|implements| QSAgent
    IGSupervisor -.->|implements| SupAgent
    IGSummarize -.->|implements| ConvManager

    %% L4 → L5 (tool calls)
    EvalAgent -->|calls @tool| RubricTool
    EvalAgent -->|calls @tool| CodeTool
    FeedAgent -->|calls @tool| ConceptTool

    %% L4 → L6 (RAG)
    QSAgent -->|retrieve_with_crag on cache miss| RAG

    %% L4 → L7 (cache)
    QSAgent -->|select_and_mark| TopicPool
    FeedAgent -->|get_concept / set_concept| ConceptPool

    %% L4 → L7 (LLM)
    EvalAgent -->|CoT + reflection| ComplexModel
    EvalAgent -->|reflection step| FastModel
    FeedAgent -->|structured gen| FastModel
    QSAgent -->|ReAct select + fallback gen| FastModel
    QSAgent -->|follow-up / clarify gen| ComplexModel
    SupAgent -->|plan creation only| ComplexModel
    ConvManager -->|summarization| ComplexModel

    %% L5 → L8
    ConceptTool -->|retrieve_concepts| ChromaDB

    %% L6 → L8
    ChromaRetrieval -->|vector similarity search| ChromaDB

    %% L6 → L7 (cache store)
    PackageResults -.->|store with CRAG grade TTL| TopicPool

    %% L4 → L8
    SupAgent -.->|session + evaluation persistence| SQLiteDB

    %% ─────────────────────────────────────────────
    %% STYLES
    %% ─────────────────────────────────────────────
    classDef layerHeader fill:#4f46e5,color:#fff,font-weight:bold
    classDef agentBox fill:#eef2ff,stroke:#4f46e5,stroke-width:2px,color:#1a1a2e
    classDef toolBox fill:#f0fdf4,stroke:#22c55e,stroke-width:1.5px,color:#1a1a2e
    classDef ragBox fill:#fff7ed,stroke:#f97316,stroke-width:1.5px,color:#1a1a2e
    classDef cacheBox fill:#eff6ff,stroke:#3b82f6,stroke-width:1.5px,color:#1a1a2e
    classDef llmBox fill:#fdf4ff,stroke:#a855f7,stroke-width:1.5px,color:#1a1a2e
    classDef dataBox fill:#f8fafc,stroke:#64748b,stroke-width:1.5px,color:#1a1a2e
    classDef gateBox fill:#fff8f0,stroke:#f97316,stroke-width:2px,color:#1a1a2e
    classDef graphNode fill:#e0e7ff,stroke:#4f46e5,stroke-width:2px,color:#1a1a2e
    classDef decision fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#1a1a2e

    class EA,FA,QS,SA,CM agentBox
    class RT,CT,CLT toolBox
    class ChromaRetrieval,DocGrader,QueryRef,PackageResults ragBox
    class TP,CP,CacheNotes cacheBox
    class CM14,FM7,LLMNotes llmBox
    class CDB,SDB,FS dataBox
    class VG gateBox
    class IGEval,IGFeedback,IGQS,IGSupervisor,IGSummarize graphNode
    class GradeCheck decision
