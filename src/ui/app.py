"""
AI Interview System — Streamlit Frontend
Connects to FastAPI backend at BASE_URL.
"""

import streamlit as st
import httpx

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8000/api/v1/interview"
REQUEST_TIMEOUT = 120.0

st.set_page_config(
    page_title="AI Interview System",
    page_icon="🤖",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "stage": "setup",
        "session_id": None,
        "user_id": None,
        "current_question": None,
        "feedback_history": [],
        "progress": None,
        "final_report": None,
        "time_budget": 30,
        "target_questions": 0,
        "error": None,
        "available_topics": None,   # cached on first load
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ─────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────

def api_get_topics() -> list[dict]:
    """Fetch available topics from backend. Returns list of {label, value}."""
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(f"{BASE_URL}/topics")
        resp.raise_for_status()
        return resp.json()["topics"]


def api_start(difficulty: str, time_budget: int, focus_topics: list[str]) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/start", json={
            "difficulty": difficulty,
            "time_budget_minutes": time_budget,
            "focus_topics": focus_topics,
        })
        resp.raise_for_status()
        return resp.json()


def api_submit(session_id: str, response: str) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/submit_response", json={
            "session_id": session_id,
            "response": response,
        })
        resp.raise_for_status()
        return resp.json()


def api_end(session_id: str) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.delete(f"{BASE_URL}/end", params={"session_id": session_id})
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .question-box {
        background-color: #1e1e2e;
        border-left: 4px solid #7c6af7;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #cdd6f4;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .feedback-box {
        background-color: #1e2e1e;
        border-left: 4px solid #a6e3a1;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #cdd6f4;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .topic-badge {
        display: inline-block;
        background-color: #313244;
        color: #cba6f7;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .score-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem;
    }
    .score-number {
        font-size: 2rem;
        font-weight: 700;
        color: #cba6f7;
    }
    .score-label {
        font-size: 0.85rem;
        color: #a6adc8;
    }
    .history-item {
        background-color: #181825;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SETUP SCREEN
# ─────────────────────────────────────────────────────────────────

def render_setup():
    st.title("🤖 AI Interview System")
    st.markdown("*Adaptive technical interviews powered by multi-agent AI*")
    st.divider()

    # Load topics once — cache in session state
    if st.session_state.available_topics is None:
        with st.spinner("Loading topics..."):
            try:
                st.session_state.available_topics = api_get_topics()
            except Exception:
                st.session_state.available_topics = []
                st.warning("Could not load topics from server. Proceeding with general interview.")

    topics = st.session_state.available_topics  # list of {label, value}
    label_to_value = {t["label"]: t["value"] for t in topics}
    topic_labels = [t["label"] for t in topics]

    col1, col2 = st.columns(2)

    with col1:
        difficulty = st.selectbox(
            "Difficulty",
            options=["easy", "medium", "hard"],
            index=1,
            help="Starting difficulty — adapts based on your performance"
        )

    with col2:
        time_budget = st.slider(
            "Time Budget (minutes)",
            min_value=10,
            max_value=60,
            value=30,
            step=5,
        )

    # Topic multiselect — populated from backend
    selected_labels = st.multiselect(
        "Focus Topics (optional)",
        options=topic_labels,
        default=[],
        help="Select topics to focus on. Leave blank for a general AI/ML interview."
    )

    # Map selected labels back to raw values
    focus_topics = [label_to_value[label] for label in selected_labels]

    st.markdown(f"""
    **Interview summary:**
    - Difficulty: `{difficulty}`
    - Duration: `up to {time_budget} minutes`
    - Topics: `{"General AI/ML" if not focus_topics else ", ".join(selected_labels)}`
    """)

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        st.session_state.error = None

    if st.button("Start Interview →", type="primary", use_container_width=True):
        with st.spinner("Initializing your interview..."):
            try:
                result = api_start(difficulty, time_budget, focus_topics)
                st.session_state.session_id = result["session_id"]
                st.session_state.user_id = result["user_id"]
                st.session_state.current_question = result["question"]
                st.session_state.time_budget = result["time_budget_minutes"]
                st.session_state.target_questions = result["target_questions"]
                st.session_state.stage = "active"
                st.rerun()
            except httpx.HTTPStatusError as e:
                st.session_state.error = f"Server error: {e.response.status_code}"
                st.rerun()
            except httpx.ConnectError:
                st.session_state.error = "Cannot connect to server. Is the API running?"
                st.rerun()
            except Exception as e:
                st.session_state.error = str(e)
                st.rerun()


# ─────────────────────────────────────────────────────────────────
# ACTIVE INTERVIEW SCREEN
# ─────────────────────────────────────────────────────────────────

def render_active():
    progress = st.session_state.progress
    questions_done = progress["questions_completed"] if progress else 0
    time_remaining = progress["time_remaining_minutes"] if progress else st.session_state.time_budget
    time_budget = st.session_state.time_budget

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Answered", f"{questions_done}")
    with col2:
        st.metric("Time Remaining", f"{time_remaining:.1f} min")

    pct = min(1.0, 1.0 - (time_remaining / time_budget)) if time_budget else 0.0
    st.progress(pct)
    st.divider()

    # Current question
    question = st.session_state.current_question
    if question:
        st.markdown(
            f'<div class="topic-badge">📌 {question["topic"].replace("_", " ").title()}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="question-box">💬 {question["text"]}</div>',
            unsafe_allow_html=True
        )
        st.caption(f"⏱ Estimated time: {question.get('estimated_time_minutes', 5):.0f} min")

    # Last feedback
    if st.session_state.feedback_history:
        last = st.session_state.feedback_history[-1]
        st.markdown(
            f'<div class="feedback-box">💡 {last["feedback"]}</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Response input
    response = st.text_area(
        "Your Answer",
        height=180,
        placeholder="Type your answer here. Take your time — think out loud.",
        key=f"response_{questions_done}",
    )

    col_submit, col_end = st.columns([4, 1])

    with col_submit:
        if st.button(
            "Submit Answer →",
            type="primary",
            use_container_width=True,
        ):
            if not response.strip():
                st.warning("Please enter your answer before submitting.")
            else:
                _handle_submit(response, question)

    with col_end:
        if st.button("End Interview", use_container_width=True):
            _handle_end()

    # Collapsible history
    if st.session_state.feedback_history:
        with st.expander(f"📜 Interview History ({len(st.session_state.feedback_history)} turns)"):
            for i, item in enumerate(reversed(st.session_state.feedback_history)):
                st.markdown('<div class="history-item">', unsafe_allow_html=True)
                st.markdown(f"**Q{len(st.session_state.feedback_history) - i}:** {item['question']}")
                st.markdown(f"*Your answer:* {item['response'][:200]}{'...' if len(item['response']) > 200 else ''}")
                st.markdown(f"*Feedback:* {item['feedback']}")
                st.markdown('</div>', unsafe_allow_html=True)


def api_submit_stream(session_id: str, response: str) -> dict:
    """
    SSE streaming submit — collects streamed tokens for feedback display,
    returns final structured result when turn_complete event arrives.
    """
    feedback_tokens = []
    final_result = None

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        with client.stream("POST", f"{BASE_URL}/submit_response/stream", json={
            "session_id": session_id,
            "response": response,
        }) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                import json
                payload = json.loads(line[6:])  # strip "data: "

                if payload["type"] == "token":
                    feedback_tokens.append(payload["content"])

                elif payload["type"] == "turn_complete":
                    final_result = payload["data"]
                    # Reassemble streamed feedback tokens into final result
                    if feedback_tokens:
                        final_result["feedback"] = "".join(feedback_tokens)

                elif payload["type"] == "error":
                    raise Exception(payload["detail"])

    return final_result


def _handle_submit(response: str, question: dict):
    with st.spinner("Evaluating your response..."):
        try:
            result = api_submit(st.session_state.session_id, response)

            st.session_state.feedback_history.append({
                "question": question["text"],
                "response": response,
                "feedback": result["feedback"],
            })

            st.session_state.progress = result["progress"]

            if result["continue_interview"] and result["next_question"]:
                st.session_state.current_question = result["next_question"]
                st.rerun()
            else:
                _handle_end()

        except httpx.HTTPStatusError as e:
            st.error(f"Submission failed: {e.response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")


def _handle_end():
    with st.spinner("Generating your final report..."):
        try:
            report = api_end(st.session_state.session_id)
            st.session_state.final_report = report
            st.session_state.stage = "complete"
            st.rerun()
        except httpx.HTTPStatusError as e:
            st.error(f"Failed to end interview: {e.response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# FINAL REPORT SCREEN
# ─────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 7.0:
        return "#a6e3a1"   # green
    elif score >= 5.0:
        return "#f9e2af"   # yellow
    return "#f38ba8"       # red


def render_complete():
    report = st.session_state.final_report

    st.title("📊 Interview Complete")
    st.divider()

    # ── Top-line scores ──────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number" style="color:{_score_color(report['overall_score'])}">
                {report['overall_score']:.1f}
            </div>
            <div class="score-label">Overall Score</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number" style="color:{_score_color(report['adjusted_score'])}">
                {report['adjusted_score']:.1f}
            </div>
            <div class="score-label">Difficulty-Adjusted</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number">{report['questions_asked']}</div>
            <div class="score-label">Questions Answered</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"⏱ Time taken: {report['time_taken_minutes']:.1f} minutes")
    st.divider()

    # ── Topic breakdown ──────────────────────────────────────────
    if report.get("topic_scores"):
        st.subheader("📌 Topic Breakdown")
        for topic, score in report["topic_scores"].items():
            label = topic.replace("_", " ").title()
            col_label, col_bar, col_score = st.columns([2, 5, 1])
            with col_label:
                st.markdown(f"**{label}**")
            with col_bar:
                st.progress(min(1.0, score / 10))
            with col_score:
                st.markdown(
                    f"<span style='color:{_score_color(score)}'>{score:.1f}</span>",
                    unsafe_allow_html=True
                )
        st.divider()

    # ── Strengths & improvements ─────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("✅ Strengths")
        if report.get("strengths"):
            for s in report["strengths"]:
                st.success(s.replace("_", " ").title())
        else:
            st.info("Keep practicing to build strengths.")

    with col_right:
        st.subheader("📈 To Improve")
        if report.get("areas_for_improvement"):
            for a in report["areas_for_improvement"]:
                st.warning(a.replace("_", " ").title())

    # ── Performance notes ────────────────────────────────────────
    if report.get("performance_notes"):
        st.divider()
        for note in report["performance_notes"]:
            st.info(f"📝 {note}")

    # ── Per-question breakdown ───────────────────────────────────
    evals = [e for e in report.get("detailed_evaluations", [])
             if not e.get("is_fallback", False)]

    if evals:
        st.divider()
        st.subheader("🔍 Question Breakdown")
        st.caption("Expand each question to see what you covered and what you missed.")

        for i, ev in enumerate(evals, 1):
            score = ev["overall_score"]
            topic = ev.get("topic", "").replace("_", " ").title()
            color = _score_color(score)

            with st.expander(f"Q{i} · {topic} · Score: {score:.1f} / 10"):

                # Sub-scores
                sub_cols = st.columns(4)
                sub_fields = [
                    ("technical_accuracy", "Technical"),
                    ("completeness", "Completeness"),
                    ("depth", "Depth"),
                    ("clarity", "Clarity"),
                ]
                for col, (field, label) in zip(sub_cols, sub_fields):
                    val = ev.get(field, {})
                    s = val.get("score", 0) if isinstance(val, dict) else val
                    col.metric(label, f"{s:.0f} / 10")

                # Key points covered
                covered = ev.get("key_points_covered", [])
                if covered:
                    st.markdown("**✅ What you covered:**")
                    for pt in covered:
                        # strip markdown bold markers from rubric text
                        clean = pt.replace("**", "")
                        st.markdown(f"- {clean}")

                # Key points missed
                missed = ev.get("key_points_missed", [])
                if missed:
                    st.markdown("**📌 What you missed:**")
                    for pt in missed:
                        clean = pt.replace("**", "")
                        st.markdown(f"- {clean}")

                # Misconceptions
                misconceptions = ev.get("misconceptions", [])
                if misconceptions:
                    st.markdown("**⚠️ Misconceptions to address:**")
                    for m in misconceptions:
                        st.warning(m)

    st.divider()
    if st.button("Start New Interview", type="primary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────

stage = st.session_state.stage

if stage == "setup":
    render_setup()
elif stage == "active":
    render_active()
elif stage == "complete":
    render_complete()