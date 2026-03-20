"""
AI Interview System — Streamlit Frontend
Connects to FastAPI backend at BASE_URL.
"""

import json
from pathlib import Path

import httpx
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_URL        = "http://localhost:8000/api/v1/interview"
REQUEST_TIMEOUT = 120.0

st.set_page_config(
    page_title="AI Interview System",
    page_icon="🤖",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────
# CSS INJECTION  (loaded once from styles.css)
# ─────────────────────────────────────────────────────────────────

def _inject_css() -> None:
    css_path = Path(__file__).parent / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()


# ─────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────

def init_session() -> None:
    defaults = {
        "stage":             "setup",
        "session_id":        None,
        "user_id":           None,
        "current_question":  None,
        "feedback_history":  [],
        "progress":          None,
        "final_report":      None,
        "time_budget":       30,
        "target_questions":  0,
        "error":             None,
        "available_topics":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────

def api_get_topics() -> list[dict]:
    with httpx.Client(timeout=10.0) as c:
        r = c.get(f"{BASE_URL}/topics")
        r.raise_for_status()
        return r.json()["topics"]


def api_start(difficulty: str, time_budget: int, focus_topics: list[str]) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        r = c.post(f"{BASE_URL}/start", json={
            "difficulty":         difficulty,
            "time_budget_minutes": time_budget,
            "focus_topics":       focus_topics,
        })
        r.raise_for_status()
        return r.json()


def api_submit(session_id: str, response: str) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        r = c.post(f"{BASE_URL}/submit_response", json={
            "session_id": session_id,
            "response":   response,
        })
        r.raise_for_status()
        return r.json()


def api_end(session_id: str) -> dict:
    with httpx.Client(timeout=REQUEST_TIMEOUT) as c:
        r = c.delete(f"{BASE_URL}/end", params={"session_id": session_id})
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────
# CODING QUESTION DETECTION
# ─────────────────────────────────────────────────────────────────

_CODING_TOPICS = {"coding", "programming_for_ml", "software_engineering"}
_CODING_KEYWORDS = [
    "implement", "write a function", "write code", "code a ",
    "design a class", "design an algorithm", "program a", "function that",
    "write the code", "code the", "leetcode", "big-o", "time complexity",
]

def is_coding_question(question: dict | None) -> bool:
    if not question:
        return False
    if any(t in question.get("topic", "").lower() for t in _CODING_TOPICS):
        return True
    return any(kw in question.get("text", "").lower() for kw in _CODING_KEYWORDS)


# ─────────────────────────────────────────────────────────────────
# SCORE COLOUR HELPER
# ─────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 7.0: return "#a6e3a1"
    if score >= 5.0: return "#f9e2af"
    return "#f38ba8"


# ─────────────────────────────────────────────────────────────────
# SETUP SCREEN
# ─────────────────────────────────────────────────────────────────

def render_setup() -> None:
    st.markdown("""
    <div class="ais-hero">
        <h1>🤖 AI Interview System</h1>
        <p>Adaptive technical interviews powered by multi-agent AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Load topics once
    if st.session_state.available_topics is None:
        with st.spinner("Loading topics..."):
            try:
                st.session_state.available_topics = api_get_topics()
            except Exception:
                st.session_state.available_topics = []
                st.warning("Could not load topics from server. Proceeding with general interview.")

    topics        = st.session_state.available_topics
    label_to_val  = {t["label"]: t["value"] for t in topics}
    topic_labels  = [t["label"] for t in topics]

    col1, col2 = st.columns(2)
    with col1:
        difficulty = st.selectbox(
            "Difficulty",
            options=["easy", "medium", "hard"],
            index=1,
            help="Starting difficulty — adapts based on your performance",
        )
    with col2:
        time_budget = st.slider(
            "Time Budget (minutes)",
            min_value=10, max_value=60, value=30, step=5,
        )

    selected_labels = st.multiselect(
        "Focus Topics (optional)",
        options=topic_labels,
        default=[],
        help="Select topics to focus on. Leave blank for a general AI/ML interview.",
    )
    focus_topics = [label_to_val[l] for l in selected_labels]

    topic_str = "General AI/ML" if not focus_topics else ", ".join(selected_labels)
    st.markdown(f"""
    **Interview summary:**  
    Difficulty `{difficulty}` · Duration `up to {time_budget} min` · Topics `{topic_str}`
    """)

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        st.session_state.error = None

    if st.button("Start Interview →", type="primary", use_container_width=True):
        with st.spinner("Initializing your interview..."):
            try:
                result = api_start(difficulty, time_budget, focus_topics)
                st.session_state.session_id        = result["session_id"]
                st.session_state.user_id           = result["user_id"]
                st.session_state.current_question  = result["question"]
                st.session_state.time_budget       = result["time_budget_minutes"]
                st.session_state.target_questions  = result["target_questions"]
                st.session_state.stage             = "active"
                st.rerun()
            except httpx.HTTPStatusError as e:
                st.session_state.error = f"Server error {e.response.status_code}"
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

_CODE_LANGUAGES = ["Python", "Java", "C++", "JavaScript", "Go", "SQL", "Rust", "TypeScript"]


def render_test_results(tr: dict) -> None:
    if tr.get("skipped"):
        st.caption(f"⚙️ Test execution skipped — {tr.get('skip_reason', '')}")
        return

    passed, total = tr["passed"], tr["total"]
    all_pass  = passed == total
    badge_cls = "ais-test-badge ais-test-badge-pass" if all_pass else "ais-test-badge ais-test-badge-fail"
    badge_txt = f"✓ {passed}/{total} passed" if all_pass else f"✗ {passed}/{total} passed"

    st.markdown(
        f'<div class="ais-test-header">'
        f'<span>🧪 Test Cases</span>'
        f'<span class="{badge_cls}">{badge_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for r in tr["results"]:
        if r["error"]:
            row_cls = "ais-test-row ais-test-row-error"
            detail  = f"⚠ {r['error']}"
        elif r["passed"]:
            row_cls = "ais-test-row ais-test-row-pass"
            detail  = f"expected {r['expected']}"
        else:
            row_cls = "ais-test-row ais-test-row-fail"
            detail  = f"expected {r['expected']} &nbsp;·&nbsp; got {r['actual']}"

        st.markdown(
            f'<div class="{row_cls}">'
            f'{"✅" if r["passed"] else "❌"} '
            f'<b>Test {r["index"]}</b> &nbsp; {detail}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_active() -> None:
    progress        = st.session_state.progress
    questions_done  = progress["questions_completed"]   if progress else 0
    time_remaining  = progress["time_remaining_minutes"] if progress else st.session_state.time_budget
    time_budget     = st.session_state.time_budget

    # ── Top metrics ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions Answered", questions_done)
    with col2:
        st.metric("Time Remaining", f"{time_remaining:.1f} min")

    pct = min(1.0, 1.0 - (time_remaining / time_budget)) if time_budget else 0.0
    st.progress(pct)
    st.divider()

    question = st.session_state.current_question

    # Persist answer mode per question; auto-detect coding questions
    mode_key = f"answer_mode_{questions_done}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "code" if is_coding_question(question) else "text"

    # ── Question card ──────────────────────────────────────────────
    if question:
        topic_label = question["topic"].replace("_", " ").title()
        st.markdown(f'<div class="ais-badge">📌 {topic_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ais-question">💬 {question["text"]}</div>', unsafe_allow_html=True)
        st.caption(f"⏱ Estimated time: {question.get('estimated_time_minutes', 5):.0f} min")

    # ── Last feedback + test results ───────────────────────────────
    if st.session_state.feedback_history:
        last = st.session_state.feedback_history[-1]
        st.markdown(f'<div class="ais-feedback">💡 {last["feedback"]}</div>', unsafe_allow_html=True)
        if last.get("test_results"):
            render_test_results(last["test_results"])

    st.divider()

    # ── Answer section ─────────────────────────────────────────────
    answer_mode = st.session_state[mode_key]

    toolbar_left, toolbar_right = st.columns([3, 2])
    with toolbar_left:
        lang = st.selectbox(
            "Language",
            options=_CODE_LANGUAGES,
            key=f"lang_{questions_done}",
            label_visibility="collapsed",
        ) if answer_mode == "code" else "Python"

    with toolbar_right:
        new_mode = st.radio(
            "Answer type",
            options=["code", "text"],
            index=0 if answer_mode == "code" else 1,
            horizontal=True,
            key=f"mode_radio_{questions_done}",
            label_visibility="collapsed",
        )
        if new_mode != answer_mode:
            st.session_state[mode_key] = new_mode
            st.rerun()

    if answer_mode == "code":
        st.markdown(
            f'<div class="ais-code-toolbar">'
            f'<span class="ais-win-dot ais-win-dot-r"></span>'
            f'<span class="ais-win-dot ais-win-dot-y"></span>'
            f'<span class="ais-win-dot ais-win-dot-g"></span>'
            f'<span class="ais-code-lang">{lang}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        response = st.text_area(
            "Your Code",
            height=300,
            placeholder=f"# Write your {lang} solution here\n",
            key=f"code_response_{questions_done}",
            label_visibility="collapsed",
        )
    else:
        response = st.text_area(
            "Your Answer",
            height=180,
            placeholder="Type your answer here. Take your time — think out loud.",
            key=f"text_response_{questions_done}",
        )

    # ── Submit / End ───────────────────────────────────────────────
    col_submit, col_end = st.columns([4, 1])
    with col_submit:
        if st.button("Submit Answer →", type="primary", use_container_width=True):
            if not response.strip():
                st.warning("Please enter your answer before submitting.")
            else:
                _handle_submit(response, question)
    with col_end:
        if st.button("End Interview", use_container_width=True):
            _handle_end()

    # ── History ────────────────────────────────────────────────────
    # Replaced st.expander (whose animated content overlaps surrounding elements
    # in Streamlit's rendering pipeline) with a plain session-state toggle.
    # All history items render in the normal document flow — no overlap possible.
    if st.session_state.feedback_history:
        n_turns = len(st.session_state.feedback_history)

        if "show_history" not in st.session_state:
            st.session_state.show_history = False

        label = f"▼ Hide History ({n_turns} turns)" if st.session_state.show_history \
                else f"▶ Show History ({n_turns} turns)"
        if st.button(f"📜 {label}", use_container_width=True, key="toggle_history"):
            st.session_state.show_history = not st.session_state.show_history

        if st.session_state.show_history:
            st.markdown("<div class='ais-history-section'>", unsafe_allow_html=True)
            for i, item in enumerate(reversed(st.session_state.feedback_history)):
                n       = n_turns - i
                preview = item["response"][:200] + ("..." if len(item["response"]) > 200 else "")
                with st.container(border=True):
                    st.markdown(f"**Q{n}:** {item['question']}")
                    st.caption(f"Your answer:  {preview}")
                    st.caption(f"Feedback:  {item['feedback']}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)


def _handle_submit(response: str, question: dict) -> None:
    with st.spinner("Evaluating your response..."):
        try:
            result = api_submit(st.session_state.session_id, response)
            st.session_state.feedback_history.append({
                "question":     question["text"],
                "response":     response,
                "feedback":     result["feedback"],
                "test_results": result.get("test_results"),
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


def _handle_end() -> None:
    with st.spinner("Generating your final report..."):
        try:
            report = api_end(st.session_state.session_id)
            st.session_state.final_report = report
            st.session_state.stage        = "complete"
            st.rerun()
        except httpx.HTTPStatusError as e:
            st.error(f"Failed to end interview: {e.response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────
# FINAL REPORT SCREEN
# ─────────────────────────────────────────────────────────────────

def render_complete() -> None:
    report = st.session_state.final_report

    st.title("📊 Interview Complete")
    st.divider()

    # ── Top-line scores ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    for col, val, label in [
        (col1, report["overall_score"],  "Overall Score"),
        (col2, report["adjusted_score"], "Difficulty-Adjusted"),
        (col3, report["questions_asked"], "Questions Answered"),
    ]:
        color = _score_color(val) if label != "Questions Answered" else "#cba6f7"
        with col:
            st.markdown(
                f'<div class="ais-score-card">'
                f'<div class="ais-score-number" style="color:{color}">{val:.0f}</div>'
                f'<div class="ais-score-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption(f"⏱ Time taken: {report['time_taken_minutes']:.1f} minutes")
    st.divider()

    # ── Topic breakdown ───────────────────────────────────────────
    if report.get("topic_scores"):
        st.subheader("📌 Topic Breakdown")
        for topic, score in report["topic_scores"].items():
            pct   = min(100, score * 10)
            color = _score_color(score)
            st.markdown(
                f'<div class="ais-topic-row">'
                f'<div class="ais-topic-label">{topic.replace("_", " ").title()}</div>'
                f'<div class="ais-topic-bar-wrap">'
                f'<div class="ais-topic-bar-fill" style="width:{pct}%"></div>'
                f'</div>'
                f'<div class="ais-topic-score" style="color:{color}">{score:.1f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    # ── Strengths / Needs Practice / To Improve ───────────────────
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        st.subheader("✅ Strengths")
        st.caption("Score ≥ 7.0")
        if report.get("strengths"):
            for s in report["strengths"]:
                st.success(s.replace("_", " ").title())
        else:
            st.caption("Keep going — strengths come with practice.")

    with col_mid:
        st.subheader("🔶 Needs Practice")
        st.caption("Score 6.0 – 6.9")
        if report.get("needs_practice"):
            for t in report["needs_practice"]:
                st.info(t.replace("_", " ").title())
        else:
            st.caption("—")

    with col_right:
        st.subheader("📈 To Improve")
        st.caption("Score < 6.0")
        if report.get("areas_for_improvement"):
            for a in report["areas_for_improvement"]:
                st.warning(a.replace("_", " ").title())
        else:
            st.caption("—")

    # ── Performance notes ─────────────────────────────────────────
    if report.get("performance_notes"):
        st.divider()
        for note in report["performance_notes"]:
            st.info(f"📝 {note}")

    # ── Per-question breakdown ────────────────────────────────────
    # Uses session-state toggles instead of st.expander to avoid the
    # overlap/layering bug in Streamlit's animated expander widget.
    evals = [e for e in report.get("detailed_evaluations", []) if not e.get("is_fallback")]
    if evals:
        st.divider()
        st.subheader("🔍 Question Breakdown")
        st.caption("Click a question to see what you covered and what you missed.")

        for i, ev in enumerate(evals, 1):
            score     = ev["overall_score"]
            topic     = ev.get("topic", "").replace("_", " ").title()
            color     = _score_color(score)
            open_key  = f"qb_open_{i}"

            if open_key not in st.session_state:
                st.session_state[open_key] = False

            arrow = "▼" if st.session_state[open_key] else "▶"
            btn_label = f"{arrow}  Q{i} · {topic} · {score:.1f} / 10"
            if st.button(btn_label, key=f"qb_btn_{i}", use_container_width=True):
                st.session_state[open_key] = not st.session_state[open_key]

            if st.session_state[open_key]:
                with st.container(border=True):
                    sub_cols   = st.columns(4)
                    sub_fields = [
                        ("technical_accuracy", "Technical"),
                        ("completeness",       "Completeness"),
                        ("depth",              "Depth"),
                        ("clarity",            "Clarity"),
                    ]
                    for col, (field, label) in zip(sub_cols, sub_fields):
                        val = ev.get(field, {})
                        s   = val.get("score", 0) if isinstance(val, dict) else val
                        col.metric(label, f"{s:.0f} / 10")

                    covered = ev.get("key_points_covered", [])
                    if covered:
                        st.markdown("**✅ What you covered:**")
                        for pt in covered:
                            st.markdown(f"- {pt.replace('**', '')}")

                    missed = ev.get("key_points_missed", [])
                    if missed:
                        st.markdown("**📌 What you missed:**")
                        for pt in missed:
                            st.markdown(f"- {pt.replace('**', '')}")

                    for m in ev.get("misconceptions", []):
                        st.warning(m)

    st.divider()
    if st.button("Start New Interview", type="primary", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────

{
    "setup":    render_setup,
    "active":   render_active,
    "complete": render_complete,
}.get(st.session_state.stage, render_setup)()