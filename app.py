import random
import time
from collections import Counter

import pandas as pd
import streamlit as st

from question_generator import QuestionGenerator
from curriculum_mapper import CurriculumMapper
from difficulty_estimator import DifficultyEstimator
from adaptive_engine import AdaptiveSequencer

# ----------------------------
# Page config (minimal)
# ----------------------------
st.set_page_config(page_title="AI Adaptive Question Generator", layout="wide")
st.title("AI Adaptive Question Generator")
st.caption("Primary Cybersecurity & Digital Literacy • Qatar • Adaptive Difficulty • Multiple Choice Practice")
st.divider()

# ----------------------------
# Sidebar (Configuration)
# ----------------------------
st.sidebar.header("Configuration")

grade = st.sidebar.selectbox("Grade Level", [3, 4, 5, 6])
topic = st.sidebar.selectbox(
    "Topic Area",
    [
        "Online Safety",
        "Password Security",
        "Phishing Recognition",
        "Digital Footprint",
        "Digital Citizenship",
        "Cybersecurity Basics",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Tip: changing grade/topic during a quiz may reset progress.")

# ----------------------------
# Secrets API key
# ----------------------------
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Server configuration error: OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

if not (api_key and api_key.startswith("sk-")):
    st.error("Invalid API key in Streamlit Secrets.")
    st.stop()

qg = QuestionGenerator(api_key=api_key)
estimator = DifficultyEstimator()

# ----------------------------
# Load curriculum (cache per grade/topic)
# ----------------------------
@st.cache_data
def load_curriculum(selected_grade: int, selected_topic: str):
    mapper = CurriculumMapper()
    sample_outcomes = [
        {
            "description": f"Students will understand {selected_topic.lower()}.",
            "grade_level": selected_grade,
            "topic": selected_topic,
        },
        {
            "description": "Identify safe and unsafe online behaviors.",
            "grade_level": selected_grade,
            "topic": selected_topic,
        },
        {
            "description": "Recognize phishing attempts in emails.",
            "grade_level": selected_grade,
            "topic": selected_topic,
        },
    ]
    return mapper.map_outcomes(sample_outcomes)


outcomes = load_curriculum(grade, topic)

# ----------------------------
# Session state init
# ----------------------------
if "sequencer" not in st.session_state:
    st.session_state.sequencer = AdaptiveSequencer(initial_ability=0.0)
    st.session_state.sequencer.initialize_session("demo_student", outcomes * 10)

if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "current_options" not in st.session_state:
    st.session_state.current_options = None
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "answered" not in st.session_state:
    st.session_state.answered = False
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "answered_count" not in st.session_state:
    st.session_state.answered_count = 0
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "attempt_log" not in st.session_state:
    st.session_state.attempt_log = []
if "last_config" not in st.session_state:
    st.session_state.last_config = {"grade": grade, "topic": topic}
if "gen_metrics" not in st.session_state:
    st.session_state.gen_metrics = []

sequencer = st.session_state.sequencer

# ----------------------------
# Helpers
# ----------------------------
def generate_new_question():
    outcome = random.choice(outcomes)

    # Avoid repeats (last 5 stems)
    avoid = st.session_state.question_history[-5:]

    # Metrics for dissertation evidence
    t0 = time.perf_counter()
    q, meta = qg.generate_question(outcome, avoid_stems=avoid, return_meta=True)
    latency = time.perf_counter() - t0

    st.session_state.gen_metrics.append(
        {
            "ts": time.time(),
            "grade": grade,
            "topic": topic,
            "bloom": outcome.get("bloom_level"),
            "success": bool(q),
            "attempts_used": meta.get("attempts_used"),
            "latency_sec": round(latency, 3),
            "fail_reasons": meta.get("fail_reasons", []),
        }
    )

    if not q:
        st.session_state.current_question = None
        st.session_state.current_options = None
        st.session_state.feedback = {"correct": False, "message": "Failed to generate a question. Try again."}
        st.session_state.answered = False
        st.session_state.selected_option = None
        return

    st.session_state.question_history.append(q["question_stem"])

    diff = estimator.estimate_difficulty(q)
    q["estimated_difficulty"] = diff["composite_difficulty"]

    options = [q["correct_answer"]] + q["distractors"]
    random.shuffle(options)

    st.session_state.current_question = q
    st.session_state.current_options = options
    st.session_state.feedback = None
    st.session_state.answered = False
    st.session_state.selected_option = None


def reset_quiz():
    st.session_state.current_question = None
    st.session_state.current_options = None
    st.session_state.selected_option = None
    st.session_state.feedback = None
    st.session_state.answered = False
    st.session_state.correct_count = 0
    st.session_state.answered_count = 0
    st.session_state.question_history = []
    st.session_state.attempt_log = []
    st.session_state.gen_metrics = []

    st.session_state.sequencer = AdaptiveSequencer(initial_ability=0.0)
    st.session_state.sequencer.initialize_session("demo_student", outcomes * 10)


# Auto reset if grade/topic changed
if (st.session_state.last_config["grade"] != grade) or (st.session_state.last_config["topic"] != topic):
    reset_quiz()
    st.session_state.last_config = {"grade": grade, "topic": topic}
    st.info("Configuration changed — quiz reset for consistency.")

# ----------------------------
# UI: Tabs
# ----------------------------
tab_quiz, tab_teacher = st.tabs(["Quiz", "Teacher view"])

# ============================
# QUIZ TAB
# ============================
with tab_quiz:
    # Top status row (compact)
    with st.container(border=True):
        st.subheader("Session status")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Student ability (θ)", f"{sequencer.current_ability:.2f}")
        c2.metric("Answered", str(st.session_state.answered_count))
        c3.metric("Correct", str(st.session_state.correct_count))

        accuracy = (st.session_state.correct_count / max(1, st.session_state.answered_count)) * 100
        c4.metric("Accuracy", f"{accuracy:.0f}%")

        # ✅/❌ kept here (allowed)
        if st.session_state.attempt_log:
            st.caption("Recent attempts: " + " ".join(st.session_state.attempt_log[-10:]))

    # Controls row
    controls = st.columns([2, 1])
    with controls[0]:
        if st.button("Next question", type="primary"):
            generate_new_question()
            st.rerun()

    with controls[1]:
        if st.button("Reset quiz", type="secondary"):
            reset_quiz()
            st.rerun()

    # Main two-column layout: Question (left) + Feedback (right)
    left, right = st.columns([2, 1], gap="large")

    q = st.session_state.current_question
    options = st.session_state.current_options

    with left:
        with st.container(border=True):
            st.subheader("Question")

            if q and options:
                st.write(q["question_stem"])

                st.session_state.selected_option = st.radio(
                    "Choose one answer:",
                    options,
                    index=None,
                    key="answer_radio",
                    disabled=st.session_state.answered,
                )

                submit = st.button(
                    "Submit",
                    type="primary",
                    disabled=(st.session_state.selected_option is None or st.session_state.answered),
                )

                if submit:
                    chosen = st.session_state.selected_option
                    is_correct = (chosen == q["correct_answer"])

                    st.session_state.answered_count += 1
                    if is_correct:
                        st.session_state.correct_count += 1

                    # ✅/❌ only
                    st.session_state.attempt_log.append("✅" if is_correct else "❌")
                    st.session_state.attempt_log = st.session_state.attempt_log[-10:]

                    sequencer.update_ability(q, is_correct)

                    if is_correct:
                        st.session_state.feedback = {"correct": True, "message": "✅ Correct"}
                    else:
                        st.session_state.feedback = {
                            "correct": False,
                            "message": f"❌ Wrong\n\nCorrect answer: {q['correct_answer']}",
                        }

                    st.session_state.answered = True
                    st.rerun()
            else:
                st.info("Click Next question to start.")

    with right:
        with st.container(border=True):
            st.subheader("Feedback")

            fb = st.session_state.feedback
            if fb:
                if fb["correct"]:
                    st.success(fb["message"])
                else:
                    st.error(fb["message"])
            else:
                st.caption("Submit an answer to see feedback.")

            if q and st.session_state.answered:
                with st.expander("Details"):
                    st.write("Explanation:")
                    st.write(q["explanation"])
                    st.write(f"Difficulty: {q.get('estimated_difficulty', 0):.2f} / 5.0")
                    st.write(f"Bloom level: {q.get('bloom_level', 'N/A')}")

# ============================
# TEACHER VIEW TAB (selectbox panels)
# ============================
with tab_teacher:
    with st.container(border=True):
        st.subheader("Quiz configuration")
        st.write(f"Grade: {grade}")
        st.write(f"Topic: {topic}")

    # Panel picker (prevents long scroll)
    panel = st.selectbox("Select a panel", ["Stats", "Failures", "Raw log", "Download"], index=0)

    gm = st.session_state.get("gen_metrics", [])

    if not gm:
        st.info("No generator metrics yet. Generate a few questions first.")
    else:
        df = pd.DataFrame(gm)

        # Common aggregates used by multiple panels
        total = len(df)
        success_n = int(df["success"].sum())
        success_rate = (success_n / total) * 100 if total > 0 else 0.0
        avg_attempts = df["attempts_used"].mean() if total > 0 else 0.0
        avg_latency = df["latency_sec"].mean() if total > 0 else 0.0
        p95_latency = df["latency_sec"].quantile(0.95) if total > 0 else 0.0

        if panel == "Stats":
            with st.container(border=True):
                st.subheader("Generator statistics")

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Generations", f"{total}")
                g2.metric("Success rate", f"{success_rate:.1f}%")
                g3.metric("Avg attempts used", f"{avg_attempts:.2f}")
                g4.metric("Avg latency", f"{avg_latency:.2f}s")

                st.caption(f"95th percentile latency: {p95_latency:.2f}s")

                by_cfg = (
                    df.groupby(["grade", "topic", "bloom"])["success"]
                    .agg(total="count", successes="sum")
                    .reset_index()
                )
                by_cfg["success_rate_%"] = (by_cfg["successes"] / by_cfg["total"] * 100).round(1)

                st.dataframe(by_cfg.sort_values(["grade", "topic", "bloom"]), use_container_width=True)

        elif panel == "Failures":
            with st.container(border=True):
                st.subheader("Failure analysis")

                failure_counter = Counter()
                failed_rows = df[df["success"] == False]  # noqa: E712

                for event in gm:
                    if not event.get("success", False):
                        for attempt_entry in event.get("fail_reasons", []):
                            for reason in attempt_entry.get("reasons", []):
                                failure_counter[reason] += 1

                if failed_rows.empty:
                    st.success("No failed generations recorded.")
                else:
                    st.write(f"Failed generations: {len(failed_rows)} / {total}")

                if failure_counter:
                    st.dataframe(
                        pd.DataFrame(failure_counter.most_common(), columns=["reason", "count"]),
                        use_container_width=True,
                    )
                else:
                    st.info("No structured failure reasons captured for the failed events.")

        elif panel == "Raw log":
            with st.container(border=True):
                st.subheader("Raw generation log")
                st.dataframe(df, use_container_width=True)

        elif panel == "Download":
            with st.container(border=True):
                st.subheader("Export")
                st.download_button(
                    "Download generation log (CSV)",
                    data=df.to_csv(index=False),
                    file_name="generation_log.csv",
                    mime="text/csv",
                )
