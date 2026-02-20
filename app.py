import os
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
# Page config
# ----------------------------
st.set_page_config(page_title="AI Adaptive Question Generator", layout="wide")
st.title("AI-Powered Adaptive Question Generator")
st.caption("Primary Cybersecurity & Digital Literacy • Qatar • Adaptive Difficulty • MCQ Practice")
st.divider()

# ----------------------------
# Sidebar (Configuration)
# ----------------------------
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

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
# Load curriculum (cache per grade/topic)
# ----------------------------
@st.cache_data
def load_curriculum(selected_grade: int, selected_topic: str):
    mapper = CurriculumMapper()
    sample_outcomes = [
        {"description": f"Students will understand {selected_topic.lower()}.", "grade_level": selected_grade, "topic": selected_topic},
        {"description": "Identify safe and unsafe online behaviors.", "grade_level": selected_grade, "topic": selected_topic},
        {"description": "Recognize phishing attempts in emails.", "grade_level": selected_grade, "topic": selected_topic},
    ]
    return mapper.map_outcomes(sample_outcomes)

outcomes = load_curriculum(grade, topic)

# ----------------------------
# Initialize components
# ----------------------------
if not (api_key and api_key.startswith("sk-")):
    st.error("Please enter a valid OpenAI API key in the sidebar")
    st.stop()

qg = QuestionGenerator(api_key=api_key)
estimator = DifficultyEstimator()

# ----------------------------
# Session state init
# ----------------------------
if "sequencer" not in st.session_state:
    st.session_state.sequencer = AdaptiveSequencer(initial_ability=0.0)
    st.session_state.sequencer.initialize_session("demo_student", outcomes * 10)  # fake pool

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

# prevent repetition
if "question_history" not in st.session_state:
    st.session_state.question_history = []

# show recent attempts
if "attempt_log" not in st.session_state:
    st.session_state.attempt_log = []

# store last selected config to detect changes
if "last_config" not in st.session_state:
    st.session_state.last_config = {"grade": grade, "topic": topic}

# generation metrics for dissertation evidence (success/fail/retries/time/reasons)
if "gen_metrics" not in st.session_state:
    st.session_state.gen_metrics = []

sequencer = st.session_state.sequencer

# ----------------------------
# Helpers
# ----------------------------
def generate_new_question():
    # Pick outcome for variety
    outcome = random.choice(outcomes)

    # Avoid repeats (last 5 stems)
    avoid = st.session_state.question_history[-5:]

    # Capture generation reliability metrics
    t0 = time.perf_counter()
    q, meta = qg.generate_question(outcome, avoid_stems=avoid, return_meta=True)
    latency = time.perf_counter() - t0

    st.session_state.gen_metrics.append({
        "ts": time.time(),
        "grade": grade,
        "topic": topic,
        "bloom": outcome.get("bloom_level"),
        "success": bool(q),
        "attempts_used": meta.get("attempts_used"),
        "latency_sec": round(latency, 3),
        "fail_reasons": meta.get("fail_reasons", []),
    })

    if not q:
        st.session_state.current_question = None
        st.session_state.current_options = None
        st.session_state.feedback = {"correct": False, "message": "⚠️ Failed to generate a question. Try again."}
        st.session_state.answered = False
        st.session_state.selected_option = None
        return

    # Save stem to history
    st.session_state.question_history.append(q["question_stem"])

    # Estimate difficulty
    diff = estimator.estimate_difficulty(q)
    q["estimated_difficulty"] = diff["composite_difficulty"]

    # Shuffle options fairly
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

    # reset generation metrics too
    st.session_state.gen_metrics = []

    st.session_state.sequencer = AdaptiveSequencer(initial_ability=0.0)
    st.session_state.sequencer.initialize_session("demo_student", outcomes * 10)


# ----------------------------
# Auto reset if grade/topic changed (optional but clean)
# ----------------------------
if (st.session_state.last_config["grade"] != grade) or (st.session_state.last_config["topic"] != topic):
    reset_quiz()
    st.session_state.last_config = {"grade": grade, "topic": topic}
    st.info("Configuration changed → quiz reset for consistency ✅")

# ----------------------------
# Dashboard / Status (Top card)
# ----------------------------
with st.container(border=True):
    st.subheader("📊 Student Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Student Ability (θ)", f"{sequencer.current_ability:.2f}")
    c2.metric("Answered", str(st.session_state.answered_count))
    c3.metric("Correct", str(st.session_state.correct_count))

    accuracy = (st.session_state.correct_count / max(1, st.session_state.answered_count)) * 100
    c4.metric("Accuracy", f"{accuracy:.0f}%")

    if st.session_state.attempt_log:
        st.caption("Recent attempts: " + " ".join(st.session_state.attempt_log[-10:]))

# ----------------------------
# Controls row (clean)
# ----------------------------
controls = st.columns([2, 1])
with controls[0]:
    if st.button("🎲 Generate / Next Question", type="primary"):
        generate_new_question()
        st.rerun()

with controls[1]:
    if st.button("🔄 Reset Quiz", type="secondary"):
        reset_quiz()
        st.rerun()

# ----------------------------
# Question card
# ----------------------------
q = st.session_state.current_question
options = st.session_state.current_options

with st.container(border=True):
    st.subheader("🧠 Question")

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
            "✅ Submit Answer",
            type="primary",
            disabled=(st.session_state.selected_option is None or st.session_state.answered),
        )

        if submit:
            chosen = st.session_state.selected_option
            is_correct = (chosen == q["correct_answer"])

            # Update counts
            st.session_state.answered_count += 1
            if is_correct:
                st.session_state.correct_count += 1

            # Log attempts (✅/❌)
            st.session_state.attempt_log.append("✅" if is_correct else "❌")
            st.session_state.attempt_log = st.session_state.attempt_log[-10:]

            # Update adaptive model
            sequencer.update_ability(q, is_correct)

            # Feedback
            if is_correct:
                st.session_state.feedback = {"correct": True, "message": "✅ Correct! Great job 🎉"}
            else:
                st.session_state.feedback = {"correct": False, "message": f"❌ Incorrect. Correct answer: **{q['correct_answer']}**"}

            st.session_state.answered = True
            st.rerun()
    else:
        st.info("Click **Generate / Next Question** to start ✅")

# ----------------------------
# Feedback card
# ----------------------------
with st.container(border=True):
    st.subheader("📌 Feedback")

    if st.session_state.feedback:
        if st.session_state.feedback["correct"]:
            st.success(st.session_state.feedback["message"])
        else:
            st.error(st.session_state.feedback["message"])

    if q and st.session_state.answered:
        st.info(f"💡 Explanation: {q['explanation']}")

        m1, m2 = st.columns(2)
        m1.write(f"**Difficulty:** {q.get('estimated_difficulty', 0):.2f} / 5.0")
        m2.write(f"**Bloom Level:** {q.get('bloom_level', 'N/A')}")






# ----------------------------
# Generator Statistics Panel (Dissertation Evidence)
# ----------------------------
with st.container(border=True):
    st.subheader("📈 Generator Statistics (Dissertation Evidence)")

    gm = st.session_state.get("gen_metrics", [])
    if not gm:
        st.info("No generator metrics yet. Generate a few questions first.")
    else:
        df = pd.DataFrame(gm)

        total = len(df)
        success_n = int(df["success"].sum())
        success_rate = (success_n / total) * 100 if total > 0 else 0.0

        avg_attempts = df["attempts_used"].mean() if total > 0 else 0.0
        avg_latency = df["latency_sec"].mean() if total > 0 else 0.0
        p95_latency = df["latency_sec"].quantile(0.95) if total > 0 else 0.0

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Generations", f"{total}")
        g2.metric("Success rate", f"{success_rate:.1f}%")
        g3.metric("Avg attempts used", f"{avg_attempts:.2f}")
        g4.metric("Avg latency", f"{avg_latency:.2f}s")

        st.caption(f"95th percentile latency: {p95_latency:.2f}s")

        # Failure reasons (only failed generations)
        failure_counter = Counter()
        for event in gm:
            if not event.get("success", False):
                for attempt_entry in event.get("fail_reasons", []):
                    for reason in attempt_entry.get("reasons", []):
                        failure_counter[reason] += 1

        if failure_counter:
            st.write("**Top failure causes:**")
            st.dataframe(
                pd.DataFrame(failure_counter.most_common(), columns=["reason", "count"]),
                use_container_width=True
            )
        else:
            st.success("No recorded failure reasons (all generations succeeded). ✅")

        # Optional: Success by grade/topic/bloom (helpful if you switch configs during tests)
        by_cfg = (
            df.groupby(["grade", "topic", "bloom"])["success"]
              .agg(total="count", successes="sum")
              .reset_index()
        )
        by_cfg["success_rate_%"] = (by_cfg["successes"] / by_cfg["total"] * 100).round(1)

        with st.expander("Show success rate by grade/topic/Bloom"):
            st.dataframe(by_cfg.sort_values(["grade", "topic", "bloom"]), use_container_width=True)

        with st.expander("Show raw generation log"):
            st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Download generation log (CSV)",
            data=df.to_csv(index=False),
            file_name="generation_log.csv",
            mime="text/csv",
        )

