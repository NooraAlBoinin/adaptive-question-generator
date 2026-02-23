# ----------------------------
# Sidebar (Configuration)
# ----------------------------
st.sidebar.header("Configuration")

# ✅ Do NOT collect API keys via UI
def get_api_key() -> str | None:
    # 1) Prefer Streamlit secrets (best for Streamlit Cloud)
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # 2) Fallback to environment variable (local / other hosting)
    return os.getenv("OPENAI_API_KEY")


api_key = get_api_key()

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
# Initialize components
# ----------------------------
if not api_key:
    st.error("Server is missing OPENAI_API_KEY. Configure it via Streamlit Secrets or environment variables.")
    st.stop()

# optional: basic sanity check (don’t enforce 'sk-' if you might use non-OpenAI providers later)
if not api_key.strip():
    st.error("OPENAI_API_KEY is empty.")
    st.stop()

qg = QuestionGenerator(api_key=api_key)
estimator = DifficultyEstimator()
