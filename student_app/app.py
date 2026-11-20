from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from features import RAW_COLUMNS, add_features

BASE_DIR = Path(__file__).resolve().parent   # works wherever the app is launched from
MODEL_PATH = BASE_DIR / "student_rating_model.joblib"
THRESHOLD = 0.5

st.set_page_config(page_title="Student Passing Calculator", layout="centered")

# ------------------------------------------------------------------ styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        background-image:
            linear-gradient(#e5e7eb 1px, transparent 1px),
            linear-gradient(90deg, #e5e7eb 1px, transparent 1px) !important;
        background-size: 28px 28px !important;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 720px;
        background: #ffffff !important;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 2rem 2rem 2.5rem 2rem;
        margin-top: 2rem;
    }
    .stApp h1, .stApp h3, .stApp label, .stApp label p,
    .stApp [data-testid="stCaptionContainer"] { color: #111827 !important; }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #000000 !important; opacity: 1 !important;
    }
    .stApp [data-testid="stCaptionContainer"] { margin-top: -0.4rem; }

    /* number boxes: soft teal tint */
    .stApp [data-testid="stNumberInput"] [data-baseweb="input"],
    .stApp [data-testid="stNumberInput"] [data-baseweb="base-input"],
    .stApp [data-testid="stNumberInput"] input {
        background-color: #e3f1ef !important; color: #0b3b36 !important;
    }
    .stApp [data-testid="stNumberInput"] [data-baseweb="input"] {
        border: 1px solid #8fc3bc !important; border-radius: 6px;
    }
    .stApp [data-testid="stNumberInputStepUp"],
    .stApp [data-testid="stNumberInputStepDown"] {
        background-color: #cfe7e3 !important; color: #0b3b36 !important;
    }

    /* predict button: deep teal */
    .stApp button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {
        background-color: #0f766e !important; border: 1px solid #0f766e !important;
        color: #ffffff !important;
    }
    .stApp button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
        background-color: #115e59 !important; border-color: #115e59 !important;
    }
    .stApp button[kind="primary"] p, .stApp [data-testid="stBaseButton-primary"] p {
        color: #ffffff !important;
    }
    .result-box {
        display: flex; align-items: center; gap: 1rem;
        padding: 1rem 1.25rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-box svg { flex: none; }
    .result-title { font-size: 1.15rem; font-weight: 700; margin: 0; }
    .result-sub { font-size: 0.95rem; margin: 0.15rem 0 0 0; }
    .pass { background: #e6f4ea; border: 1px solid #34a853; color: #137333; }
    .fail { background: #fce8e6; border: 1px solid #d93025; color: #a50e0e; }
    </style>
    """,
    unsafe_allow_html=True,
)

THUMB_PATH = ("M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06"
              "L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22"
              "l3.02-7.05c.09-.23.14-.47.14-.73v-2z")


def thumb_icon(up: bool, color: str) -> str:
    rotate = "" if up else ' transform="rotate(180 12 12)"'
    return (f'<svg width="44" height="44" viewBox="0 0 24 24" fill="{color}">'
            f'<path d="{THUMB_PATH}"{rotate}/></svg>')


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    """Load the saved model; retrain from the CSV if the file is missing or
    was saved with an incompatible scikit-learn version."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from train_model import train
        return train()


def prob_pass(values: dict, bundle) -> float:
    row = pd.DataFrame([values])
    feats = add_features(row[RAW_COLUMNS])[bundle["feature_columns"]]
    return float(bundle["model"].predict_proba(feats)[0, 1])


bundle = load_bundle()

st.title("Student Passing Calculator")

# ------------------------------------------------------------------ inputs
FIELDS = [
    ("Submission.Actions", "Submission", "Assignments or work handed in", 5, 1000),
    ("Quiz.Actions", "Quiz", "Quiz attempts and answers submitted", 0, 5000),
    ("Forum.Actions", "Forum", "Posts and replies in discussion forums", 0, 1000),
    ("Create.Actions", "Create", "New content the student created", 5, 1000),
    ("Read.Actions", "Read", "Course pages and materials viewed", 50, 5000),
    ("Update.Actions", "Update", "Edits made to existing work", 2, 1000),
    ("Delete.Actions", "Delete", "Content the student removed", 0, 100),
    ("Days.Logged.In", "Days logged in", "Number of days the student logged into the LMS", 14, 200),
]

vals = {}
left, right = st.columns(2)
for i, (key, label, desc, default, maxv) in enumerate(FIELDS):
    with (left if i < 4 else right):
        vals[key] = st.number_input(label, min_value=0, max_value=maxv, value=default, key=key)
        st.caption(desc)

# ------------------------------------------------------------------ result
if st.button("Predict", type="primary", use_container_width=True):
    p = prob_pass(vals, bundle)
    if p >= THRESHOLD:
        st.markdown(
            f"""<div class="result-box pass">{thumb_icon(True, "#137333")}
            <div><p class="result-title">Likely to pass (High)</p>
            <p class="result-sub">Probability of passing: {p:.1%}</p></div></div>""",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"""<div class="result-box fail">{thumb_icon(False, "#a50e0e")}
            <div><p class="result-title">At risk (Low)</p>
            <p class="result-sub">Probability of passing: {p:.1%}. Consider early academic support.</p></div></div>""",
            unsafe_allow_html=True)
