from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

from features import RAW_COLUMNS, add_features

BASE_DIR = Path(__file__).resolve().parent   # works wherever the app is launched from
MODEL_PATH = BASE_DIR / "student_rating_model.joblib"
THRESHOLD = 0.5

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    """Load the saved model; retrain from the CSV if the file is missing or
    was saved with an incompatible scikit-learn version."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from train_model import train
        return train()


def predict(raw_df, bundle):
    feats = add_features(raw_df[RAW_COLUMNS])[bundle["feature_columns"]]
    p_high = bundle["model"].predict_proba(feats)[:, 1]
    out = raw_df.copy()
    out["Prob_Pass"] = p_high.round(3)
    out["Prediction"] = ["Likely to pass (High)" if p >= THRESHOLD else "At risk (Low)" for p in p_high]
    return out


bundle = load_bundle()

st.title("🎓 Student Success Predictor")
st.caption("Predicts whether a student is likely to pass (High rating) or is at risk (Low rating) "
           "from midterm LMS activity. It is an early-warning aid, not a final decision.")

tab_single, tab_batch, tab_about = st.tabs(["Single student", "Whole class (CSV)", "About the model"])

# ---------------------------------------------------------------- single student
with tab_single:
    st.subheader("Enter midterm LMS activity")
    c1, c2 = st.columns(2)
    vals = {}
    with c1:
        vals["Submission.Actions"] = st.number_input("Submission actions", 0, 1000, 5, help="Assignment submissions")
        vals["Quiz.Actions"] = st.number_input("Quiz actions", 0, 5000, 0)
        vals["Forum.Actions"] = st.number_input("Forum actions", 0, 1000, 0)
        vals["Create.Actions"] = st.number_input("Create actions", 0, 1000, 5)
    with c2:
        vals["Read.Actions"] = st.number_input("Read actions", 0, 5000, 50)
        vals["Update.Actions"] = st.number_input("Update actions", 0, 1000, 2)
        vals["Delete.Actions"] = st.number_input("Delete actions", 0, 100, 0)
        vals["Days.Logged.In"] = st.number_input("Days logged in", 0, 200, 14)

    if st.button("Predict", type="primary", use_container_width=True):
        res = predict(pd.DataFrame([vals]), bundle).iloc[0]
        p = float(res["Prob_Pass"])
        if p >= THRESHOLD:
            st.success(f"**{res['Prediction']}**")
        else:
            st.error(f"**{res['Prediction']}** - consider early academic support")
        st.metric("Probability of High rating (pass)", f"{p:.1%}")
        st.progress(p)
        st.caption("Confidence is highest near 0% or 100% and lowest near 50%.")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Upload a CSV for a whole course")
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in RAW_COLUMNS) +
             ". Any extra columns (e.g. a student ID) are kept.")
    st.download_button("Download CSV template",
                       pd.DataFrame(columns=["Student_ID"] + RAW_COLUMNS).to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in RAW_COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        elif data[RAW_COLUMNS].isna().any().any() or (data[RAW_COLUMNS] < 0).any().any():
            st.error("Activity columns must be non-negative numbers with no blanks.")
        else:
            res = predict(data, bundle).sort_values("Prob_Pass")
            n_low = int((res["Prob_Pass"] < THRESHOLD).sum())
            a, b = st.columns(2)
            a.metric("Students", len(res))
            b.metric("Flagged at risk", f"{n_low} ({n_low / len(res):.0%})")
            st.dataframe(res, use_container_width=True)
            st.download_button("Download results", res.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("About")
    st.markdown(f"""
- **Model:** Extra Trees classifier (300 trees), chosen over Random Forest and Histogram Gradient Boosting by 5-fold cross-validation (88.1% CV accuracy).
- **Test accuracy:** **{bundle['test_accuracy']:.2%}** on a held-out 20% test set (stratified split, seed {bundle['random_seed']}).
- **Inputs:** 8 midterm LMS activity counts, plus 4 derived features (total actions, read share, reads per login day, quiz actions per login day).
- **Meaning of the output:** *High* = activity resembles students who finished with a high rating (likely pass); *Low* = resembles low-rated students (at risk).
- **Limits:** the model finds an association between activity and rating, not a cause. Training data is augmented and balanced, so real-world accuracy may differ. Use predictions to start a conversation with the student, not to make final decisions.
""")
