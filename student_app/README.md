# Student Success Predictor (Streamlit)

Predicts whether a student is likely to pass (High rating) or at risk (Low rating) from midterm LMS activity.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (single student form, CSV batch, model info) |
| `features.py` | Feature engineering shared by training and the app |
| `train_model.py` | Retrains and saves `student_rating_model.joblib` |
| `student_rating_model.joblib` | Saved model |
| `augmented_balanced_dataset_m.csv` | Training data |
| `requirements.txt` | Dependencies |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. To rebuild the model: `python train_model.py`.

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and upload every file in this folder (the model is ~9 MB, fine for GitHub).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
4. You get a public link such as `https://your-app.streamlit.app` to submit.

If the saved model was made with a different scikit-learn version, the app automatically retrains from the CSV on first load.
