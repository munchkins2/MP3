"""Train the final model and save it to student_rating_model.joblib.
Run:  python train_model.py
Same pipeline as the notebook: stratified 80/20 split, seed 42, Extra Trees
(the model selected by 5-fold CV). Test set is used once, for the final score."""
import joblib, pandas as pd, sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from features import RAW_COLUMNS, add_features

SEED = 42
CSV, OUT = "augmented_balanced_dataset_m.csv", "student_rating_model.joblib"


def train():
    df = pd.read_csv(CSV)
    y = (df["rating"] == "High").astype(int)          # High=1 (likely pass), Low=0 (at risk)
    X = add_features(df[RAW_COLUMNS])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1,
                                 random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    bundle = {"model": model, "feature_columns": list(X.columns), "raw_columns": RAW_COLUMNS,
              "label_map": {0: "Low", 1: "High"}, "random_seed": SEED,
              "test_accuracy": float(acc), "sklearn_version": sklearn.__version__}
    joblib.dump(bundle, OUT, compress=3)
    return bundle


if __name__ == "__main__":
    b = train()
    print(f"Saved {OUT} | test accuracy = {b['test_accuracy']:.4f} | sklearn {b['sklearn_version']}")
