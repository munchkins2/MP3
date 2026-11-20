"""Shared feature engineering (used by training and by the web app)."""
RAW_COLUMNS = ["Submission.Actions", "Quiz.Actions", "Forum.Actions", "Create.Actions",
               "Read.Actions", "Update.Actions", "Delete.Actions", "Days.Logged.In"]


def add_features(X):
    X = X.copy()
    total = X.drop(columns="Days.Logged.In").sum(axis=1)
    X["Total"] = total
    X["Read_share"] = X["Read.Actions"] / (total + 1)
    X["Read_per_day"] = X["Read.Actions"] / (X["Days.Logged.In"] + 1)
    X["Quiz_per_day"] = X["Quiz.Actions"] / (X["Days.Logged.In"] + 1)
    return X
