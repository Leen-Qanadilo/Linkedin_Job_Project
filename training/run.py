import argparse
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# (optional but nice for model registration/metrics)
import mlflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data")
    parser.add_argument("--test_data")
    parser.add_argument("--selected_features")
    parser.add_argument("--metrics")
    args = parser.parse_args()

    # ---------- Load data ----------
    train_df = pd.read_parquet(args.train_data)
    test_df = pd.read_parquet(args.test_data)

    label_col = "job_role"

    # Load selected feature names from GA
    with open(args.selected_features, "r") as f:
        selected_features = json.load(f)

    X_train = train_df[selected_features]
    y_train = train_df[label_col]

    X_test = test_df[selected_features]
    y_test = test_df[label_col]

    # Encode labels to integers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # ---------- Train model ----------
    mlflow.autolog()  # logs params, metrics, model to AML automatically

    clf = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train_enc)

    # ---------- Evaluate ----------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    cm = confusion_matrix(y_test_enc, y_pred).tolist()

    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "num_features": len(selected_features),
    }

    # Save metrics to JSON output
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model to local file (optional – extra to be safe)
    dump(clf, "model.joblib")


if __name__ == "__main__":
    main()
