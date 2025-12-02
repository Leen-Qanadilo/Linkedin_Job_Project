import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# If you install sklearn-genetic in your env:
from sklearn_genetic import GAFeatureSelectionCV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data")
    parser.add_argument("--selected_features")
    parser.add_argument("--baseline_metrics")
    parser.add_argument("--ga_metrics")
    args = parser.parse_args()

    # ---------------- Load data ----------------
    df = pd.read_parquet(args.train_data)

    label_col = "job_role"
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Encode labels to integers (good practice for classifiers)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ---------------- Baseline selection ----------------
    vt = VarianceThreshold()          # simple filter
    X_vt = vt.fit_transform(X)

    baseline_features = X.columns[vt.get_support()].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X_vt, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    clf_baseline = RandomForestClassifier(n_estimators=80, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred = clf_baseline.predict(X_val)
    baseline_acc = accuracy_score(y_val, y_pred)

    baseline_info = {
        "baseline_accuracy": float(baseline_acc),
        "baseline_num_features": len(baseline_features),
    }
    with open(args.baseline_metrics, "w") as f:
        json.dump(baseline_info, f, indent=2)

    # ---------------- Genetic Algorithm selection ----------------
    start = time.time()

    estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    ga = GAFeatureSelectionCV(
        estimator=estimator,
        cv=3,
        scoring="accuracy",
        population_size=20,
        generations=10,
        n_jobs=-1,
        verbose=False,
    )

    ga.fit(X, y_enc)
    ga_time = time.time() - start

    ga_mask = ga.best_features_
    ga_features = X.columns[ga_mask].tolist()

    ga_info = {
        "ga_accuracy": float(ga.best_score_),
        "ga_num_features": len(ga_features),
        "ga_runtime_seconds": float(ga_time),
    }
    with open(args.ga_metrics, "w") as f:
        json.dump(ga_info, f, indent=2)

    # ---------------- Save final selected features ----------------
    with open(args.selected_features, "w") as f:
        json.dump(ga_features, f, indent=2)


if __name__ == "__main__":
    main()
