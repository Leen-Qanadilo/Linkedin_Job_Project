import argparse
import json
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

# ------------------------ CONFIG ------------------------

# Change this if your label column has a different name
TARGET_COL = "job_role"

# Try to import GA library; if missing, we will fall back
try:
    from sklearn_genetic import GAFeatureSelectionCV

    HAS_GA = True
except ImportError:
    HAS_GA = False


# ------------------------ MAIN ------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    # accept both --train_data and --train-data just in case
    parser.add_argument(
        "--train_data",
        "--train-data",
        dest="train_data",
        type=str,
        required=True,
        help="Path to train parquet file",
    )
    parser.add_argument(
        "--baseline_metrics",
        "--baseline-metrics",
        dest="baseline_metrics",
        type=str,
        required=True,
        help="Output JSON for baseline metrics",
    )
    parser.add_argument(
        "--ga_metrics",
        "--ga-metrics",
        dest="ga_metrics",
        type=str,
        required=True,
        help="Output JSON for GA metrics",
    )
    parser.add_argument(
        "--selected_features",
        "--selected-features",
        dest="selected_features",
        type=str,
        required=True,
        help="Output JSON for selected feature names",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading training data from:", args.train_data)
    df = pd.read_parquet(args.train_data)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' not found in dataframe columns: "
            f"{list(df.columns)}"
        )

    # ------------------------ Split X / y ------------------------
    y = df[TARGET_COL].astype(str)
    X = df.drop(columns=[TARGET_COL])

    print("Shape before feature selection:", X.shape)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # ------------------------ Variance Threshold ------------------------
    # Keep all non-constant features
    vt = VarianceThreshold(threshold=0.0)
    X_vt_array = vt.fit_transform(X)
    support_mask = vt.get_support()
    X_vt = X.loc[:, support_mask]

    baseline_features = list(X_vt.columns)
    print("Shape after VT:", X_vt.shape)

    # ------------------------ Baseline + GA with fallbacks ------------------------
    try:
        # -------- Baseline model --------
        X_train, X_val, y_train, y_val = train_test_split(
            X_vt,
            y_enc,
            test_size=0.2,
            stratify=y_enc,
            random_state=42,
        )

        clf_baseline = RandomForestClassifier(
            n_estimators=80, random_state=42
        )
        clf_baseline.fit(X_train, y_train)
        y_pred = clf_baseline.predict(X_val)
        baseline_acc = accuracy_score(y_val, y_pred)

        baseline_info = {
            "baseline_accuracy": float(baseline_acc),
            "baseline_num_features": len(baseline_features),
        }

        with open(args.baseline_metrics, "w") as f:
            json.dump(baseline_info, f, indent=2)

        # -------- GA selection (optional) --------
        if HAS_GA:
            print("Running GA feature selection...")
            start = time.time()

            estimator = RandomForestClassifier(
                n_estimators=100, random_state=42
            )

            ga = GAFeatureSelectionCV(
                estimator=estimator,
                cv=3,
                scoring="accuracy",
                population_size=20,
                generations=10,
                n_jobs=-1,
                verbose=False,
            )

            ga.fit(X_vt, y_enc)
            ga_time = time.time() - start

            ga_mask = ga.support_  # boolean mask
            ga_features = X_vt.columns[ga_mask].tolist()

            ga_info = {
                "ga_accuracy": float(ga.best_score_),
                "ga_num_features": len(ga_features),
                "ga_runtime_seconds": float(ga_time),
            }
        else:
            print(
                "WARNING: sklearn-genetic-opt not available – "
                "using baseline features as GA output."
            )
            ga_features = baseline_features
            ga_info = {
                "ga_accuracy": float(baseline_acc),
                "ga_num_features": len(ga_features),
                "ga_runtime_seconds": 0.0,
                "note": "GA library missing, reused baseline results",
            }

        with open(args.ga_metrics, "w") as f:
            json.dump(ga_info, f, indent=2)

    except Exception as e:
        # If *anything* explodes in baseline or GA, fall back
        print(
            "WARNING: Baseline + GA feature selection failed, "
            "using all VT features."
        )
        print("Reason:", e)

        ga_features = baseline_features

        baseline_info = {
            "note": "fallback – baseline/GA failed, using all VT features",
            "baseline_num_features": len(baseline_features),
        }
        ga_info = {
            "note": "fallback – GA failed, using all VT features",
            "ga_num_features": len(ga_features),
        }

        with open(args.baseline_metrics, "w") as f:
            json.dump(baseline_info, f, indent=2)
        with open(args.ga_metrics, "w") as f:
            json.dump(ga_info, f, indent=2)

    # ------------------------ Save final selected feature list ------------------------
    with open(args.selected_features, "w") as f:
        json.dump(ga_features, f, indent=2)

    print("Feature selection finished. Number of features:", len(ga_features))


if __name__ == "__main__":
    main()

