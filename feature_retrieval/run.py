import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    args = parser.parse_args()

    # Load features
    df = pd.read_parquet(args.feature_set)

    # IMPORTANT: Change this to your real label name if different
    label_col = "label"

    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_parquet(args.train_data, index=False)
    test_df.to_parquet(args.test_data, index=False)

if __name__ == "__main__":
    main()
