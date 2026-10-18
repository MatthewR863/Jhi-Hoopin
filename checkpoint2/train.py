"""
Trains classifiers on the matchup features to predict TARGET_HOME_WIN.
All models use the identical chronological split from split.py
(earlier seasons = train, most recent season = test).

Usage:
    python3 checkpoint2_ml_models/train.py --model all      # logreg + rf + xgb
    python3 checkpoint2_ml_models/train.py --model logreg
    python3 checkpoint2_ml_models/train.py --model rf
    python3 checkpoint2_ml_models/train.py --model xgb

Saves each model and its feature list to models/<name>.joblib.
XGBoost is imported only when requested, so logreg and rf still work if
xgboost's OpenMP library (libomp on Mac) is not installed.
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import paths
from split import train_test_split_by_season

FEATURE_COLS = [
    "HOME_ROLL_PTS", "HOME_ROLL_REB", "HOME_ROLL_TOV", "HOME_ROLL_FG_PCT",
    "HOME_ROLL_WIN_PCT", "HOME_SEASON_WIN_PCT", "HOME_REST_DAYS",
    "AWAY_ROLL_PTS", "AWAY_ROLL_REB", "AWAY_ROLL_TOV", "AWAY_ROLL_FG_PCT",
    "AWAY_ROLL_WIN_PCT", "AWAY_SEASON_WIN_PCT", "AWAY_REST_DAYS",
]
MODEL_NAMES = ["logreg", "rf", "xgb"]


def build_model(name: str):
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42)
    if name == "xgb":
        from xgboost import XGBClassifier  # imported here on purpose, see module docstring
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
    raise ValueError("Unknown model: " + name)


def train_one(name: str, train: pd.DataFrame, test: pd.DataFrame):
    model = build_model(name)
    model.fit(train[FEATURE_COLS], train["TARGET_HOME_WIN"])
    train_acc = model.score(train[FEATURE_COLS], train["TARGET_HOME_WIN"])
    test_acc = model.score(test[FEATURE_COLS], test["TARGET_HOME_WIN"])
    print(f"[{name}] train accuracy: {train_acc:.4f}  |  test accuracy: {test_acc:.4f}")
    os.makedirs(paths.MODEL_DIR, exist_ok=True)
    out_path = os.path.join(paths.MODEL_DIR, name + ".joblib")
    joblib.dump({"model": model, "features": FEATURE_COLS}, out_path)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES + ["all"], required=True)
    args = parser.parse_args()

    df = pd.read_csv(paths.MATCHUP_PATH)
    train, test = train_test_split_by_season(df)
    print(f"Train: {len(train)} games ({sorted(train['SEASON'].unique())[0]} to "
          f"{sorted(train['SEASON'].unique())[-1]})  |  Test: {len(test)} games "
          f"({test['SEASON'].iloc[0]})")

    names = MODEL_NAMES if args.model == "all" else [args.model]
    failed = []
    for name in names:
        try:
            train_one(name, train, test)
        except Exception as e:
            if args.model != "all":
                raise
            failed.append(name)
            print(f"[{name}] FAILED: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
    if failed:
        raise SystemExit("These models failed to train: " + ", ".join(failed))


if __name__ == "__main__":
    main()
