"""
Two non-ML baselines the trained model has to beat:

1. Naive baseline: whichever team has the better season win% so far wins.
2. Home-win-rate baseline: home team always wins (checks how much of the
   signal is just home-court advantage).

Run after features.py. Prints accuracy for both on the same train/test
split train.py will use, so comparisons in evaluate.py line up.
"""
import os

import pandas as pd
from sklearn.metrics import accuracy_score

from split import train_test_split_by_season  # shared split logic

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matchup_features.csv")


def naive_baseline(df: pd.DataFrame) -> pd.Series:
    """Predict home win if home season win% > away season win%."""
    return (df["HOME_SEASON_WIN_PCT"] > df["AWAY_SEASON_WIN_PCT"]).astype(int)


def home_baseline(df: pd.DataFrame) -> pd.Series:
    """Always predict the home team wins."""
    return pd.Series(1, index=df.index)


def main():
    df = pd.read_csv(DATA_PATH)
    _, test = train_test_split_by_season(df)

    y_true = test["TARGET_HOME_WIN"]

    naive_pred = naive_baseline(test)
    home_pred = home_baseline(test)

    print(f"Naive (season win%) baseline accuracy: {accuracy_score(y_true, naive_pred):.4f}")
    print(f"Home-win-rate baseline accuracy:        {accuracy_score(y_true, home_pred):.4f}")
    print(f"Actual home win rate in test set:       {y_true.mean():.4f}")


if __name__ == "__main__":
    main()
