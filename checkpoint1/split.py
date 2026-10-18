"""
Shared split logic: hold out the most recent season as the test set,
train on everything before it. This is more honest for sports prediction
than a random split, since a random split would leak information across
a season (rolling averages from December leaking into November, etc.
already handled by shift(), but a chronological holdout is still the
right way to simulate 'predicting future games').
"""
from typing import Optional

import pandas as pd


def train_test_split_by_season(df: pd.DataFrame, test_season: Optional[str] = None):
    seasons = sorted(df["SEASON"].unique())
    if test_season is None:
        test_season = seasons[-1]
    train = df[df["SEASON"] != test_season].reset_index(drop=True)
    test = df[df["SEASON"] == test_season].reset_index(drop=True)
    return train, test
