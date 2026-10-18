"""
Shared chronological split: the most recent season is the test set and every
earlier season is training data. A random split would let the model train on
games played after the games it is tested on, which is not how prediction
works in practice.
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
