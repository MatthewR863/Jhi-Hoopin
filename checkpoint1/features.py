"""
Turns per-team-per-game rows into a matchup-level feature table:
one row per game, with each team's PRE-GAME rolling stats.

Expects data/raw/team_game_logs.csv with (at minimum) these columns,
matching what nba_api's LeagueGameFinder returns:
    TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, MATCHUP, WL,
    PTS, REB, TOV, FG_PCT, IS_HOME, SEASON

If you're using a Kaggle CSV instead, rename/derive columns to match this
shape before running this script.

Output: data/processed/matchup_features.csv
"""
import os

import numpy as np
import pandas as pd

ROLLING_WINDOW = 10  # last-10-game rolling average, per the proposal

STAT_COLS = ["PTS", "REB", "TOV", "FG_PCT"]

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "team_game_logs.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "matchup_features.csv")


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add last-10-game rolling averages, shifted so today's game isn't
    included in its own average (no leakage)."""
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    for col in STAT_COLS:
        df[f"ROLL_{col}"] = (
            df.groupby("TEAM_ID")[col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
        )
    # rolling win % and rest days, same idea
    df["WIN_FLAG"] = (df["WL"] == "W").astype(int)
    df["ROLL_WIN_PCT"] = (
        df.groupby("TEAM_ID")["WIN_FLAG"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    )
    df["SEASON_WIN_PCT"] = (
        df.groupby(["TEAM_ID", "SEASON"])["WIN_FLAG"]
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    # first game of a team's history has no prior date; assume typical rest
    df["REST_DAYS"] = df["REST_DAYS"].fillna(2)
    return df


def build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Each GAME_ID has two rows (one per team). Join them side by side
    into a single matchup row: TEAM_A vs TEAM_B, with TEAM_A's win as the
    label."""
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    feature_cols = [f"ROLL_{c}" for c in STAT_COLS] + ["ROLL_WIN_PCT", "SEASON_WIN_PCT", "REST_DAYS"]

    home_feats = home[["GAME_ID", "SEASON", "TEAM_ABBREVIATION", "WIN_FLAG"] + feature_cols]
    home_feats = home_feats.rename(
        columns={"TEAM_ABBREVIATION": "HOME_TEAM", "WIN_FLAG": "HOME_WIN", **{c: f"HOME_{c}" for c in feature_cols}}
    )
    away_feats = away[["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols]
    away_feats = away_feats.rename(
        columns={"TEAM_ABBREVIATION": "AWAY_TEAM", **{c: f"AWAY_{c}" for c in feature_cols}}
    )

    matchups = home_feats.merge(away_feats, on="GAME_ID", how="inner")

    # label: 1 if home team won, 0 if away team won
    matchups["TARGET_HOME_WIN"] = matchups["HOME_WIN"]
    matchups = matchups.drop(columns=["HOME_WIN"])

    # drop rows without enough history yet (early season / early dataset)
    required = [
        c for c in matchups.columns
        if c.startswith("HOME_ROLL") or c.startswith("AWAY_ROLL")
        or c.endswith("SEASON_WIN_PCT")
    ]
    matchups = matchups.dropna(subset=required)

    return matchups


def main():
    df = pd.read_csv(RAW_PATH, parse_dates=["GAME_DATE"])
    df = add_rolling_features(df)
    matchups = build_matchups(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    matchups.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(matchups)} matchup rows to {OUT_PATH}")
    print(matchups.head())


if __name__ == "__main__":
    main()
