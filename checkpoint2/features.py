"""
Turns per-team-per-game rows into one row per game with each team's PRE-GAME stats.

Input:  data/raw/team_game_logs.csv with at least
        TEAM_ID, TEAM_ABBREVIATION, GAME_ID, GAME_DATE, WL, PTS, REB, TOV, FG_PCT,
        IS_HOME, SEASON
Output: data/processed/matchup_features.csv

Leakage control: every rolling number is computed on the team's games BEFORE the
current one (shift(1) before rolling/expanding), so a game never sees its own result.
verify_features.py re-computes a sample of these values independently to check this.
Row order of the output is unchanged from earlier versions; GAME_DATE is the new
column so every prediction can be traced to a specific game.
"""
import os

import pandas as pd

import paths

ROLLING_WINDOW = 10  # last-10-game rolling average, per the proposal
MIN_GAMES = 3        # need at least this much history to produce a feature
STAT_COLS = ["PTS", "REB", "TOV", "FG_PCT"]


def regular_season_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season games. NBA GAME_IDs are 10 digits: 001=preseason, 002=regular season,
    004=playoffs, 005=play-in. CSV loading drops the leading zeros, so pad before checking."""
    kind = df["GAME_ID"].astype(str).str.zfill(10).str[:3]
    return df[kind == "002"].copy()


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    for col in STAT_COLS:
        df["ROLL_" + col] = df.groupby("TEAM_ID")[col].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_GAMES).mean()
        )
    df["WIN_FLAG"] = (df["WL"] == "W").astype(int)
    df["ROLL_WIN_PCT"] = df.groupby("TEAM_ID")["WIN_FLAG"].transform(
        lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=MIN_GAMES).mean()
    )
    df["SEASON_WIN_PCT"] = df.groupby(["TEAM_ID", "SEASON"])["WIN_FLAG"].transform(
        lambda s: s.shift(1).expanding(min_periods=MIN_GAMES).mean()
    )
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    # a team's first game in the data has no previous date; assume typical rest
    df["REST_DAYS"] = df["REST_DAYS"].fillna(2)
    return df


def build_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Join each game's two team rows into one matchup row (home vs away).
    Label TARGET_HOME_WIN is 1 if the home team won."""
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    feature_cols = ["ROLL_" + c for c in STAT_COLS] + ["ROLL_WIN_PCT", "SEASON_WIN_PCT", "REST_DAYS"]

    home_feats = home[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ABBREVIATION", "WIN_FLAG"] + feature_cols]
    home_feats = home_feats.rename(
        columns={"TEAM_ABBREVIATION": "HOME_TEAM", "WIN_FLAG": "HOME_WIN",
                 **{c: "HOME_" + c for c in feature_cols}}
    )
    away_feats = away[["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols]
    away_feats = away_feats.rename(
        columns={"TEAM_ABBREVIATION": "AWAY_TEAM", **{c: "AWAY_" + c for c in feature_cols}}
    )

    matchups = home_feats.merge(away_feats, on="GAME_ID", how="inner")
    matchups["TARGET_HOME_WIN"] = matchups["HOME_WIN"]
    matchups = matchups.drop(columns=["HOME_WIN"])

    # drop games where either team lacks enough history (early season / start of data)
    required = [
        c for c in matchups.columns
        if c.startswith("HOME_ROLL") or c.startswith("AWAY_ROLL") or c.endswith("SEASON_WIN_PCT")
    ]
    return matchups.dropna(subset=required)


def main():
    df = pd.read_csv(paths.RAW_PATH, parse_dates=["GAME_DATE"])
    n_all = len(df)
    df = regular_season_only(df)
    print(f"Dropped {n_all - len(df)} of {n_all} team-game rows that are not regular season "
          f"(preseason / playoffs / play-in)")
    df = add_rolling_features(df)
    matchups = build_matchups(df)

    os.makedirs(paths.PROCESSED_DIR, exist_ok=True)
    matchups.to_csv(paths.MATCHUP_PATH, index=False)
    print(f"Saved {len(matchups)} matchup rows to {paths.MATCHUP_PATH}")
    print(f"Seasons: {sorted(matchups['SEASON'].unique())}")


if __name__ == "__main__":
    main()
