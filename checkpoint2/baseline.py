"""
Non-learned baselines. Every one of these gets the same test games as the ML models.

  always_home_baseline        Always predict the home team. A hard 0/1 pick, so it has
                              no probability and is scored on accuracy only.
  naive_baseline              Pick the team with the better season win % so far.
                              A hard pick; accuracy only. NOTE: when both teams have
                              exactly the same win %, this rule picks the AWAY team.
  naive_baseline_ties_home    Same as naive_baseline but ties go to the home team.
                              Reported as a sensitivity check only.
  constant_probability        Probability baseline: predict the same probability for
                              every game, equal to the home-win rate in the TRAINING
                              seasons. This is the bar for log loss / Brier score.
"""
import pandas as pd


def always_home_baseline(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index)


def naive_baseline(df: pd.DataFrame) -> pd.Series:
    return (df["HOME_SEASON_WIN_PCT"] > df["AWAY_SEASON_WIN_PCT"]).astype(int)


def naive_baseline_ties_home(df: pd.DataFrame) -> pd.Series:
    return (df["HOME_SEASON_WIN_PCT"] >= df["AWAY_SEASON_WIN_PCT"]).astype(int)


def constant_probability(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    p = float(train["TARGET_HOME_WIN"].mean())
    return pd.Series(p, index=test.index)
