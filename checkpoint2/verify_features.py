"""
Independent check that the rolling features are leakage-free.

For a random sample of games, this re-computes each team's pre-game features from
scratch straight from the raw game log, using ONLY games played strictly before that
game's date, and compares them to what features.py wrote to matchup_features.csv.
It deliberately uses a different method (filter by date, take the last 10 rows)
from features.py (groupby + shift + rolling), so agreement is real evidence.

    python3 checkpoint2_ml_models/verify_features.py
Exit code 0 = every sampled value matched; 1 = a mismatch was found.
"""
import sys

import numpy as np
import pandas as pd

import paths
from features import MIN_GAMES, ROLLING_WINDOW, STAT_COLS

N_SAMPLE = 300
TOL = 1e-9


def main():
    raw = pd.read_csv(paths.RAW_PATH, parse_dates=["GAME_DATE"])
    from features import regular_season_only
    raw = regular_season_only(raw)
    raw["WIN"] = (raw["WL"] == "W").astype(int)
    m = pd.read_csv(paths.MATCHUP_PATH, parse_dates=["GAME_DATE"])
    sample = m.sample(min(N_SAMPLE, len(m)), random_state=0)

    checks, failures, max_diff = 0, [], 0.0
    for _, row in sample.iterrows():
        for side, team_col in (("HOME", "HOME_TEAM"), ("AWAY", "AWAY_TEAM")):
            this_game = raw[(raw["GAME_ID"] == row["GAME_ID"]) & (raw["TEAM_ABBREVIATION"] == row[team_col])]
            team_id = this_game.iloc[0]["TEAM_ID"]
            hist = raw[(raw["TEAM_ID"] == team_id) & (raw["GAME_DATE"] < row["GAME_DATE"])].sort_values("GAME_DATE")
            last_n = hist.tail(ROLLING_WINDOW)
            season_hist = hist[hist["SEASON"] == row["SEASON"]]
            if len(last_n) < MIN_GAMES or len(season_hist) < MIN_GAMES:
                continue
            expected = {"ROLL_" + c: last_n[c].mean() for c in STAT_COLS}
            expected["ROLL_WIN_PCT"] = last_n["WIN"].mean()
            expected["SEASON_WIN_PCT"] = season_hist["WIN"].mean()
            expected["REST_DAYS"] = (row["GAME_DATE"] - hist["GAME_DATE"].iloc[-1]).days
            for feat, exp in expected.items():
                got = row[side + "_" + feat]
                diff = abs(got - exp)
                checks += 1
                max_diff = max(max_diff, diff)
                if not np.isfinite(diff) or diff > TOL:
                    failures.append((row["GAME_ID"], str(row["GAME_DATE"])[:10], row[team_col], feat, got, exp))

    print("Games sampled: %d | feature values checked: %d | max abs difference: %.2e"
          % (len(sample), checks, max_diff))
    if failures:
        print("MISMATCHES FOUND (first 10):")
        for f in failures[:10]:
            print("  game %s %s %s %s: file=%.6f independent=%.6f" % f)
        sys.exit(1)
    print("OK: every sampled feature equals its independent recomputation from prior games only.")


if __name__ == "__main__":
    main()
