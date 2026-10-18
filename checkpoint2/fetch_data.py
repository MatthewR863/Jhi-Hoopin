"""
Pulls team-level game logs from stats.nba.com via nba_api.

Needs live internet access to stats.nba.com, so run it on your own machine.

Usage (from anywhere):
    python3 checkpoint2_ml_models/fetch_data.py --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25

Output: data/raw/team_game_logs.csv (inside this folder). One row per team per game.
"""
import argparse
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

import paths


def fetch_season(season: str) -> pd.DataFrame:
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
        season_type_nullable="Regular Season",  # no preseason / playoffs / play-in
    )
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", required=True, help="e.g. 2018-19 2019-20")
    parser.add_argument("--out", default=paths.RAW_PATH)
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between season pulls")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    frames = []
    for season in args.seasons:
        print(f"Fetching {season}...")
        try:
            frames.append(fetch_season(season))
        except Exception as e:
            print(f"  Failed on {season}: {e}")
        time.sleep(args.sleep)

    if not frames:
        raise SystemExit("No seasons fetched successfully.")

    full = pd.concat(frames, ignore_index=True)
    # MATCHUP looks like "BOS vs. NYK" (home) or "BOS @ NYK" (away)
    full["IS_HOME"] = full["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    full["GAME_DATE"] = pd.to_datetime(full["GAME_DATE"])
    full = full.sort_values(["TEAM_ID", "GAME_DATE"])

    full.to_csv(args.out, index=False)
    print(f"Saved {len(full)} rows across {full['SEASON'].nunique()} seasons to {args.out}")


if __name__ == "__main__":
    main()
