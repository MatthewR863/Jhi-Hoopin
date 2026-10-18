"""
Pulls team-level game logs from stats.nba.com via nba_api.

RUN THIS ON YOUR OWN MACHINE, not in a sandboxed environment — nba_api
needs live internet access to stats.nba.com.

Usage:
    python data/fetch_data.py --seasons 2018-19 2019-20 2022-23

Output:
    data/raw/team_game_logs.csv
    One row per team per game, with basic box score stats and game date,
    matchup, W/L, home/away flag.
"""
import argparse
import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder


def fetch_season(season: str) -> pd.DataFrame:
    """Fetch all team game logs for a single season, e.g. '2023-24'."""
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
    )
    df = finder.get_data_frames()[0]
    df["SEASON"] = season
    return df


def add_home_away(df: pd.DataFrame) -> pd.DataFrame:
    # MATCHUP column looks like "BOS @ NYK" (away) or "BOS vs. NYK" (home)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        help="Seasons to pull, e.g. 2018-19 2019-20 2023-24",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw", "team_game_logs.csv"),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between season pulls (be polite to the API)",
    )
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
    full = add_home_away(full)
    full["GAME_DATE"] = pd.to_datetime(full["GAME_DATE"])
    full = full.sort_values(["TEAM_ID", "GAME_DATE"])

    full.to_csv(args.out, index=False)
    print(f"Saved {len(full)} rows across {full['SEASON'].nunique()} seasons to {args.out}")


if __name__ == "__main__":
    main()
