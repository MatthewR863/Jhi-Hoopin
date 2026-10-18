"""
Generates a SYNTHETIC game log (not real NBA data) in the same column layout as
fetch_data.py, so the whole pipeline can be smoke-tested without internet access.

Results from synthetic data say nothing about real NBA games. This script refuses to
write into the real data folder; run_all.py --demo points it at demo_run/ instead.
"""
import argparse
import os

import numpy as np
import pandas as pd

import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing synthetic file")
    args = parser.parse_args()

    if os.path.abspath(paths.WORK) == paths.HERE:
        raise SystemExit("Refusing to write synthetic data into the real data folder. "
                         "Use run_all.py --demo (sets JHI_WORKDIR to demo_run/).")
    if os.path.exists(paths.RAW_PATH) and not args.force:
        raise SystemExit(paths.RAW_PATH + " already exists (use --force to overwrite).")

    rng = np.random.default_rng(7)
    n_teams = 12
    team_ids = [1610612700 + i for i in range(n_teams)]
    abbrs = ["T%02d" % i for i in range(n_teams)]
    base_strength = rng.normal(0, 1, n_teams)

    rows, game_id = [], 22000001
    for si, season in enumerate(["2021-22", "2022-23", "2023-24"]):
        start = pd.Timestamp("%d-10-20" % (2021 + si))
        strength = base_strength + rng.normal(0, 0.4, n_teams)
        for day in range(70):
            date = start + pd.Timedelta(days=2 * day)
            order = rng.permutation(n_teams)
            for i in range(0, n_teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                p_home = 1 / (1 + np.exp(-(strength[h] - strength[a] + 0.35)))
                home_win = bool(rng.random() < p_home)
                for t, o, is_home, won in ((h, a, 1, home_win), (a, h, 0, not home_win)):
                    rows.append({
                        "TEAM_ID": team_ids[t], "TEAM_ABBREVIATION": abbrs[t], "GAME_ID": game_id,
                        "GAME_DATE": date.strftime("%Y-%m-%d"),
                        "MATCHUP": "%s vs. %s" % (abbrs[t], abbrs[o]) if is_home else "%s @ %s" % (abbrs[t], abbrs[o]),
                        "WL": "W" if won else "L",
                        "PTS": round(108 + 4 * strength[t] + (3 if won else -3) + rng.normal(0, 8), 0),
                        "REB": round(43 + 1.5 * strength[t] + rng.normal(0, 5), 0),
                        "TOV": round(13 - 0.8 * strength[t] + rng.normal(0, 3), 0),
                        "FG_PCT": round(0.46 + 0.015 * strength[t] + rng.normal(0, 0.035), 3),
                        "SEASON": season, "IS_HOME": is_home,
                    })
                game_id += 1

    os.makedirs(paths.RAW_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.RAW_PATH, index=False)
    with open(os.path.join(paths.RAW_DIR, "SYNTHETIC_DATA_NOTICE.txt"), "w") as f:
        f.write("team_game_logs.csv in this folder is SYNTHETIC demo data, not real NBA games.\n")
    print("Wrote %d synthetic team-game rows to %s" % (len(rows), paths.RAW_PATH))


if __name__ == "__main__":
    main()
