# Checkpoint 1 — Data & Baseline (Week 5)

**Goal:** data pulled, cleaned, and a baseline model running with a first
accuracy number.

## Files here

- `fetch_data.py` — pulls team game logs from `stats.nba.com` via `nba_api`.
  Run this on your own machine — it needs live internet access and won't
  work in a network-restricted sandbox.
- `features.py` — turns raw per-team-game rows into matchup-level rows with
  last-10-game rolling averages (points, rebounds, turnovers, FG%), season
  win%, and rest days.
- `baseline.py` — the two non-ML baselines: "team with the better season
  win% wins" and "home team always wins."
- `split.py` — shared helper: holds out the most recent season as the test
  set (chronological split, not random — more honest for sports prediction).

## Run it

From the project root (one level up from this folder):

```bash
pip install -r requirements.txt
python3 checkpoint1_data_and_baseline/fetch_data.py --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
python3 checkpoint1_data_and_baseline/features.py
python3 checkpoint1_data_and_baseline/baseline.py
```

This produces:
- `data/raw/team_game_logs.csv` — cleaned raw data
- `data/processed/matchup_features.csv` — engineered matchup features
- Printed accuracy for both baselines — write these numbers down, they're
  what Checkpoint 2's ML model has to beat.

## Using a Kaggle CSV instead

If you'd rather use a pre-packaged Kaggle dataset instead of `fetch_data.py`,
skip that script and drop your CSV in `data/raw/team_game_logs.csv` with
columns matching what's described in the docstring at the top of
`features.py`.
