# Checkpoint 2: models vs. baselines (reproducible)

Everything the Checkpoint 2 report cites is rebuilt by ONE command from raw game logs.
This folder is self-contained: data fetch, feature building, leakage check, training,
evaluation and per-game predictions are all here.

## Run it (Mac, from the jhi_hoopin folder, venv active)

```bash
pip install -r requirements.txt

# Option A: you already have the raw CSV from Checkpoint 1
mkdir -p checkpoint2_ml_models/data/raw
cp data/raw/team_game_logs.csv checkpoint2_ml_models/data/raw/
python3 checkpoint2_ml_models/run_all.py

# Option B: pull the real data again (needs internet)
python3 checkpoint2_ml_models/run_all.py --fetch
```

`run_all.py --demo` runs the same pipeline on SYNTHETIC data in `demo_run/` as a smoke
test. Demo numbers are not NBA results and must never go in a report.

If xgboost fails to load on a Mac, run `brew install libomp`. logreg and rf still train
without it, but `run_all.py` will report the xgb failure and stop.

## What gets produced

| File | What it is |
|---|---|
| `data/raw/team_game_logs.csv` | raw team-game rows from nba_api |
| `data/processed/matchup_features.csv` | one row per game: pre-game features, GAME_ID, GAME_DATE, label |
| `models/{logreg,rf,xgb}.joblib` | the trained models (identical chronological split) |
| `results/model_comparison.csv` | accuracy, log loss, Brier score, one row per method |
| `results/test_predictions.csv` | every test game: GAME_ID, GAME_DATE, teams, label, each baseline pick, each model's probability and pick |
| `results/calibration_bins.csv` | predicted vs observed home-win rate in 5 bins |
| `results/environment.txt` | Python and library versions used |

## Method in brief

- Split: most recent season is the test set, all earlier seasons train. Same split for every method.
- Features: last-10-game rolling averages (PTS, REB, TOV, FG%), rolling win %, season win %, rest days, for each team.
  Every rolling value is computed from games BEFORE the current one (`shift(1)`).
  `verify_features.py` re-derives a random sample of values a different way (filter by date, take last 10 rows) and must match.
- Baselines:
  - `always_predict_home`: hard pick, scored on accuracy only.
  - `naive_season_winpct`: pick the team with the better season win % so far (ties go to the away team; `..._ties_home` is a sensitivity check).
  - `constant_prob_train_home_rate`: the same probability for every game, equal to the home-win rate in the TRAINING seasons. This is the bar for log loss and Brier score.
- Hard-pick baselines have no probabilities, so log loss and Brier are not computed for them.
  (The first version assigned them 0.99/0.01, which made their log loss meaningless.)
- Accuracy lift over the best baseline is reported with a 95% paired-bootstrap interval, because one season of games is a small sample.

## Inspecting individual games

```bash
python3 checkpoint2_ml_models/inspect_results.py                 # example + 3 worst misses, with GAME_ID and date
python3 checkpoint2_ml_models/inspect_results.py --model rf
python3 checkpoint2_ml_models/inspect_results.py --home ATL --away SAC
```
