"""
Central place for every file location, so all scripts agree.

By default everything lives INSIDE this folder:
    checkpoint2_ml_models/data/raw/team_game_logs.csv
    checkpoint2_ml_models/data/processed/matchup_features.csv
    checkpoint2_ml_models/models/*.joblib
    checkpoint2_ml_models/results/*.csv

Set the environment variable JHI_WORKDIR to point somewhere else
(run_all.py --demo does this so synthetic demo data never touches real data).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("JHI_WORKDIR", HERE)

RAW_DIR = os.path.join(WORK, "data", "raw")
PROCESSED_DIR = os.path.join(WORK, "data", "processed")
MODEL_DIR = os.path.join(WORK, "models")
RESULTS_DIR = os.path.join(WORK, "results")

RAW_PATH = os.path.join(RAW_DIR, "team_game_logs.csv")
MATCHUP_PATH = os.path.join(PROCESSED_DIR, "matchup_features.csv")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "test_predictions.csv")
COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
CALIBRATION_PATH = os.path.join(RESULTS_DIR, "calibration_bins.csv")
