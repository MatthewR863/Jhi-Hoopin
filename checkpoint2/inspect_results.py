"""
Reads results/test_predictions.csv (written by evaluate.py) and prints the example
and failure cases for the report, each with GAME_ID and GAME_DATE.

    python3 checkpoint2_ml_models/inspect_results.py
    python3 checkpoint2_ml_models/inspect_results.py --model rf
    python3 checkpoint2_ml_models/inspect_results.py --home ATL --away SAC   # find a specific game
"""
import argparse

import pandas as pd

import paths


def describe(row, model):
    p_home = row["PROB_" + model]
    pred_home = int(row["PRED_" + model])
    winner_pred = row["HOME_TEAM"] if pred_home else row["AWAY_TEAM"]
    winner_true = row["HOME_TEAM"] if row["TARGET_HOME_WIN"] else row["AWAY_TEAM"]
    conf = p_home if pred_home else 1 - p_home
    verdict = "correct" if pred_home == row["TARGET_HOME_WIN"] else "WRONG"
    return ("GAME_ID %s | %s | %s (home) vs %s (away)\n  predicted %s (%.1f%% confidence) | actual winner %s | %s"
            % (row["GAME_ID"], str(row["GAME_DATE"])[:10], row["HOME_TEAM"], row["AWAY_TEAM"],
               winner_pred, conf * 100, winner_true, verdict)), conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb")
    parser.add_argument("--home")
    parser.add_argument("--away")
    args = parser.parse_args()

    df = pd.read_csv(paths.PREDICTIONS_PATH)
    m = args.model
    if "PROB_" + m not in df.columns:
        raise SystemExit("No predictions for model '%s'. Available: %s" % (
            m, ", ".join(c[5:] for c in df.columns if c.startswith("PROB_") and c != "PROB_constant")))

    df["CORRECT"] = df["PRED_" + m] == df["TARGET_HOME_WIN"]
    print("Model: %s | test games: %d | correct: %d | wrong: %d | accuracy: %.4f\n"
          % (m, len(df), df["CORRECT"].sum(), (~df["CORRECT"]).sum(), df["CORRECT"].mean()))

    if args.home and args.away:
        hits = df[(df["HOME_TEAM"] == args.home) & (df["AWAY_TEAM"] == args.away)]
        print("--- All %s (home) vs %s (away) games in the test set: %d ---" % (args.home, args.away, len(hits)))
        for _, r in hits.iterrows():
            print(describe(r, m)[0])
        print()

    print("--- Example prediction (earliest correct game in the test season) ---")
    print(describe(df[df["CORRECT"]].iloc[0], m)[0])

    wrong = df[~df["CORRECT"]].copy()
    wrong["CONF"] = [describe(r, m)[1] for _, r in wrong.iterrows()]
    print("\n--- Three most confident wrong predictions ---")
    for _, r in wrong.sort_values("CONF", ascending=False).head(3).iterrows():
        print(describe(r, m)[0])


if __name__ == "__main__":
    main()
