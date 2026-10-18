"""
Scores every baseline and every trained model on the SAME held-out test games.

Fixes vs. the first version: the hard-pick baselines are no longer given invented
0.99/0.01 probabilities. They are scored on accuracy only. Log loss and Brier score
are compared against a proper probability baseline: one constant probability for
every game, equal to the home-win rate in the training seasons.

Outputs (in results/):
    model_comparison.csv    one row per method
    test_predictions.csv    one row per test game: GAME_ID, GAME_DATE, teams, label,
                            every baseline pick, and each model's probability + pick
    calibration_bins.csv    predicted vs observed home-win rate in 5 probability bins

Run after train.py:  python3 checkpoint2_ml_models/evaluate.py
"""
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import paths
from baseline import (always_home_baseline, constant_probability, naive_baseline,
                      naive_baseline_ties_home)
from split import train_test_split_by_season

N_BOOT = 2000
SEED = 0


def prob_row(name, y, p, note=""):
    pred = (p >= 0.5).astype(int)
    return {"method": name, "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p), "note": note}


def hard_row(name, y, pred, note=""):
    return {"method": name, "accuracy": accuracy_score(y, pred),
            "log_loss": np.nan, "brier": np.nan, "note": note}


def paired_bootstrap_lift(y, model_pred, base_pred):
    """95% interval for (model accuracy - baseline accuracy) over resampled test games."""
    rng = np.random.default_rng(SEED)
    correct_m = (model_pred == y).astype(float)
    correct_b = (base_pred == y).astype(float)
    n = len(y)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_m[idx].mean() - correct_b[idx].mean()
    return np.percentile(diffs, [2.5, 97.5]) * 100


def calibration_table(name, y, p):
    bins = pd.qcut(p, 5, duplicates="drop")
    t = pd.DataFrame({"y": y.values, "p": p, "bin": bins}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_home_win_rate=("y", "mean")).reset_index()
    t.insert(0, "model", name)
    t["bin"] = t["bin"].astype(str)
    return t


def main():
    df = pd.read_csv(paths.MATCHUP_PATH)
    if "GAME_DATE" not in df.columns:
        raise SystemExit("matchup_features.csv has no GAME_DATE column. Re-run features.py first.")
    train, test = train_test_split_by_season(df)
    y = test["TARGET_HOME_WIN"]

    always_home = always_home_baseline(test)
    naive = naive_baseline(test)
    naive_th = naive_baseline_ties_home(test)
    const_p = constant_probability(train, test)
    const_pick = (const_p >= 0.5).astype(int)
    p_train = float(const_p.iloc[0])

    rows = [
        hard_row("always_predict_home", y, always_home, "hard pick, no probability"),
        hard_row("naive_season_winpct", y, naive, "hard pick; ties go to away team"),
        hard_row("naive_season_winpct_ties_home", y, naive_th, "sensitivity check: ties go to home team"),
        prob_row("constant_prob_train_home_rate", y, const_p.values, "p = %.4f from training seasons" % p_train),
    ]

    pred_out = test[["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM", "AWAY_TEAM", "TARGET_HOME_WIN"]].copy()
    pred_out["PRED_always_home"] = always_home.values
    pred_out["PRED_naive"] = naive.values
    pred_out["PROB_constant"] = const_p.values

    model_preds = {}
    cal_tables = []
    for path in sorted(glob.glob(os.path.join(paths.MODEL_DIR, "*.joblib"))):
        name = os.path.splitext(os.path.basename(path))[0]
        saved = joblib.load(path)
        proba = saved["model"].predict_proba(test[saved["features"]])[:, 1]
        rows.append(prob_row(name, y, proba))
        pred_out["PROB_" + name] = proba
        pred_out["PRED_" + name] = (proba >= 0.5).astype(int)
        model_preds[name] = (proba >= 0.5).astype(int)
        cal_tables.append(calibration_table(name, y, proba))

    comp = pd.DataFrame(rows)
    ref = comp.loc[comp["method"] == "constant_prob_train_home_rate"].iloc[0]
    comp["logloss_vs_constant"] = comp["log_loss"] - ref["log_loss"]
    comp["brier_skill_vs_constant"] = 1 - comp["brier"] / ref["brier"]

    # best baseline by accuracy (sensitivity row excluded: it is a variant of the naive rule)
    base_candidates = {"always_predict_home": always_home.values, "naive_season_winpct": naive.values,
                       "constant_prob_train_home_rate": const_pick.values}
    base_acc = {k: accuracy_score(y, v) for k, v in base_candidates.items()}
    best_name = max(base_acc, key=base_acc.get)

    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    comp.to_csv(paths.COMPARISON_PATH, index=False)
    pred_out.sort_values(["GAME_DATE", "GAME_ID"]).to_csv(paths.PREDICTIONS_PATH, index=False)
    if cal_tables:
        pd.concat(cal_tables).to_csv(paths.CALIBRATION_PATH, index=False)

    n_tie = int((test["HOME_SEASON_WIN_PCT"] == test["AWAY_SEASON_WIN_PCT"]).sum())
    print("Test set: %d games, season %s | train: %d games | training home-win rate: %.4f"
          % (len(test), test["SEASON"].iloc[0], len(train), p_train))
    print("Games where both teams had identical season win %%: %d\n" % n_tie)
    show = comp[["method", "accuracy", "log_loss", "brier", "brier_skill_vs_constant"]].copy()
    print(show.to_string(index=False, float_format=lambda v: "%.4f" % v, na_rep="n/a"))
    print("\nlog loss / Brier are n/a for hard picks: they make no probability claim.")
    print("Best baseline by accuracy: %s (%.4f)" % (best_name, base_acc[best_name]))

    for name, pred in model_preds.items():
        lift = (accuracy_score(y, pred) - base_acc[best_name]) * 100
        lo, hi = paired_bootstrap_lift(y.values, pred, base_candidates[best_name])
        print("%s: accuracy lift over best baseline %+.1f pts (95%% bootstrap interval %+.1f to %+.1f)"
              % (name, lift, lo, hi))

    if cal_tables:
        print("\nCalibration (5 quantile bins, predicted vs observed home-win rate):")
        print(pd.concat(cal_tables).to_string(index=False, float_format=lambda v: "%.3f" % v))
    print("\nSaved: " + paths.COMPARISON_PATH)
    print("Saved: " + paths.PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
