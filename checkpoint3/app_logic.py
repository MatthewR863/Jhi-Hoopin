
import numpy as np
import pandas as pd

ROLLING_WINDOW = 10
STAT_COLS = ["PTS", "REB", "TOV", "FG_PCT"]

# plain-English names for the 7 per-team features
FEATURE_LABELS = {
    "ROLL_PTS": "Points per game (last 10)",
    "ROLL_REB": "Rebounds per game (last 10)",
    "ROLL_TOV": "Turnovers per game (last 10)",
    "ROLL_FG_PCT": "Field goal % (last 10)",
    "ROLL_WIN_PCT": "Win % (last 10)",
    "SEASON_WIN_PCT": "Win % (season so far)",
    "REST_DAYS": "Days of rest",
}


def pretty_feature(name):
    """'HOME_ROLL_PTS' -> 'Home: Points per game (last 10)'."""
    side, _, base = name.partition("_")
    return "%s: %s" % (side.capitalize(), FEATURE_LABELS.get(base, base))


def current_teams(raw):
    """Teams that played in the most recent season in the data, sorted."""
    latest = sorted(raw["SEASON"].unique())[-1]
    return sorted(raw.loc[raw["SEASON"] == latest, "TEAM_ABBREVIATION"].unique())


def regular_season_only(raw):
    kind = raw["GAME_ID"].astype(str).str.zfill(10).str[:3]
    return raw[kind == "002"]


def latest_team_stats(raw, team):

    g = regular_season_only(raw)
    g = g[g["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")
    if g.empty:
        raise ValueError("No games found for team %s" % team)
    season = g.iloc[-1]["SEASON"]
    season_games = g[g["SEASON"] == season]
    last10 = g.tail(ROLLING_WINDOW)
    if len(season_games) < 3:
        raise ValueError("%s has only %d game(s) in %s; need at least 3." % (team, len(season_games), season))
    stats = {"ROLL_" + c: float(last10[c].mean()) for c in STAT_COLS}
    stats["ROLL_WIN_PCT"] = float((last10["WL"] == "W").mean())
    stats["SEASON_WIN_PCT"] = float((season_games["WL"] == "W").mean())
    stats["_season"] = season
    stats["_last_game"] = pd.Timestamp(g.iloc[-1]["GAME_DATE"])
    stats["_games_in_season"] = int(len(season_games))
    return stats


def build_feature_row(home_stats, away_stats, home_rest, away_rest, features):
    values = {}
    for side, stats, rest in (("HOME", home_stats, home_rest), ("AWAY", away_stats, away_rest)):
        for base in FEATURE_LABELS:
            values["%s_%s" % (side, base)] = rest if base == "REST_DAYS" else stats[base]
    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError("Model expects features this app cannot build: %s" % missing)
    return pd.DataFrame([[values[f] for f in features]], columns=features)


def predict_home_win(model, row):
    return float(model.predict_proba(row)[0, 1])


def describe_prediction(p_home, home, away):
    winner = home if p_home >= 0.5 else away
    p_win = p_home if p_home >= 0.5 else 1.0 - p_home
    if p_win < 0.55:
        label = "toss-up"
    elif p_win < 0.65:
        label = "lean"
    elif p_win < 0.80:
        label = "fairly confident"
    else:
        label = "very confident"
    return winner, p_win, label


def feature_importance(model, features):
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(clf, "coef_"):
        coef = np.ravel(clf.coef_)
        df = pd.DataFrame({"feature": features, "importance": np.abs(coef), "direction": np.where(coef >= 0, "helps home", "helps away")})
    elif hasattr(clf, "feature_importances_"):
        df = pd.DataFrame({"feature": features, "importance": np.ravel(clf.feature_importances_), "direction": ""})
    else:
        raise ValueError("Model type exposes no importances")
    df["label"] = [pretty_feature(f) for f in df["feature"]]
    total = df["importance"].sum()
    df["share"] = df["importance"] / total if total > 0 else 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
