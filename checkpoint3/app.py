
import os
import sys

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "checkpoint2_ml_models"))

import app_logic as logic  # noqa: E402
import paths

st.set_page_config(page_title="Jhi Hoopin", page_icon="🏀", layout="centered")


@st.cache_data
def load_raw(path, mtime):
    return pd.read_csv(path, parse_dates=["GAME_DATE"])


@st.cache_resource
def load_model(path, mtime):
    saved = joblib.load(path)
    return saved["model"], saved["features"]


@st.cache_data
def load_comparison(path, mtime):
    return pd.read_csv(path)


def mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


def setup_error(msg):
    st.error(msg)
    st.code("python3 checkpoint2_ml_models/run_all.py", language="bash")
    st.stop()


st.title("🏀 Jhi Hoopin")
st.caption("Predicts an NBA game winner from each team's recent form. "
           "A class project, not betting advice.")

if not os.path.exists(paths.RAW_PATH):
    setup_error("Game data not found. Build it first:")
model_files = sorted(f for f in os.listdir(paths.MODEL_DIR) if f.endswith(".joblib")) if os.path.isdir(paths.MODEL_DIR) else []
if not model_files:
    setup_error("No trained models found. Train them first:")

raw = load_raw(paths.RAW_PATH, mtime(paths.RAW_PATH))
teams = logic.current_teams(raw)
latest_season = sorted(raw["SEASON"].unique())[-1]

comparison = None
if os.path.exists(paths.COMPARISON_PATH):
    comparison = load_comparison(paths.COMPARISON_PATH, mtime(paths.COMPARISON_PATH))

model_names = [os.path.splitext(f)[0] for f in model_files]
default_model = model_names[0]
if comparison is not None:
    scored = comparison[comparison["method"].isin(model_names)].dropna(subset=["log_loss"])
    if not scored.empty:
        default_model = scored.sort_values("log_loss").iloc[0]["method"]

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", model_names, index=model_names.index(default_model),
                              help="Defaults to the model with the lowest log loss on the held-out season.")
    st.caption("Stats come from games through %s (%s season)." % (raw["GAME_DATE"].max().date(), latest_season))

tab_predict, tab_why, tab_eval = st.tabs(["Predict a game", "What drives it", "How good is it?"])

model, features = load_model(os.path.join(paths.MODEL_DIR, model_name + ".joblib"),
                             mtime(os.path.join(paths.MODEL_DIR, model_name + ".joblib")))

with tab_predict:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=0)
    away = c2.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    r1, r2 = st.columns(2)
    home_rest = r1.number_input("Home days of rest", min_value=0, max_value=10, value=2)
    away_rest = r2.number_input("Away days of rest", min_value=0, max_value=10, value=2)

    if home == away:
        st.warning("Pick two different teams.")
    else:
        try:
            hs = logic.latest_team_stats(raw, home)
            as_ = logic.latest_team_stats(raw, away)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        row = logic.build_feature_row(hs, as_, home_rest, away_rest, features)
        p_home = logic.predict_home_win(model, row)
        winner, p_win, label = logic.describe_prediction(p_home, home, away)

        st.subheader("Predicted winner: %s" % winner)
        m1, m2, m3 = st.columns(3)
        m1.metric("%s (home) win probability" % home, "%.1f%%" % (p_home * 100))
        m2.metric("%s (away) win probability" % away, "%.1f%%" % ((1 - p_home) * 100))
        m3.metric("Read", label)
        st.progress(min(max(p_home, 0.0), 1.0))
        if label == "toss-up":
            st.info("The model sees these teams as nearly even.")

        st.markdown("**Inputs the model used**")
        rows = []
        for base, name in logic.FEATURE_LABELS.items():
            h = home_rest if base == "REST_DAYS" else hs[base]
            a = away_rest if base == "REST_DAYS" else as_[base]
            if base in ("ROLL_FG_PCT", "ROLL_WIN_PCT", "SEASON_WIN_PCT"):
                fmt = lambda v: "%.1f%%" % (v * 100)  # noqa: E731
            elif base == "REST_DAYS":
                fmt = lambda v: "%d" % v  # noqa: E731
            else:
                fmt = lambda v: "%.1f" % v  # noqa: E731
            rows.append({"Stat": name, home + " (home)": fmt(h), away + " (away)": fmt(a)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("%s: %d games this season, last played %s. %s: %d games, last played %s."
                   % (home, hs["_games_in_season"], hs["_last_game"].date(),
                      away, as_["_games_in_season"], as_["_last_game"].date()))

with tab_why:
    st.markdown("Which inputs move the **%s** model the most (share of total influence)." % model_name)
    imp = logic.feature_importance(model, features)
    imp["pct"] = imp["share"] * 100
    cols = ["label", "pct"] + (["direction"] if imp["direction"].astype(bool).any() else [])
    st.dataframe(
        imp[cols].rename(columns={"label": "Input", "direction": "Higher value..."}),
        hide_index=True,
        column_config={"pct": st.column_config.ProgressColumn(
            "Share of influence (%)", min_value=0.0, max_value=100.0, format="%.0f")},
    )
    if imp["direction"].astype(bool).any():
        st.caption("Logistic regression: influence is the size of each standardized coefficient.")
    else:
        st.caption("Tree model: influence is the built-in feature importance. "
                   "It shows what the model uses, not that a stat causes wins.")

with tab_eval:
    if comparison is None:
        st.info("Run checkpoint2_ml_models/run_all.py to generate the evaluation table.")
    else:
        st.markdown("Tested on every game of the most recent season, which none of the models saw in training.")
        show = comparison[["method", "accuracy", "log_loss", "brier"]].copy()
        show["accuracy"] = (show["accuracy"] * 100).map("{:.1f}%".format)
        for c in ("log_loss", "brier"):
            show[c] = show[c].map(lambda v: "n/a" if pd.isna(v) else "%.3f" % v)
        st.dataframe(show.rename(columns={"method": "Method", "accuracy": "Accuracy",
                                          "log_loss": "Log loss", "brier": "Brier"}),
                     hide_index=True)
        st.caption("Lower log loss and Brier mean better probabilities. Hard-pick baselines make no "
                   "probability claim, so those cells are n/a.")
    st.markdown("**Limits.** Only team-level box-score stats, no injuries or lineups. "
                "Pro basketball is noisy: even a good model is wrong roughly one game in three.")
