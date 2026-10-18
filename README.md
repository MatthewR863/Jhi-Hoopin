# Jhi Hoopin

Predicting NBA game winners from pre-game team stats, benchmarked against
naive baselines. Split into three folders, one per project checkpoint —
each has its own README with scoped instructions.

## Structure

```
jhi_hoopin/
├── checkpoint1_data_and_baseline/   # Week 5: data + baseline
│   ├── fetch_data.py
│   ├── features.py
│   ├── baseline.py
│   ├── split.py
│   └── README.md
├── checkpoint2_ml_models/           # Week 7: self-contained, reproducible (see its README)
│   ├── run_all.py                   # ONE command: features -> verify -> train 3 models -> evaluate
│   ├── fetch_data.py, features.py, verify_features.py
│   ├── train.py, evaluate.py, baseline.py, split.py, inspect_results.py
│   ├── data/ models/ results/       # created by run_all.py
│   └── README.md
├── checkpoint3_web_app/             # Week 8: demo-ready web app
│   ├── app.py
│   └── README.md
├── data/                            # Checkpoint 1 outputs
└── requirements.txt
```

## Setup (once, from this top-level folder)

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running each checkpoint

See the README inside each `checkpointN_.../` folder for exact commands
and what to expect. Short version, run in order from this top-level folder:

```bash
# Checkpoint 1
python3 checkpoint1_data_and_baseline/fetch_data.py --seasons 2018-19 2019-20 2020-21 2021-22 2022-23 2023-24 2024-25
python3 checkpoint1_data_and_baseline/features.py
python3 checkpoint1_data_and_baseline/baseline.py

# Checkpoint 2 (copy data/raw/team_game_logs.csv into checkpoint2_ml_models/data/raw/ first)
python3 checkpoint2_ml_models/run_all.py

streamlit run checkpoint3_web_app/app.py
```

## Notes on data access

`nba_api` (used in `fetch_data.py`) hits `stats.nba.com`, which requires
real internet access — it won't work in a sandboxed/offline environment.
If you'd rather use the Kaggle route from the proposal, drop a CSV into
`data/raw/team_game_logs.csv` matching the column shape described in the
docstring at the top of `checkpoint1_data_and_baseline/features.py` —
everything downstream works the same either way.
