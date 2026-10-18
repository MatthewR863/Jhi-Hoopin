# Final: web app

Pick a home team and an away team, get a predicted winner and win probability.

Three tabs:
- **Predict a game**: winner, both teams' win probabilities, and the exact inputs the model used
  (each team's last-10-game stats through its most recent game; days of rest are editable).
- **What drives it**: which inputs the chosen model relies on most.
- **How good is it?**: the held-out-season comparison of every model against the baselines.

## Run it

From the `jhi_hoopin` folder, venv active:

```bash
pip install -r requirements.txt
python3 checkpoint2_ml_models/run_all.py      # builds data, models, results (skip if already done)
streamlit run checkpoint3_web_app/app.py
```

It opens in your browser. If the data or models are missing, the app says so and shows the command above.
The model dropdown defaults to the model with the lowest log loss on the held-out season.

## Files

- `app.py`: the Streamlit interface.
- `app_logic.py`: the prediction logic, kept separate from the interface so it can be tested on its own.
