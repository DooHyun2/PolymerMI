# Bayesian optimization using a Random Forest model (Optuna)
import optuna, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load synthetic dataset
df = pd.read_csv("synthetic_polymer.csv")
# Split features and target
X, y = df.drop(columns=["density"]), df["density"]
# Train / test split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr, ytr)

def objective(trial):
    """Objective function for Optuna using RF prediction."""
    mw = trial.suggest_float("mw", 5, 500)
    xk = trial.suggest_int("xlink", 0, 5)
    hyd = trial.suggest_float("hyd", 0.05, 0.7)
    side = trial.suggest_float("side", 0.0, 0.6)
    tg = trial.suggest_float("tg_like", -50, 120)
    return rf.predict(np.array([[mw, xk, hyd, side, tg]]))[0]

# Create and run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=120)

# Print best result
print("best params:", study.best_params)
print("best predicted density:", study.best_value)

# Save results
pd.Series(study.best_params).to_json("results/bo_best_params.json")
with open("results/bo_best_value.txt","w") as f: f.write(str(study.best_value))
print("saved -> results/bo_best_params.json, bo_best_value.txt")

