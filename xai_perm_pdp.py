import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

os.makedirs("results", exist_ok=True)
df = pd.read_csv("synthetic_polymer.csv")
X, y = df.drop(columns=["density"]), df["density"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(Xtr, ytr)

perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)
pi = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("Permutation importance:\n", pi)
pi.to_csv("results/perm_importance.csv")

top = list(pi.index[:3])
disp = PartialDependenceDisplay.from_estimator(rf, X, features=top, kind="average")
plt.tight_layout(); plt.savefig("results/pdp_top.png", dpi=160); plt.close()
print("saved -> results/perm_importance.csv, pdp_top.png")
