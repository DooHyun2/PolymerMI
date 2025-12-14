import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# data road
CSV = "synthetic_polymer.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth.py first.")

df = pd.read_csv(CSV)

# feature / target 
X_all = df.drop(columns=["density"])
y_all = df["density"].values

feat_names = list(X_all.columns)
dim = X_all.shape[1]

# using RF (Random Forest)
rf_oracle = RandomForestRegressor(
    n_estimators=500,
    random_state=0,
    n_jobs=-1
)
rf_oracle.fit(X_all, y_all)

def f_oracle(x: np.ndarray) -> float:
    """
    x: shape (dim,)
    return: 예측 density (스칼라)
    """
    x = np.asarray(x).reshape(1, -1)
    return float(rf_oracle.predict(x)[0])

# set bound
  
bounds = []
for col in feat_names:
    lo = X_all[col].min()
    hi = X_all[col].max()
    bounds.append((float(lo), float(hi)))
bounds = np.array(bounds)   # shape (dim, 2)


rng = np.random.default_rng(42)

def sample_random(n: int) -> np.ndarray:
    """bounds 안에서 균일 샘플링"""
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    u = rng.random((n, dim))
    return lows + u * (highs - lows)

# first 10
n_init = 10
X_sample = sample_random(n_init)
y_sample = np.array([f_oracle(x) for x in X_sample])


# GPR + Expected Improvement(EI)

kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
    length_scale=np.ones(dim),
    length_scale_bounds=(0.1, 10.0)
) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))

def fit_gpr(X, y):
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=0,
        n_restarts_optimizer=3,
    )
    gpr.fit(X, y)
    return gpr

def expected_improvement(X_cand: np.ndarray,
                         gpr: GaussianProcessRegressor,
                         y_best: float,
                         xi: float = 0.01) -> np.ndarray:
    """
    X_cand: shape (N, d)
    EI(x) = E[max(0, f(x) - y_best - xi)]
    """
    from scipy.stats import norm

    mu, sigma = gpr.predict(X_cand, return_std=True)
    sigma = sigma.reshape(-1)

    # sigma=0
    sigma_safe = np.where(sigma < 1e-12, 1e-12, sigma)

    imp = mu - y_best - xi
    Z = imp / sigma_safe

    ei = imp * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    ei[sigma < 1e-12] = 0.0
    return ei

def propose_next(gpr: GaussianProcessRegressor,
                 n_candidates: int = 2000) -> np.ndarray:
    """
    후보 점들을 랜덤 샘플링하고 EI가 최대인 점을 선택
    """
    X_cand = sample_random(n_candidates)
    y_best = float(np.max(y_sample))
    ei = expected_improvement(X_cand, gpr, y_best)
    idx = int(np.argmax(ei))
    return X_cand[idx]



# BO loof

n_iter = 30  # BO cycle

history = []

for t in range(n_iter):
    # GPR fit
    gpr = fit_gpr(X_sample, y_sample)

    # last best model
    best_idx = int(np.argmax(y_sample))
    best_x = X_sample[best_idx]
    best_y = y_sample[best_idx]
    print(f"[iter {t:02d}] best_y = {best_y:.4f}")

    # history
    history.append({
        "iter": t,
        "best_y": best_y,
        **{f"best_{name}": float(val) for name, val in zip(feat_names, best_x)}
    })

    # suggest
    x_next = propose_next(gpr)
    y_next = f_oracle(x_next)

    # add samples
    X_sample = np.vstack([X_sample, x_next])
    y_sample = np.append(y_sample, y_next)


# save the results
best_idx = int(np.argmax(y_sample))
best_x = X_sample[best_idx]
best_y = y_sample[best_idx]

print("\n=== GPR-BO Result ===")
print("best_y (density):", best_y)
print("best_x:")
for name, val in zip(feat_names, best_x):
    print(f"  {name}: {val}")

os.makedirs("results", exist_ok=True)
hist_df = pd.DataFrame(history)
hist_df.to_csv("results/bo_gpr_history.csv", index=False)
print("saved -> results/bo_gpr_history.csv")

