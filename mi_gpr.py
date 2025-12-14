import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Load data
CSV = "synthetic_polymer.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth.py first.")

df = pd.read_csv(CSV)
X = df.drop(columns=["density"])
y = df["density"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

# Define kernel (RBF + white noise)
n_features = X.shape[1]
kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(
    length_scale=np.ones(n_features),
    length_scale_bounds=(0.1, 10.0)
) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42
)

# 3) Train model
gpr.fit(Xtr, ytr)

# Predict
y_pred, y_std = gpr.predict(Xte, return_std=True)

r2 = r2_score(yte, y_pred)
mae = mean_absolute_error(yte, y_pred)

print("GPR R2:", round(r2, 4))
print("GPR MAE:", round(mae, 4))
print("Kernel after fit:", gpr.kernel_)

# plot save
os.makedirs("results", exist_ok=True)

plt.figure()
plt.scatter(yte, y_pred, s=10)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "k--", linewidth=1)
plt.xlabel("True density")
plt.ylabel("Predicted density (GPR)")
plt.title("GPR: true vs pred")
plt.tight_layout()
plt.savefig("results/gpr_true_vs_pred.png", dpi=200)
plt.close()

print("saved -> results/gpr_true_vs_pred.png")

