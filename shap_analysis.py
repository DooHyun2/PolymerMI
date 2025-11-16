import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 결과 폴더
os.makedirs("results", exist_ok=True)

# 1) 데이터 로드
CSV = "synthetic_polymer.csv"
if not os.path.exists(CSV):
    raise FileNotFoundError(f"{CSV} not found. Run data_synth.py first.")

df = pd.read_csv(CSV)
X = df.drop(columns=["density"])
y = df["density"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

# 2) RF 모델 (mi_baseline이랑 같은 느낌)
rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
rf.fit(Xtr, ytr)

# 3) SHAP 계산 (TreeExplainer: RF/GBDT 전용)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(Xte)   # shape = [n_samples, n_features]

# 4) summary plot (각 특성의 전체 중요도 + 분포)
shap.summary_plot(shap_values, Xte, show=False)
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=200)
plt.close()

# 5) 상위 3개 특성에 대한 dependence plot
top_feats = ["mw", "hyd", "xlink"]
for feat in top_feats:
    shap.dependence_plot(feat, shap_values, Xte, show=False)
    plt.tight_layout()
    plt.savefig(f"results/shap_depend_{feat}.png", dpi=200)
    plt.close()

print("saved SHAP plots -> results/shap_summary.png, shap_depend_*.png")

