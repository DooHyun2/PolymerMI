# data_synth.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600

mw = rng.uniform(5, 500, n)
xlink = rng.integers(0, 6, n)
hyd = rng.uniform(0.05, 0.7, n)
side = rng.uniform(0.0, 0.6, n)
tg_like = rng.uniform(-50, 120, n)

noise = rng.normal(0, 0.01, n)
target_density = (
    1.00 + 0.0012*mw - 0.045*xlink + 0.28*hyd
    - 0.06*side + 0.0008*mw*hyd + 0.0005*xlink*tg_like/100
    + noise
)

df = pd.DataFrame({
    "mw": mw, "xlink": xlink, "hyd": hyd, "side": side, "tg_like": tg_like,
    "density": target_density
})
df.to_csv("synthetic_polymer.csv", index=False)
print("saved: synthetic_polymer.csv", df.shape)

