import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "gp_danger_model.pkl"

TEST_DEMO = (
    "demos/MazeDrive-v2/test/20260929_124145.h5"
)


# ============================================================
# LOAD OBS
# ============================================================

def load_demo_obs(path):

    with h5py.File(path, "r") as f:
        obs = f["traj_0/obs"][:]

    return obs.astype(np.float64)


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_FILE, "rb") as f:
    saved = pickle.load(f)

gp = saved["gp"]
scaler = saved["scaler"]

WALL_START = saved["wall_start"]
WALL_END = saved["wall_end"]


# ============================================================
# LOAD TEST DEMO
# ============================================================

obs = load_demo_obs(TEST_DEMO)

X = obs[:, WALL_START:WALL_END]

X_scaled = scaler.transform(X)


# ============================================================
# GP PREDICTION
# ============================================================

# Probability of class 1 = dangerous
probabilities = gp.predict_proba(X_scaled)

danger_probability = probabilities[:, 1]


# ============================================================
# PRINT
# ============================================================

print("Test demo:", TEST_DEMO)
print("Observations:", obs.shape)

print()
print("Danger probabilities:")

for t, p in enumerate(danger_probability):

    print(
        f"t={t:3d}: "
        f"P(danger)={p:.4f}"
    )


print()
print("Minimum:",
      danger_probability.min())

print("Maximum:",
      danger_probability.max())

print("Mean:",
      danger_probability.mean())


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(len(danger_probability)),
    danger_probability,
    marker="o",
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Timestep")
plt.ylabel("P(danger)")
plt.title("GP Danger Probability")

plt.ylim(0, 1)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()