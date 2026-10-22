import h5py
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    WhiteKernel,
)

GOOD_DEMO = (
    "demos/MazeDrive-v2/good/20260929_123829.h5"
)

BAD_DEMO = (
    "demos/MazeDrive-v2/bad/20260929_123944.h5"
)

OUTPUT_FILE = "gp_danger_model.pkl"

# Wall positions
WALL_START = 34
WALL_END = 79

SEED = 42


def load_demo_obs(path):

    with h5py.File(path, "r") as f:
        obs = f["traj_0/obs"][:]

    return obs.astype(np.float64)


good_obs = load_demo_obs(GOOD_DEMO)
bad_obs = load_demo_obs(BAD_DEMO)

print("Good observations:", good_obs.shape)
print("Bad observations: ", bad_obs.shape)


good_X = good_obs[:, WALL_START:WALL_END]
bad_X = bad_obs[:, WALL_START:WALL_END]

print("Wall feature dimension:", good_X.shape[1])


good_y = np.zeros(len(good_X), dtype=np.int64)
bad_y = np.ones(len(bad_X), dtype=np.int64)


X = np.concatenate([good_X, bad_X], axis=0)

y = np.concatenate([good_y, bad_y], axis=0)

print("Training X:", X.shape)
print("Training y:", y.shape)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


kernel = (ConstantKernel(1.0) * RBF(length_scale=np.ones(WALL_END - WALL_START)) + WhiteKernel(noise_level=1e-3))


gp = GaussianProcessClassifier(
    kernel=kernel,
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=2,
    random_state=SEED,
)

print()
print("Training GP...")

gp.fit(
    X_scaled,
    y,
)

print("GP training complete.")

print()
print("Learned kernel:")
print(gp.kernel_)


# ============================================================
# SAVE
# ============================================================

with open(OUTPUT_FILE, "wb") as f:

    pickle.dump(
        {
            "gp": gp,
            "scaler": scaler,
            "wall_start": WALL_START,
            "wall_end": WALL_END,
        },
        f,
    )

print()
print("Saved to:", OUTPUT_FILE)