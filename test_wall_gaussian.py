import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "gaussian_danger_models.pkl"

GOOD_DEMO = "/Users/bhagyashreerane/Projects/melon-rl/demos/MazeDrive-v1/good/20260927_224013.h5"

BAD_DEMO = "/Users/bhagyashreerane/Projects/melon-rl/demos/MazeDrive-v1/bad/20260927_224044.h5"

TEST_DEMO = "/Users/bhagyashreerane/Projects/melon-rl/demos/MazeDrive-v1/bad/20260928_073542.h5"

# Wall positions are observation indices 34-78
WALL_START = 34
WALL_END = 79

REG = 1e-4


# ============================================================
# LOAD OBSERVATIONS
# ============================================================

def load_demo_obs(path):

    with h5py.File(path, "r") as f:
        obs = f["traj_0/obs"][:]

    return obs.astype(np.float64)


# ============================================================
# FIT GAUSSIAN
# ============================================================

def fit_gaussian(X):

    mean = np.mean(X, axis=0)

    covariance = np.cov(
        X,
        rowvar=False,
    )

    covariance += REG * np.eye(X.shape[1])

    inv_covariance = np.linalg.pinv(covariance)

    return mean, inv_covariance


# ============================================================
# MAHALANOBIS
# ============================================================

def mahalanobis(X, mean, inv_covariance):

    diff = X - mean

    return np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            diff,
            inv_covariance,
            diff,
        )
    )


# ============================================================
# LOAD DEMOS
# ============================================================

good_obs = load_demo_obs(GOOD_DEMO)
bad_obs = load_demo_obs(BAD_DEMO)
test_obs = load_demo_obs(TEST_DEMO)

print("Good:", good_obs.shape)
print("Bad: ", bad_obs.shape)
print("Test:", test_obs.shape)


# ============================================================
# EXTRACT WALL POSITIONS ONLY
# ============================================================

good_wall = good_obs[:, WALL_START:WALL_END]
bad_wall = bad_obs[:, WALL_START:WALL_END]
test_wall = test_obs[:, WALL_START:WALL_END]

print()
print("Wall dimensions:", good_wall.shape[1])


# ============================================================
# FIT GOOD + BAD GAUSSIANS
# ============================================================

good_mean, good_inv_cov = fit_gaussian(good_wall)

bad_mean, bad_inv_cov = fit_gaussian(bad_wall)


# ============================================================
# CALCULATE DANGER
# ============================================================

d_good = mahalanobis(
    test_wall,
    good_mean,
    good_inv_cov,
)

d_bad = mahalanobis(
    test_wall,
    bad_mean,
    bad_inv_cov,
)

danger = d_good - d_bad


# ============================================================
# PRINT
# ============================================================

print()
print("Wall-only danger:")
print()

for t in range(len(danger)):

    print(
        f"t={t:3d}: "
        f"d_good={d_good[t]:8.3f}  "
        f"d_bad={d_bad[t]:8.3f}  "
        f"danger={danger[t]:8.3f}"
    )


print()
print("Minimum danger:", danger.min())
print("Maximum danger:", danger.max())
print("Mean danger:   ", danger.mean())


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.arange(len(danger)),
    danger,
    marker="o",
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Timestep")
plt.ylabel("Danger")
plt.title("Wall-only Mahalanobis Danger")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()