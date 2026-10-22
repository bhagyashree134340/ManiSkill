import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

MODEL_FILE = "gaussian_danger_models.pkl"
TEST_DEMO = "demos/MazeDrive-v2/test/20260929_124145.h5"

def load_demo_obs(path):
    with h5py.File(path, "r") as f:
        obs = f["traj_0/obs"][:]
    return obs.astype(np.float64)

def mahalanobis_distance(X, gaussian):
    mean = gaussian["mean"]
    inv_cov = gaussian["inv_covariance"]
    diff = X - mean
    distance = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))
    return distance

def calculate_danger(obs, models):
    all_dangers = []

    for model in models:
        indices = model["feature_indices"]
        X = obs[:, indices]

        d_good = mahalanobis_distance(X, model["good"])
        d_bad = mahalanobis_distance(X, model["bad"])

        danger = d_good - d_bad
        all_dangers.append(danger)

    all_dangers = np.array(all_dangers)
    return all_dangers

with open(MODEL_FILE, "rb") as f:
    models = pickle.load(f)

print("Loaded models:", len(models))

test_obs = load_demo_obs(TEST_DEMO)

print("Test demo:", TEST_DEMO)
print("Test observations:", test_obs.shape)

expected_dim = max(np.max(model["feature_indices"]) for model in models) + 1

if test_obs.shape[1] < expected_dim:
    raise ValueError(f"Test demo has only {test_obs.shape[1]} dimensions, but model requires at least {expected_dim}.")

all_dangers = calculate_danger(test_obs, models)

mean_danger = np.mean(all_dangers, axis=0)
min_danger = np.min(all_dangers, axis=0)
max_danger = np.max(all_dangers, axis=0)
positive_models = np.sum(all_dangers > 0, axis=0)

print()
print("Danger statistics per timestep:")
print()

for t in range(len(test_obs)):
    print(f"t={t:3d}: mean={mean_danger[t]:8.3f} min={min_danger[t]:8.3f} max={max_danger[t]:8.3f} positive={positive_models[t]:3d}/100")

print()
print("Overall statistics:")
print()

print("Minimum mean danger:", mean_danger.min())
print("Maximum mean danger:", mean_danger.max())
print("Mean of mean danger:", mean_danger.mean())

print()
print("Maximum number of positive models:", positive_models.max())
print("Timestep with most positive models:", np.argmax(positive_models))

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(mean_danger)), mean_danger, marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Timestep")
plt.ylabel("Mean danger score")
plt.title("Mahalanobis Danger — Unseen Test Demo")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

