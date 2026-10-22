import glob
import h5py
import pickle
import numpy as np


GOOD_DIR = "demos/MazeDrive-v2/good"
BAD_DIR = "demos/MazeDrive-v2/bad"

N_MODELS = 100
FEATURES_PER_MODEL = 6
REG = 1e-4
SEED = 42

OUTPUT_FILE = "gaussian_danger_models.pkl"


def load_demo_obs(path):
    with h5py.File(path, "r") as f:
        obs = f["traj_0/obs"][:]

    return obs.astype(np.float64)



def fit_gaussian(X):
    mean = np.mean(X, axis=0)

    covariance = np.cov(X, rowvar=False,)

    # Regularization for numerical stability
    covariance += REG * np.eye(X.shape[1])

    # Pseudo-inverse is safer than inverse
    inv_covariance = np.linalg.pinv(covariance)

    return {
        "mean": mean,
        "covariance": covariance,
        "inv_covariance": inv_covariance,
    }


good_files = sorted(glob.glob(f"{GOOD_DIR}/*.h5"))

bad_files = sorted(glob.glob(f"{BAD_DIR}/*.h5"))

print("Good demos:", len(good_files))
print("Bad demos: ", len(bad_files))

good_obs_list = []
for path in good_files:

    obs = load_demo_obs(path)

    print("Good:", path, "->", obs.shape)
    
    good_obs_list.append(obs)



bad_obs_list = []
for path in bad_files:

    obs = load_demo_obs(path)

    print("Bad: ", path, "->", obs.shape)
    
    bad_obs_list.append(obs)



good_obs = np.concatenate(good_obs_list, axis=0)

bad_obs = np.concatenate(bad_obs_list, axis=0)

print()
print("Combined good observations:", good_obs.shape)
print("Combined bad observations: ", bad_obs.shape)


obs_dim = good_obs.shape[1]
print("Obs dim:", obs_dim)


rng = np.random.default_rng(SEED)

models = []

for i in range(N_MODELS):

    # Randomly select observation dimensions
    feature_indices = rng.choice(obs_dim, size=FEATURES_PER_MODEL, replace=False)

    feature_indices = np.sort(feature_indices)

    # Select those features from ALL demos
    good_X = good_obs[:, feature_indices]
    bad_X = bad_obs[:, feature_indices]

    # Fit Gaussian to all good/bad observations
    good_gaussian = fit_gaussian(good_X)
    bad_gaussian = fit_gaussian(bad_X)

    models.append({
        "feature_indices": feature_indices,
        "good": good_gaussian,
        "bad": bad_gaussian,
    })



with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(models, f)


print()
print(f"Created {len(models)} Gaussian pairs.")
print(
    f"Each Gaussian uses "
    f"{FEATURES_PER_MODEL} dimensions."
)
print(f"Saved to: {OUTPUT_FILE}")



# WALL_START = 34

# wall_models = 0

# for i, model in enumerate(models):

#     indices = model["feature_indices"]

#     n_wall = np.sum(
#         indices >= WALL_START
#     )

#     if n_wall > 0:
#         wall_models += 1

#     print(
#         i,
#         "features:",
#         indices,
#         "wall features:",
#         n_wall,
#     )


# print()
# print(
#     "Models containing wall features:",
#     wall_models,
# )

# print(
#     "Total models:",
#     len(models),
# )