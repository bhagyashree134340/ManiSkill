import pickle
import numpy as np
import gymnasium as gym


class GaussianDangerCostWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        model_file="/Users/bhagyashreerane/Projects/melon-rl/gaussian_danger_models.pkl",
        cost_dim=1,
    ):
        super().__init__(env)

        with open(model_file, "rb") as f:
            self.models = pickle.load(f)

        self.cost_dim = cost_dim

        print(
            f"Loaded {len(self.models)} Gaussian danger models."
        )

    @staticmethod
    def mahalanobis_squared(x, gaussian):
        diff = x - gaussian["mean"]

        return float(
            diff @ gaussian["inv_covariance"] @ diff
        )

    def compute_cost(self, obs):
        obs = np.asarray(
            obs,
            dtype=np.float64
        ).reshape(-1)

        model_costs = []
        raw_dangers = []

        for model in self.models:

            feature_indices = model["feature_indices"]

            x = obs[feature_indices]

            good = model["good"]
            bad = model["bad"]

            d_good = self.mahalanobis_squared(
                x,
                good
            )

            d_bad = self.mahalanobis_squared(
                x,
                bad
            )

            danger = d_good - d_bad

            

            raw_dangers.append(danger)

            # Convert each model's danger to [0, 1]
            model_cost = 1.0 / (
                1.0 + np.exp(
                    -np.clip(danger, -50, 50)
                )
            )

            # print(
            #     "cost =", model_cost,
            #     "d_good =", d_good,
            #     "d_bad =", d_bad,
            #     "danger =", danger,
            # )

            model_costs.append(model_cost)

        raw_dangers = np.asarray(raw_dangers)
        model_costs = np.asarray(model_costs)

        cost = np.mean(model_costs)

        # # --------------------------------------------------
        # # DEBUG
        # # --------------------------------------------------

        # print("\n========== GAUSSIAN DANGER DEBUG ==========")

        # print(
        #     f"Raw danger: "
        #     f"min={raw_dangers.min():.4f}, "
        #     f"mean={raw_dangers.mean():.4f}, "
        #     f"median={np.median(raw_dangers):.4f}, "
        #     f"max={raw_dangers.max():.4f}"
        # )

        # print(
        #     f"Model cost: "
        #     f"min={model_costs.min():.4f}, "
        #     f"mean={model_costs.mean():.4f}, "
        #     f"median={np.median(model_costs):.4f}, "
        #     f"max={model_costs.max():.4f}"
        # )

        # print(
        #     f"Models with cost < 0.1: "
        #     f"{np.sum(model_costs < 0.1)}/100"
        # )

        # print(
        #     f"Models with cost > 0.9: "
        #     f"{np.sum(model_costs > 0.9)}/100"
        # )

        # print(
        #     f"FINAL COST: {cost:.6f}"
        # )

        # print("============================================\n")

        return float(cost)


    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(
            action
        )

        cost = self.compute_cost(obs)

        # This is what the SafeRL code can use
        info = dict(info)
        info["cost"] = np.array([cost], dtype=np.float32)

        print(
            f"cost={cost:.3f} "
            f"collision={info.get('collision', False)}"
        )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )