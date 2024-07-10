import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
import pyswarms as ps

from .design_optimization import DesignOptimization


class PSOBatch(DesignOptimization):
    def __init__(self, config, replay, env):
        self._env = env
        self._replay = replay
        self._state_batch_size = config["state_batch_size"]
        self._condition_on_preference = config["condition_on_preference"]

    def optimize_design(self, design, q_network, policy_network, verbose=False):
        self._replay.set_mode("start")
        initial_batch = self._replay.random_batch(self._state_batch_size)
        prefs = ptu.from_numpy(initial_batch["weight_preferences"])

        design_idxs = self._env.get_design_dimensions()

        def f_qval(x_input, **kwargs):
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            with torch.no_grad():
                for i in range(shape[0]):
                    x = x_input[i : i + 1, :]

                    state_batch = initial_batch["observations"].copy()
                    state_batch[:, design_idxs] = x
                    state_torch = ptu.from_numpy(state_batch)

                    pref_arg = [prefs] if self._condition_on_preference else []
                    (action, _, _, _, _, _, _, _) = policy_network(
                        state_torch, *pref_arg, deterministic=True
                    )
                    output = q_network(state_torch, action, *pref_arg)

                    if output.shape[1] != 1:  # TODO: make option more explicit
                        loss = -torch.sum(output * prefs, axis=1).mean()
                    else:
                        loss = -output.mean()
                    fval = float(loss.item())
                    cost[i] = fval

            return cost

        lower_bounds = np.array([l for l, _ in self._env.design_params_bounds])
        upper_bounds = np.array([u for _, u in self._env.design_params_bounds])
        bounds = (lower_bounds, upper_bounds)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=700, dimensions=len(design), bounds=bounds, options=options
        )

        # Perform optimization
        cost, new_design = optimizer.optimize(
            f_qval, print_step=100, iters=250, verbose=verbose
        )  # , n_processes=2)

        return new_design
