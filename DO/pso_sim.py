import numpy as np
import pyswarms as ps

from .design_optimization import DesignOptimization


class PSOSimulation(DesignOptimization):
    def __init__(self, config, replay, env):
        self._config = config
        self._replay = replay
        self._env = env

        self._episode_length = self._config["steps_per_episodes"]
        # TODO Make this independent of rl_algo config
        self._reward_scale = self._config["rl_algorithm_config"]["algo_params"][
            "reward_scale"
        ]

    def optimize_design(
        self, design, q_network, policy_network, weights
    ):  # weights needed for MORL
        # Important: We reset the design of the environment. Previous design
        #   will be lost

        # temporarily disable video recording to avoid creation of many useless folders
        video_recording_was_enabled = self._env._record_video
        self._env.disable_video_recording()

        def get_reward_for_design(design):
            self._env.set_new_design(design)
            state = self._env.reset()
            reward_episode = []  # SORL # MORL
            # reward_episode = np.empty([])
            weights_np = (
                weights.cpu().numpy()
            )  # Convert from tensor to numpy array #MORL
            done = False
            nmbr_of_steps = 0
            while not (done) and nmbr_of_steps <= self._episode_length:
                nmbr_of_steps += 1
                action, _ = policy_network.get_action(state, deterministic=True)
                new_state, reward, done, info = self._env.step(action)
                reward = reward * self._reward_scale
                reward = reward.reshape(1, 2)
                # reward_episode.append(float(reward)) # SORL original
                reward = np.matmul(
                    reward, weights_np
                )  # .reshape(1,) # Scalarization of reward per iteration
                reward_episode.append(reward)  # Changed for MORL - list
                # np.column_stack((reward_episode, reward))# MORL - as numpy array
                state = new_state

            # print(reward_episode.shape)
            # print(reward_episode)

            # reward_mean = np.matmul(reward_episode, weights.numpy) # scalarisation # I assume that the reward_episode is NDarray since .step(action) from evoenvsMO.py is returning NDarray as reward
            reward_mean = np.mean(reward_episode)  # SORL
            # print(reward_mean)
            # reward_mean = np.mean(reward_mean) # SORL Need to be changed still?
            return reward_mean

        def f_qval(x_input, **kwargs):
            shape = x_input.shape
            cost = np.zeros((shape[0],))
            for i in range(shape[0]):
                x = x_input[i, :]
                reward = get_reward_for_design(x)  # Same here -> changes for the MORL?
                cost[i] = -reward

            return cost

        lower_bounds = np.array([l for l, _ in self._env.design_params_bounds])
        upper_bounds = np.array([u for _, u in self._env.design_params_bounds])
        bounds = (lower_bounds, upper_bounds)
        options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
        optimizer = ps.single.GlobalBestPSO(
            n_particles=35, dimensions=len(design), bounds=bounds, options=options
        )

        # Perform optimization
        cost, new_design = optimizer.optimize(
            f_qval, print_step=100, iters=30, verbose=3
        )  # , n_processes=2)

        if video_recording_was_enabled:
            self._env.enable_video_recording()

        return new_design
