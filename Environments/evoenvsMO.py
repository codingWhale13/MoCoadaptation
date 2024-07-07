from copy import copy
import os

from gym import spaces
import numpy as np
import wandb

from .pybullet_evo.gym_locomotion_envs import HalfCheetahMoBulletEnv
from utils import BestEpisodesVideoRecorder


class HalfCheetahEnvMO:
    def __init__(
        self,
        config={"env": {"render": True, "record_video": False}},
        reward_scaling_energy=1,
        use_wandb=False,
    ):
        self._config = config
        self._record_video = config["env"]["record_video"]
        self._reward_scaling_energy = reward_scaling_energy
        self._current_design = [1.0] * 6
        self._config_numpy = np.array(self._current_design)
        self.design_params_bounds = [(0.8, 2.0)] * 6
        self._env = HalfCheetahMoBulletEnv(
            render=config["env"]["render"], design=self._current_design
        )
        self.reward_dim = self._env.reward_dim
        self.init_sim_params = [
            [1.0] * 6,
            [1.41, 0.96, 1.97, 1.73, 1.97, 1.17],
            [1.52, 1.07, 1.11, 1.97, 1.51, 0.99],
            [1.08, 1.18, 1.39, 1.76, 1.85, 0.92],
            [0.85, 1.54, 0.97, 1.38, 1.10, 1.49],
        ]
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=[self._env.observation_space.shape[0] + 6],
            dtype=np.float32,
        )  # env.observation_space
        self.action_space = self._env.action_space
        self._initial_state = self._env.reset()

        if self._record_video:
            if config["video"]["video_save_dir"] is not None:
                save_dir = config["video"]["video_save_dir"]
            else:
                save_dir = os.path.join(config["run_folder"], "videos")

            kwargs = dict()
            for key in ["record_evy_n_episodes", "max_videos"]:
                if key in config["video"]:
                    kwargs[key] = config["video"][key]

            self._video_recorder = BestEpisodesVideoRecorder(path=save_dir, **kwargs)

        # Which dimensions in the state vector are design parameters?
        self._design_dims = list(
            range(
                self.observation_space.shape[0] - len(self._current_design),
                self.observation_space.shape[0],
            )
        )
        assert len(self._design_dims) == 6

        self._use_wandb = use_wandb

    def render(self):
        pass

    def step(self, a):
        info = {}
        state, reward, done, _ = self._env.step(a)
        state = np.append(state, self._config_numpy)
        info["orig_action_cost"] = 0.1 * np.mean(np.square(a))
        info["orig_reward"] = reward
        reward[1] *= self._reward_scaling_energy

        if self._record_video:
            self._video_recorder.step(
                env=self._env, state=state, reward=reward, done=done
            )

        return state, reward, False, info

    def reset(self):
        state = self._env.reset()
        self._initial_state = state
        state = np.append(state, self._config_numpy)

        if self._record_video:
            self._video_recorder.reset(env=self._env, state=state, reward=0, done=False)

        return state

    def set_new_design(self, vec):
        self._env.reset_design(vec)
        self._current_design = vec
        self._config_numpy = np.array(vec)

        if self._record_video:
            self._video_recorder.increase_folder_counter()

    def get_random_design(self):
        optimized_params = np.random.uniform(low=0.8, high=2.0, size=6)
        return optimized_params

    def get_current_design(self):
        return copy(self._current_design)

    def get_design_dimensions(self):
        return copy(self._design_dims)

    def disable_video_recording(self):
        self._record_video = False

    def enable_video_recording(self):
        self._record_video = True
