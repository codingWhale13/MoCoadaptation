import warnings
from collections import OrderedDict

from gym.spaces import Box, Discrete, Tuple
import numpy as np


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


class SimpleReplayBuffer:
    def __init__(
        self,
        env,
        max_replay_buffer_size,
        replace=True,
        env_info_sizes=None,
    ):
        self._observation_dim = get_dim(env.observation_space)
        self._action_space = env.action_space
        self._action_dim = get_dim(self._action_space)

        self._max_replay_buffer_size = max_replay_buffer_size
        self._replace = replace
        self._top = 0
        self._size = 0

        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._observations = np.zeros((max_replay_buffer_size, self._observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, self._observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, env.reward_dim))
        # self._terminals[i] represents the terminal flag received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")
        self._weight_pref = np.zeros((max_replay_buffer_size, env.reward_dim))

        # self._env_infos[key][i] represents the return value of env_info[key] at time i
        if env_info_sizes is None:
            env_info_sizes = env.info_sizes if hasattr(env, "info_sizes") else dict()
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        weight_preference,
        env_info,
    ):
        if isinstance(self._action_space, Discrete):
            action_temp = np.zeros(self._action_dim)
            action_temp[action] = 1
            action = action_temp

        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_observation
        self._terminals[self._top] = terminal
        self._weight_pref[self._top] = weight_preference
        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(
            self._size,
            size=batch_size,
            replace=self._replace or self._size < batch_size,
        )
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay."
            )

        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_observations=self._next_obs[indices],
            terminals=self._terminals[indices],
            weight_preferences=self._weight_pref[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]

        return batch

    def get_contents(self):
        batch = dict(
            observations=self._observations,
            actions=self._actions,
            rewards=self._rewards,
            next_observations=self._next_obs,
            terminals=self._terminals,
            weight_preferences=self._weight_pref,
            top=self._top,
            size=self._size,
        )
        # NOTE: the remaining attributes (_observation_dim, _action_dim,
        # _max_replay_buffer_size, _env_infos, _replace) are not included
        # because they are only set once in __init__ and we assume these
        # values to not change between saving and loading the replay buffer.

        return batch

    def set_contents(self, contents):
        self._observations = contents["observations"]
        self._actions = contents["actions"]
        self._rewards = contents["rewards"]
        self._next_obs = contents["next_observations"]
        self._terminals = contents["terminals"]
        self._weight_pref = contents["weight_preferences"]
        self._top = contents["top"]
        self._size = contents["size"]

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {key: self._env_infos[key][indices] for key in self._env_info_keys}

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([("size", self._size)])
