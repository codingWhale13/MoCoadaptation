from gym.spaces import Box, Discrete, Tuple
import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


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


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        env,
        env_info_sizes=None,
        replace=True,
        reward_dim=2,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, "info_sizes"):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        observation_dim = get_dim(self._ob_space)

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=observation_dim,
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            replace=replace,
            reward_dim=reward_dim,
        )

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        weight_preference,
        **kwargs
    ):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            weight_preference=weight_preference,
            **kwargs
        )
