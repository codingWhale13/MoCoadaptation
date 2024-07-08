import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer


class EvoReplayLocalGlobalStart(ReplayBuffer):
    def __init__(
        self,
        env,
        max_replay_buffer_size_species,
        max_replay_buffer_size_population,
        condition_on_preference=False,
        verbose=False,
    ):
        self._species_buffer = EnvReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_species,
            condition_on_preference=condition_on_preference,
        )
        self._population_buffer = EnvReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_population,
            condition_on_preference=condition_on_preference,
        )
        self._init_state_buffer = EnvReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_population,
            condition_on_preference=condition_on_preference,
        )
        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._mode = "species"
        self._ep_counter = 0
        self._expect_init_state = True
        self._condition_on_preference = condition_on_preference

        if verbose:
            print("Using EvoReplayLocalGlobalStart as replay buffer")

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        weight_preference_species=None,
        weight_preference_population=None,
        **kwargs,
    ):
        """
        Add a transition tuple.
        """
        self._species_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            weight_preference=weight_preference_species,
            terminal=terminal,
            env_info={},
            **kwargs,
        )

        self._population_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            weight_preference=weight_preference_population,
            terminal=terminal,
            env_info={},
            **kwargs,
        )

        if self._expect_init_state:
            self._init_state_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                weight_preference=weight_preference_species,  # TODO: should this also be pop?
                terminal=terminal,
                env_info={},
                **kwargs,
            )
            self._init_state_buffer.terminate_episode()
            self._expect_init_state = False

    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        if self._mode == "species":
            self._species_buffer.terminate_episode()
            self._population_buffer.terminate_episode()
            self._ep_counter += 1
            self._expect_init_state = True

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        if self._mode == "species":
            return self._species_buffer.num_steps_can_sample(**kwargs)
        elif self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        if self._mode == "species":
            # return self._species_buffer.random_batch(batch_size)
            species_batch_size = int(np.floor(batch_size * 0.9))
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            pop = self._population_buffer.random_batch(pop_batch_size)
            spec = self._species_buffer.random_batch(species_batch_size)
            for key, item in pop.items():
                pop[key] = np.concatenate([pop[key], spec[key]], axis=0)
            return pop
        elif self._mode == "population":
            return self._population_buffer.random_batch(batch_size)
        elif self._mode == "start":
            return self._init_state_buffer.random_batch(batch_size)

    def set_mode(self, mode):
        if mode in ["species", "population", "start"]:
            self._mode = mode
        else:
            raise ValueError(f"Mode {mode} for replay buffer does not exist")

    def reset_species_buffer(self):
        self._species_buffer = EnvReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species,
            condition_on_preference=self._condition_on_preference,
        )
        self._ep_counter = 0

    def get_contents(self):
        contents = {
            "species_buffer": self._species_buffer.get_contents(),
            "population_buffer": self._population_buffer.get_contents(),
            "init_state_buffer": self._init_state_buffer.get_contents(),
        }

        return contents

    def set_contents(self, contents):
        self._species_buffer.set_contents(contents["species_buffer"])
        self._population_buffer.set_contents(contents["population_buffer"])
        self._init_state_buffer.set_contents(contents["init_state_buffer"])
