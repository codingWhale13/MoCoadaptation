import numpy as np

from RL.replay_simple import SimpleReplayBuffer


class EvoReplayLocalGlobalStart:
    def __init__(
        self,
        env,
        max_replay_buffer_size_population,
        max_replay_buffer_size_species,
        replace=True,
    ):
        self._init_state_buffer = SimpleReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_population,
            replace=replace,
        )
        self._population_buffer = SimpleReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_population,
            replace=replace,
        )
        self._species_buffer = SimpleReplayBuffer(
            env=env,
            max_replay_buffer_size=max_replay_buffer_size_species,
            replace=replace,
        )

        self._env = env
        self._max_replay_buffer_size_species = max_replay_buffer_size_species
        self._replace = replace

        self._mode = "species"
        self._ep_counter = 0
        self._expect_init_state = True

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        weight_preference_population,
        weight_preference_species,
    ):
        """
        Add a transition tuple.
        """
        if self._expect_init_state:
            self._init_state_buffer.add_sample(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminal=terminal,
                weight_preference=weight_preference_species,
                env_info={},
            )
            self._init_state_buffer.terminate_episode()
            self._expect_init_state = False

        self._population_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            weight_preference=weight_preference_population,
            env_info={},
        )
        self._species_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            weight_preference=weight_preference_species,
            env_info={},
        )

    def terminate_episode(self):
        """Let the replay buffer know that the episode has terminated in case
        some special book-keeping has to happen."""
        if self._mode == "species":
            self._population_buffer.terminate_episode()
            self._species_buffer.terminate_episode()
            self._ep_counter += 1
            self._expect_init_state = True

    def num_steps_can_sample(self, **kwargs):
        """Return number of unique items that can be sampled"""
        if self._mode == "population":
            return self._population_buffer.num_steps_can_sample(**kwargs)
        elif self._mode == "species":
            return self._species_buffer.num_steps_can_sample(**kwargs)

    def random_batch(self, batch_size):
        """Return a batch of size `batch_size`"""
        if self._mode == "start":
            return self._init_state_buffer.random_batch(batch_size)
        elif self._mode == "population":
            return self._population_buffer.random_batch(batch_size)
        elif self._mode == "species":
            # Return a batch of 10% population samples and 90% species samples
            pop_batch_size = int(np.ceil(batch_size * 0.1))
            species_batch_size = int(np.floor(batch_size * 0.9))
            assert pop_batch_size + species_batch_size == batch_size

            batch_pop = self._population_buffer.random_batch(pop_batch_size)
            batch = self._species_buffer.random_batch(species_batch_size)
            for key in batch.keys():
                batch[key] = np.concatenate([batch_pop[key], batch[key]], axis=0)

            return batch

    def set_mode(self, mode):
        if mode in ["start", "species", "population"]:
            self._mode = mode
        else:
            raise ValueError(f"Mode {mode} for replay buffer does not exist")

    def reset_species_buffer(self):
        self._species_buffer = SimpleReplayBuffer(
            env=self._env,
            max_replay_buffer_size=self._max_replay_buffer_size_species,
            replace=self._replace,
        )
        self._ep_counter = 0

    def get_contents(self):
        return {
            "init_state_buffer": self._init_state_buffer.get_contents(),
            "population_buffer": self._population_buffer.get_contents(),
            "species_buffer": self._species_buffer.get_contents(),
        }

    def set_contents(self, contents):
        self._init_state_buffer.set_contents(contents["init_state_buffer"])
        self._population_buffer.set_contents(contents["population_buffer"])
        self._species_buffer.set_contents(contents["species_buffer"])
