import numpy as np

from RL.replay_evo import EvoReplayLocalGlobalStart


def assert_which(which: str):
    """Asserts that `which` is a valid option"""
    if which not in ["old", "new", "both"]:
        raise ValueError('Argument for `which` should be "old", "new", or "both"')


class MixedEvoReplayLocalGlobalStart:
    """This class handles two replay buffers under the hood but appears as one.

    This allows for loading a previous ("old") replay buffer and
    feeding new samples to the new replay buffer. The option `which`
    is biased to this behaviour, see default options of the methods.
    """

    def __init__(
        self,
        env,
        max_replay_buffer_size_population,
        max_replay_buffer_size_species,
        replace=True,
    ):
        self._replay_old = EvoReplayLocalGlobalStart(
            env,
            max_replay_buffer_size_population=max_replay_buffer_size_population,
            max_replay_buffer_size_species=max_replay_buffer_size_species,
            replace=replace,
        )
        self._replay_new = EvoReplayLocalGlobalStart(
            env,
            max_replay_buffer_size_population=max_replay_buffer_size_population,
            max_replay_buffer_size_species=max_replay_buffer_size_species,
            replace=replace,
        )

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
        """Add a sample to the new replay buffer."""
        self._replay_new.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            weight_preference_population=weight_preference_population,
            weight_preference_species=weight_preference_species,
        )

    def terminate_episode(self, which="new"):
        assert_which(which)
        if which in ["old", "both"]:
            self._replay_old.terminate_episode()
        if which == ["new", "both"]:
            self._replay_new.terminate_episode()

    def num_steps_can_sample(self, which="new", **kwargs):
        assert_which(which)
        if which == "old":
            return self._replay_old.num_steps_can_sample(*kwargs)
        elif which == "new":
            return self._replay_new.num_steps_can_sample(**kwargs)
        else:
            raise ValueError('Please choose "old" or "new" for `which`')

    def random_batch(self, batch_size, old_replay_portion=0):
        """Return batch of samples mixed from old and new replay buffer"""
        batch_size_old = int(np.ceil(batch_size * old_replay_portion))
        batch_size_new = int(np.floor(batch_size * (1 - old_replay_portion)))
        assert batch_size_old + batch_size_new == batch_size

        batch = self._replay_old.random_batch(batch_size_old)
        batch_new = self._replay_new.random_batch(batch_size_new)
        for key in batch.keys():
            batch[key] = np.concatenate([batch[key], batch_new[key]], axis=0)

        return batch

    def set_mode(self, mode, which="both"):
        assert_which(which)
        if which in ["old", "both"]:
            self._replay_old.set_mode(mode)
        if which == ["new", "both"]:
            self._replay_new.set_mode(mode)

    def reset_species_buffer(self, which="new"):
        assert_which(which)
        if which in ["old", "both"]:
            self._replay_old.reset_species_buffer()
        if which == ["new", "both"]:
            self._replay_new.reset_species_buffer()

    def get_contents(self, which="new"):
        assert_which(which)
        if which == "old":
            return self._replay_old.get_contents()
        elif which == "new":
            return self._replay_new.get_contents()
        else:
            raise ValueError('Please choose "old" or "new" for `which`')

    def set_contents(self, contents, which="old"):
        assert_which(which)
        if which in ["old", "both"]:
            self._replay_old.set_contents(contents)
        if which == ["new", "both"]:
            self._replay_new.set_contents(contents)
