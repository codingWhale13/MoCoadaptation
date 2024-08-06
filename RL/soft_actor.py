import numpy as np
import torch

from RL.replay_mix import MixedEvoReplayLocalGlobalStart
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit
import rlkit.torch.pytorch_util as ptu
import utils


class SoftActorCritic:
    def __init__(
        self,
        config,
        env,
        replay: MixedEvoReplayLocalGlobalStart,
        networks,
        use_gpu=False,
        wandb_instance=None,
    ):
        """Bascally a wrapper class for SAC from rlkit.

        Args:
            config: Configuration dictonary
            env: Environment
            replay: Replay buffer
            networks: dict containing two sub-dicts, 'individual' and 'population'
                which contain the networks.
        """
        self._env = env
        self._replay = replay
        self._networks = networks
        self._wandb_instance = wandb_instance
        self._use_gpu = use_gpu

        self._use_vector_q = config["use_vector_q"]
        self._condition_on_preference = config["condition_on_preference"]
        self._batch_size = config["rl_algorithm_config"]["batch_size"]
        self._number_pop_updates = config["rl_algorithm_config"]["pop_updates"]
        self._number_indiv_updates = config["rl_algorithm_config"]["indiv_updates"]
        self._copy_from_global = config["rl_algorithm_config"]["copy_from_gobal"]

        self._ind_qf1 = networks["individual"]["qf1"]
        self._ind_qf2 = networks["individual"]["qf2"]
        self._ind_qf1_target = networks["individual"]["qf1_target"]
        self._ind_qf2_target = networks["individual"]["qf2_target"]
        self._ind_policy = networks["individual"]["policy"]
        self._ind_kwargs = config["rl_algorithm_config"]["algo_params"]

        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            condition_on_preference=self._condition_on_preference,
            use_vector_q=self._use_vector_q,
            use_automatic_entropy_tuning=False,
            use_gpu=self._use_gpu,
            wandb_instance=self._wandb_instance,
            **self._ind_kwargs,
        )
        self._algorithm_pop = SoftActorCritic_rlkit(
            env=self._env,
            policy=networks["population"]["policy"],
            qf1=networks["population"]["qf1"],
            qf2=networks["population"]["qf2"],
            target_qf1=networks["population"]["qf1_target"],
            target_qf2=networks["population"]["qf2_target"],
            condition_on_preference=self._condition_on_preference,
            use_vector_q=self._use_vector_q,
            use_automatic_entropy_tuning=False,
            use_gpu=self._use_gpu,
            wandb_instance=self._wandb_instance,
            **config["rl_algorithm_config"]["algo_params_pop"],
        )

    def episode_init(self):
        """Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
        self._algorithm_ind = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
            condition_on_preference=self._condition_on_preference,
            use_vector_q=self._use_vector_q,
            use_automatic_entropy_tuning=False,
            use_gpu=self._use_gpu,
            wandb_instance=self._wandb_instance,
            **self._ind_kwargs,
        )
        if self._copy_from_global:
            utils.copy_pop_to_ind(
                networks_pop=self._networks["population"],
                networks_ind=self._networks["individual"],
            )

    def single_train_step(self, old_replay_portion=0, train_ind=True, train_pop=False):
        """A single trianing step.

        Args:
            train_ind: Boolean. If true, the individual networks will be trained.
            train_pop: Boolean. If true, the population networks will be trained.
        """
        if train_ind:
            self._replay.set_mode("species")
            for _ in range(self._number_indiv_updates):
                batch = self._replay.random_batch(self._batch_size, old_replay_portion)
                self._algorithm_ind.train(batch, scalarize_before_q_loss=False)

        if train_pop:
            self._replay.set_mode("population")
            for _ in range(self._number_pop_updates):
                batch = self._replay.random_batch(self._batch_size, old_replay_portion)
                self._algorithm_pop.train(batch, scalarize_before_q_loss=False)

    @staticmethod
    def create_networks(env, config):
        """Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        Args:
            config: A configuration dictonary containing population and
                individual networks
        Returns:
            A dictonary which contains the networks.
        """
        return {
            "individual": SoftActorCritic._create_networks(env=env, config=config),
            "population": SoftActorCritic._create_networks(env=env, config=config),
        }

    @staticmethod
    def _create_networks(env, config):
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        net_size = config["rl_algorithm_config"]["net_size"]
        hidden_sizes = [net_size] * config["rl_algorithm_config"]["network_depth"]

        q_input_size = obs_dim + action_dim
        policy_input_size = obs_dim
        if config["condition_on_preference"]:
            q_input_size += env.reward_dim
            policy_input_size += env.reward_dim
        q_output_size = env.reward_dim if config["use_vector_q"] else 1

        qf1 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=q_input_size,
            output_size=q_output_size,
        ).to(device=ptu.device)
        qf2 = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=q_input_size,
            output_size=q_output_size,
        ).to(device=ptu.device)
        qf1_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=q_input_size,
            output_size=q_output_size,
        ).to(device=ptu.device)
        qf2_target = FlattenMlp(
            hidden_sizes=hidden_sizes,
            input_size=q_input_size,
            output_size=q_output_size,
        ).to(device=ptu.device)
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            input_size=policy_input_size,
            output_size=action_dim,
        ).to(device=ptu.device)

        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {
            "qf1": qf1,
            "qf2": qf2,
            "qf1_target": qf1_target,
            "qf2_target": qf2_target,
            "policy": policy,
        }

    @staticmethod
    def get_q_network(networks):
        return networks["qf1"]

    @staticmethod
    def get_policy_network(networks):
        return networks["policy"]
