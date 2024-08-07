import csv
import os
import random
import time
from typing import Union

import numpy as np
import torch
import wandb

from DO.pso_batch import PSOBatch
from DO.pso_sim import PSOSimulation
from environments.evoenvsMO import HalfCheetahEnvMO, HopperEnvMO, Walker2dEnvMO
from RL.replay_mix import MixedEvoReplayLocalGlobalStart
from RL.soft_actor import SoftActorCritic
import rlkit.torch.pytorch_util as ptu
from utils import (
    DO_CHECKPOINTS,
    RL_CHECKPOINTS,
    SimpleVideoRecorder,
    copy_network,
    get_cycle_count,
    move_to_cpu,
    move_to_cuda,
    save_csv,
)


def select_env(env_name):
    if env_name == "HalfCheetah":
        return HalfCheetahEnvMO
    elif env_name == "Walker2d":
        return Walker2dEnvMO
    elif env_name == "Hopper":
        return HopperEnvMO

    raise ValueError(f"Environment name '{env_name}' not found")


def select_rl_algo(algo_name):
    if algo_name == "SoftActorCritic":
        return SoftActorCritic

    raise ValueError(f"RL method '{algo_name}' not found")


def get_modified_preference(original_preference):
    preference = original_preference.copy()

    preference[0] += np.random.normal(scale=0.1)
    preference[1] += np.random.normal(scale=0.1)
    preference = np.clip(preference, 0, 1)

    sum = preference.sum()
    if sum > 0:
        return preference / sum
    else:  # edge case, to avoid division by 0
        return original_preference


def select_design_opt_algo(algo_name):
    if algo_name == "pso_batch":
        return PSOBatch
    elif algo_name == "pso_sim":
        return PSOSimulation

    raise ValueError(f"Design optimization method '{algo_name}' not found")


class Coadaptation:
    def __init__(self, config, design_iter_to_load: Union[str, int, None] = None):
        self._verbose = config["verbose"]
        self._use_wandb = config["use_wandb"]
        self._run_folder = config["run_folder"]
        self._initial_model_dir = config["initial_model_dir"]
        self._save_networks = config["save_networks"]
        self._save_replay_buffer = config["save_replay_buffer"]
        self._condition_on_preference = config["condition_on_preference"]

        self._iterations_init = config["iterations_init"]
        self._design_cycles = config["design_cycles"]
        self._initial_episodes = config["initial_episodes"]
        self._iterations = config["iterations"]
        self._episode_length = config["steps_per_episodes"]

        self._use_gpu = config["use_gpu"]
        self._cuda_device = config["cuda_device"]

        self._weight_pref = np.array(config["weight_preference"])
        self._old_replay_portion = config["old_replay_portion"]

        # Start wandb experiment
        if self._use_wandb:
            wandb.init(
                project=config["project_name"],
                name=config["run_name"],
                dir="/scratch/work/kielen1",
                # Avoid large debug-internal.log files (see https://github.com/wandb/wandb/issues/4223)
                settings=wandb.Settings(
                    log_internal="/scratch/work/kielen1/wandb/null",
                ),
            )

        # Set device to GPU or CPU
        if self._use_gpu:
            move_to_cuda(self._cuda_device)
        else:
            move_to_cpu()

        # Initialize env
        env_cls = select_env(config["env"]["env_name"])
        # energy reward is approximately 1.725 larger than running reward
        if config["env"]["env_name"] == "HalfCheetah":
            self.env = env_cls(config=config, reward_scaling_energy=0.27532)
        else:
            self.env = env_cls(config=config)
        self.reward_dim = self.env.reward_dim
        self.reward_names = self.env.reward_names

        # Initialize replay buffer (used by both RL algo and DO algo)
        self._replay = MixedEvoReplayLocalGlobalStart(
            self.env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
        )

        # Initialize RL algo
        self._rl_algo_class = select_rl_algo(config["rl_method"])
        self._networks = self._rl_algo_class.create_networks(self.env, config=config)
        self._rl_algo = self._rl_algo_class(
            config=config,
            env=self.env,
            replay=self._replay,
            networks=self._networks,
            wandb_instance=wandb.run if self._use_wandb else None,
            use_gpu=self._use_gpu,
        )

        # Initialize DO algo
        do_algo_class = select_design_opt_algo(config["design_optim_method"])
        self._do_algo = do_algo_class(config=config, replay=self._replay, env=self.env)

        # Initialize counters
        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0

        # Load model from checkpoint if available
        if self._initial_model_dir is None:
            self._data_design_type = "Initial"
        else:
            self._data_design_type = "Pre-trained"
            self.load_checkpoint(self._initial_model_dir, design_iter_to_load)

    def load_checkpoint(self, old_run_dir, design_iter=None):
        # 1. Load design checkpoint
        num_cycles = get_cycle_count(old_run_dir)
        if design_iter is None or design_iter == "last":
            design_iter = num_cycles
        else:
            assert isinstance(design_iter, int), '`design_iter` not int, None or "last"'
            assert design_iter <= num_cycles, f"Design cycle {design_iter} not found"
        file = f"data_design_{num_cycles}.csv"

        rows = []
        with open(os.path.join(old_run_dir, DO_CHECKPOINTS, file), newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)

        link_lengths = np.array(rows[1], dtype=float)
        self.env.set_new_design(link_lengths)

        # 2. Load RL checkpoint
        rl_dir = os.path.join(old_run_dir, RL_CHECKPOINTS)
        if design_iter is None:
            file = f"networks_for_design_{num_cycles}.chk"
        else:
            file = f"networks_for_design_{design_iter}.chk"

        network_path = os.path.join(rl_dir, file)
        network_data = torch.load(network_path, map_location=ptu.device)
        for key, net in self._networks["population"].items():
            params = network_data["population"][key]
            net.load_state_dict(params)
        for key, net in self._networks["individual"].items():
            params = network_data["individual"][key]
            net.load_state_dict(params)

        # 3. Load replay buffer if requried and available
        if self._old_replay_portion > 0:
            replay_path = os.path.join(rl_dir, "latest_replay_buffer_0.chk")
            if os.path.isfile(replay_path):
                replay_buffer = torch.load(replay_path)
                self._replay.set_contents(replay_buffer)
            else:
                raise ValueError(
                    f"Can't satisfy `old_replay_portion`={self._old_replay_portion}>0"
                    "because previous replay buffer could not be loaded"
                )

    def initialize_episode(self):
        """Initializations required before the first episode.

        Should be called before the first episode of a new design is
        executed. Resets variables such as _rewards for logging purposes etc.
        """
        self._rl_algo.episode_init()
        self._replay.reset_species_buffer()

        self.states = []
        self.actions = []
        self.rewards = []  # Will be a list of tuples in MORL setting
        self._episode_counter = 0

    def _prepare_single_episode(self):
        # Set self._policy
        network_name = "individual"
        if self._episode_counter < self._initial_episodes:
            network_name = "population"
        self._policy = self._rl_algo_class.get_policy_network(
            self._networks[network_name]
        )

        if not self._use_gpu:
            self._policy = copy_network(
                network_to=self._policy,
                network_from=self._policy,
                cuda_device=self._cuda_device,
                force_cpu=True,
            )

        # Reset environment and return initial state
        state = self.env.reset()

        return state

    def _collect_training_experience(self):
        """Collect training data.

        This function executes a single episode in the environment using the policy.
        The state-action-reward-nextState data is stored in the replay buffer.
        """
        state = self._prepare_single_episode()

        step_count = 0
        done = False
        while not done and step_count < self._episode_length:
            step_count += 1
            pref = self._weight_pref if self._condition_on_preference else None
            action, _ = self._policy.get_action(state, pref, determinstic=False)
            new_state, reward, done, _ = self.env.step(action)

            if bool(random.getrandbits(1)):
                # with probability of 50%, modify the weight preference for the population buffer
                weight_pref_pop = get_modified_preference(self._weight_pref)
            else:
                weight_pref_pop = self._weight_pref

            self._replay.add_sample(
                observation=state,
                action=action,
                reward=reward,
                next_observation=new_state,
                terminal=np.array([done]),
                weight_preference_species=self._weight_pref,
                weight_preference_population=weight_pref_pop,
            )
            state = new_state

        self._replay.terminate_episode()

    def execute_policy(self):
        """Evaluates the current deterministic policy.

        Evaluates the current policy in the environment by unrolling a single
        episode in the environment. The achieved cumulative reward is logged.
        """
        state = self._prepare_single_episode()

        ep_states = np.empty((0, state.shape[-1]))
        ep_actions = np.empty((0, self.env.action_dim))
        ep_rewards = np.zeros(self.env.reward_dim)
        step_count = 0
        done = False
        while not done and step_count < self._episode_length:
            step_count += 1

            pref = self._weight_pref if self._condition_on_preference else None
            action, _ = self._policy.get_action(state, pref, deterministic=True)

            new_state, reward, done, _ = self.env.step(action)

            ep_states = np.vstack((ep_states, state))
            ep_actions = np.vstack((ep_actions, action))
            ep_rewards = np.add(ep_rewards, reward, casting="unsafe")

            state = new_state

        self.rewards.append(ep_rewards)
        self.states.append(ep_states)
        self.actions.append(ep_actions)

        if self._use_wandb:
            # Save episodic reward
            wandb.log({n: ep_rewards[i] for i, n in enumerate(self.reward_names)})

    def create_video_of_episode(self, save_dir, filename):
        self.initialize_episode()
        video_recorder = SimpleVideoRecorder(self.env._env, save_dir, filename)

        state = self._prepare_single_episode()

        step_count = 0
        done = False
        while not done and step_count < self._episode_length:
            pref = self._weight_pref if self._condition_on_preference else None
            action, _ = self._policy.get_action(state, pref, deterministic=True)

            new_state, _, done, _ = self.env.step(action)
            video_recorder.step()

            state = new_state
            step_count += 1

        video_recorder.save_video()

    def _save_do_checkpoint(self):
        """Saves the logged data to disk as a CSV file.

        Each design is saved as an individual log-file in the DO_CHECKPOINTS folder.
        """
        save_csv(
            save_dir=os.path.join(self._run_folder, DO_CHECKPOINTS),
            filename=f"data_design_{self._design_counter}.csv",
            design_type=self._data_design_type,
            design_parameters=list(self.env.get_current_design()),
            rewards={
                self.reward_names[i]: [reward[i] for reward in self.rewards]
                for i in range(self.reward_dim)
            },
        )

    def _save_rl_checkpoint(self):
        """Saves the networks and replay buffer to disk if specified in config."""
        save_dir = os.path.join(self._run_folder, RL_CHECKPOINTS)
        os.makedirs(save_dir, exist_ok=True)

        if self._save_networks:
            # create new checkpoint for model parameters
            network_checkpoint = dict()

            checkpoints_pop = {}
            for key, net in self._networks["population"].items():
                checkpoints_pop[key] = net.state_dict()
            network_checkpoint["population"] = checkpoints_pop

            checkpoints_ind = {}
            for key, net in self._networks["individual"].items():
                checkpoints_ind[key] = net.state_dict()
            network_checkpoint["individual"] = checkpoints_ind

            filename = f"networks_for_design_{self._design_counter}.chk"
            torch.save(network_checkpoint, os.path.join(save_dir, filename))

        if self._save_replay_buffer:
            # overwrite last checkpoint for replay buffer
            # (saving it for each design would consume a large amount of memory)
            replay_buffer_checkpoint = self._replay.get_contents()
            filename = f"latest_replay_buffer_0.chk"  # "_0" is for convenient loading
            torch.save(replay_buffer_checkpoint, os.path.join(save_dir, filename))

    def _single_rl_iteration(self, train_pop=True):
        """A single iteration.

        Makes all necessary function calls for a single iterations such as:
            - Collecting training data
            - Executing a training step
            - Evaluate the current policy
            - Log data
        """
        if self._verbose:
            print(
                f"Time for one iteration: {time.time() - self._last_single_iteration_time}"
            )
        self._last_single_iteration_time = time.time()
        self._replay.set_mode("species")
        self._collect_training_experience()

        if self._episode_counter >= self._initial_episodes:
            self._rl_algo.single_train_step(
                old_replay_portion=self._old_replay_portion,
                train_ind=True,
                train_pop=train_pop,
            )

        self._episode_counter += 1
        self.execute_policy()

    def _train_rl_policy(self, iterations, train_pop=True):
        """Run this method after a new design has been chosen."""
        # Optimize RL policy for new design
        for _ in range(iterations):
            self._single_rl_iteration(train_pop)

        self._save_do_checkpoint()  # save DO checkpoint (and rewards) in CSV
        self._save_rl_checkpoint()  # save RL checkpoint (network weights)

    def _initial_design_loop(self, iterations):
        """The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.
        """
        self._data_design_type = "Initial"

        if self._initial_model_dir is None:
            for params in self.env.init_sim_params:
                self._design_counter += 1
                self.env.set_new_design(params)
                self.initialize_episode()
                self._train_rl_policy(iterations, train_pop=False)
        else:
            self.initialize_episode()
            self._design_counter += 1
            self._train_rl_policy(iterations, train_pop=False)

    def _training_loop(self, iterations, design_cycles):
        """The training process which optimizes designs and policies.

        The function executes the reinforcement learning loop and the design
        optimization process.

        Args:
            iterations: An integer stating the number of iterations/episodes
                to be used per design during the reinforcement learning loop.
            design_cycles: Integer stating how many designs to evaluate.

        """
        self.initialize_episode()
        # TODO fix the following
        self.env._env.reset()

        self._data_design_type = "Optimized"

        optimized_params = self.env.get_random_design()
        q_network = self._rl_algo_class.get_q_network(self._networks["population"])
        policy_network = self._rl_algo_class.get_policy_network(
            self._networks["population"]
        )

        optimized_params = self._do_algo.optimize_design(
            design=optimized_params,
            q_network=q_network,
            policy_network=policy_network,
            old_replay_portion=self._old_replay_portion,
            verbose=self._verbose,
        )

        optimized_params = list(optimized_params)

        for i in range(design_cycles):
            self._design_counter += 1
            self.env.set_new_design(optimized_params)

            # Reinforcement Learning
            self._train_rl_policy(iterations, train_pop=True)

            # Design Optimization
            if i % 2 == 1:
                self._data_design_type = "Optimized"
                q_network = self._rl_algo_class.get_q_network(
                    self._networks["population"]
                )
                policy_network = self._rl_algo_class.get_policy_network(
                    self._networks["population"]
                )
                optimized_params = self._do_algo.optimize_design(
                    design=optimized_params,
                    q_network=q_network,
                    policy_network=policy_network,
                    old_replay_portion=self._old_replay_portion,
                    verbose=self._verbose,
                )
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = "Random"
                optimized_params = self.env.get_random_design()
                optimized_params = list(optimized_params)
            self.initialize_episode()

    def run(self):
        """Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
        self._initial_design_loop(self._iterations_init)
        self._training_loop(
            iterations=self._iterations,
            design_cycles=self._design_cycles,
        )
