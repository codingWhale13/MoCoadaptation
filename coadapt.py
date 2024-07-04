import csv
import os
import time
import utils

import numpy as np
import torch
import wandb


from DO.pso_batch import PSOBatch
from DO.pso_sim import PSOSimulation
from Environments.evoenvsMO import HalfCheetahEnvMO
from RL.evoreplay import EvoReplayLocalGlobalStart
from RL.soft_actor import SoftActorCritic
import rlkit.torch.pytorch_util as ptu


def select_env(env_name):
    if env_name == "HalfCheetah":
        return HalfCheetahEnvMO

    raise ValueError("Environment class not found.")


def select_rl_algo(rl_name):
    if rl_name == "SoftActorCritic":
        return SoftActorCritic

    raise ValueError("RL method not fund")


def select_design_opt_algo(algo_name):
    if algo_name == "pso_batch":
        return PSOBatch
    elif algo_name == "pso_sim":
        return PSOSimulation

    raise ValueError(f"Design Optimization {algo_name} method not found")


class Coadaptation:
    def __init__(self, config):
        project_name = config["project_name"]
        self._initial_model_dir = config["initial_model_dir"]

        self._use_wandb = config["use_wandb"]
        self._use_gpu = config["use_gpu"]
        self.run_folder = config["run_folder"]

        self._config = config
        utils.move_to_cuda(self._config)

        if self._use_wandb:
            wandb.init(project=project_name, name=config["run_name"])

        self._episode_length = config["steps_per_episodes"]

        weight_pref = config["weight_preference"]
        self._weights_pref = torch.tensor(weight_pref).reshape(2, 1)
        if config["use_gpu"]:
            self._weights_pref = self._weights_pref.to("cuda")

        # initialize env
        env_cls = select_env(config["env"]["env_name"])
        # energy reward is approximately 1.725 larger than running reward
        self._env = env_cls(config=config, reward_scaling_energy=0.27532)
        self._reward_scale = config["rl_algorithm_config"]["algo_params"][
            "reward_scale"
        ]  # TODO: This should not depend on rl_algorithm_config in the future

        # initialize replay buffer (used by both RL algo and DO algo)
        self._replay = EvoReplayLocalGlobalStart(
            self._env,
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
        )

        # initialize RL algo
        self._rl_algo_class = select_rl_algo(config["rl_method"])
        self._networks = self._rl_algo_class.create_networks(self._env, config=config)
        self._rl_algo = self._rl_algo_class(
            config=config,
            env=self._env,
            replay=self._replay,
            networks=self._networks,
            weight_pref=self._weights_pref,
            wandb_instance=wandb.run if self._use_wandb else None,
            use_gpu=self._use_gpu,
        )
        if self._use_gpu:
            utils.move_to_cuda(config)
        else:
            utils.move_to_cpu()
            self._policy_cpu = self._rl_algo_class.get_policy_network(
                SoftActorCritic.create_networks(self._env, config=config)["individual"]
            )

        # initialize DO algo
        do_algo_class = select_design_opt_algo(config["design_optim_method"])
        self._do_alg = do_algo_class(config=config, replay=self._replay, env=self._env)

        # initialize counters
        self._last_single_iteration_time = 0
        self._design_counter = 0
        self._episode_counter = 0

        # load model from checkpoint if available
        if self._initial_model_dir is None:
            self._data_design_type = "Initial"
        else:
            self._data_design_type = "Pre-trained"
            self._load_checkpoint(self._initial_model_dir)

    def _load_checkpoint(self, exp_dir):
        # 1. load last design checkpoint
        do_dir = os.path.join(exp_dir, "do_checkpoints")
        file_path = max(os.listdir(do_dir), key=lambda fn: int(fn.split("_")[-1][:-4]))

        rows = []
        with open(os.path.join(do_dir, file_path), newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                rows.append(row)

        link_lengths = np.array(rows[1], dtype=float)
        self._env.set_new_design(link_lengths)

        # 2. load last RL checkpoint
        load_dir = os.path.join(exp_dir, "rl_checkpoints")
        filename = max(os.listdir(load_dir), key=lambda fn: int(fn.split("_")[-1][:-4]))
        network_path = os.path.join(load_dir, filename)

        network_data = torch.load(network_path, map_location=ptu.device)

        model_data_pop = network_data["population"]
        for key, net in self._networks["population"].items():
            params = model_data_pop[key]
            net.load_state_dict(params)
        model_data_ind = network_data["individual"]
        for key, net in self._networks["individual"].items():
            params = model_data_ind[key]
            net.load_state_dict(params)

        replay_path = os.path.join(load_dir, "latest_replay_buffer_0.chk")
        if os.path.isfile(replay_path):
            replay_buffer = torch.load(replay_path, map_location=torch.device("cpu"))
            self._replay.set_contents(replay_buffer)

    def initialize_episode(self):
        """Initializations required before the first episode.

        Should be called before the first episode of a new design is
        executed. Resets variables such as _data_rewards for logging purposes
        etc.
        """
        # self._rl_algo.initialize_episode(init_networks = True, copy_from_gobal = True)
        self._rl_algo.episode_init()
        self._replay.reset_species_buffer()

        self._data_rewards = []  # will be list of tuples in MORL setting
        self._states = []  # Needed for checking observations
        self._actions = []  # Needed for checking actions
        self._episode_counter = 0

    def collect_training_experience(self):
        """Collect training data.

        This function executes a single episode in the environment using the
        exploration strategy/mechanism and the policy.
        The data, i.e. state-action-reward-nextState, is stored in the replay
        buffer.
        """
        state = self._env.reset()
        nmbr_of_steps = 0
        done = False

        if self._episode_counter < self._config["initial_episodes"]:
            policy_gpu_ind = self._rl_algo_class.get_policy_network(
                self._networks["population"]
            )
        else:
            policy_gpu_ind = self._rl_algo_class.get_policy_network(
                self._networks["individual"]
            )

        if self._use_gpu:
            self._policy_cpu = policy_gpu_ind
            utils.move_to_cuda(self._config)
        else:
            self._policy_cpu = utils.copy_network(
                network_to=self._policy_cpu,
                network_from=policy_gpu_ind,
                config=self._config,
                force_cpu=True,
            )
            utils.move_to_cpu()

        while not done and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state)
            new_state, reward, done, info = self._env.step(action)
            # TODO this has to be fixed _variant_spec
            reward = reward * self._reward_scale
            terminal = np.array([done])
            # reward = reward # reward = np.array([reward]) # No need to be converted to numpy array since rewards are now in np.array([0], [1])
            self._replay.add_sample(
                observation=state,
                action=action,
                reward=reward,
                next_observation=new_state,
                terminal=terminal,
            )
            state = new_state
        self._replay.terminate_episode()
        utils.move_to_cuda(self._config)

    def execute_policy(self):
        """Evaluates the current deterministic policy.

        Evaluates the current policy in the environment by unrolling a single
        episode in the environment.
        The achieved cumulative reward is logged.

        """
        state = self._env.reset()
        done = False
        reward_ep = np.array(
            [0, 0]
        )  # SORL #reward_ep = 0.0 # We have two goals right now -> changes
        # reward_original = np.array([0, 0])# SORL #reward_original = 0.0  # -> same
        action_cost = 0.0
        nmbr_of_steps = 0
        states_arr = np.empty((0, 23))  # state saving
        actions_arr = np.empty((0, 6))  # action saving

        if self._episode_counter < self._config["initial_episodes"]:
            policy_gpu_ind = self._rl_algo_class.get_policy_network(
                self._networks["population"]
            )
        else:
            policy_gpu_ind = self._rl_algo_class.get_policy_network(
                self._networks["individual"]
            )
        if self._use_gpu:
            self._policy_cpu = policy_gpu_ind
            utils.move_to_cuda(self._config)
        else:
            self._policy_cpu = utils.copy_network(
                network_to=self._policy_cpu,
                network_from=policy_gpu_ind,
                config=self._config,
                force_cpu=True,
            )
            utils.move_to_cpu()

        while not done and nmbr_of_steps <= self._episode_length:
            nmbr_of_steps += 1
            action, _ = self._policy_cpu.get_action(state, deterministic=True)
            new_state, reward, done, info = self._env.step(action)
            # print(action.shape)
            # print(state.shape)
            states_arr = np.vstack(
                (states_arr, state)
            )  # needed for saving the states #.append(action) #
            actions_arr = np.vstack(
                (actions_arr, action)
            )  # needed for saving the actions # .append(state)
            action_cost += info["orig_action_cost"]
            # reward_ep += float(reward) #NOT WORKING  #SORL # changes need to be made here to convert scalar rewards to tuple or np.array
            reward_ep = np.add(reward_ep, reward, casting="unsafe")  # MORL
            # reward_original += info['orig_reward'] #SORL#reward_original += float(info['orig_reward'])
            # reward_original = np.add(reward_original,info['orig_reward'], casting='unsafe') # needed for UFucOutputCastingError # updated for update3 #update 6, unneeded?
            state = new_state
        utils.move_to_cuda(self._config)

        self._data_rewards.append(reward_ep)
        self._states.append(states_arr)
        self._actions.append(actions_arr)

        if self._use_wandb:
            # save episodic reward
            r1, r2 = reward_ep
            wandb.log({"Reward run": r1, "Reward energy consumption": r2})

    def save_logged_data(self):
        """Saves the logged data to the disk as csv files.

        This function creates a log-file in csv format on the disk. For each
        design an individual log-file is creates in the experiment directory.
        The first row states if the design was one of the initial designs
        (as given by the environment), a random design or an optimized design.
        The second row gives the design parameters (eta). The third row
        contains all subsequent cumulative rewards achieved by the policy
        throughout the reinforcement learning process on the current design.
        """
        current_design = self._env.get_current_design()
        save_dir = os.path.join(self.run_folder, "do_checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(
            os.path.join(save_dir, f"data_design_{self._design_counter}.csv"), "w"
        ) as file:
            cwriter = csv.writer(file)
            cwriter.writerow(["Design Type:", self._data_design_type])
            cwriter.writerow(current_design)
            cwriter.writerow([reward[0] for reward in self._data_rewards])
            cwriter.writerow([reward[1] for reward in self._data_rewards])

        if self._use_wandb:
            # save rewards and current design
            wandb.log({"Rewards": self._data_rewards, "Current design": current_design})

    def save_rl_checkpoint(self):
        """Saves the networks and replay buffer to disk if specified in config."""
        save_dir = os.path.join(self.run_folder, "rl_checkpoints")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if self._config["save_networks"]:
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

            file_name = f"networks_for_design_{self._design_counter}.chk"
            torch.save(network_checkpoint, os.path.join(save_dir, file_name))

        if self._config["save_replay_buffer"]:
            # overwrite last checkpoint for replay buffer
            # (saving it for each design would consume a large amount of memory)
            replay_buffer_checkpoint = self._replay.get_contents()
            file_name = f"latest_replay_buffer_0.chk"  # "_0" is for convenient loading
            torch.save(replay_buffer_checkpoint, os.path.join(save_dir, file_name))

    def single_rl_iteration(self):
        """A single iteration.

        Makes all necessary function calls for a single iterations such as:
            - Collecting training data
            - Executing a training step
            - Evaluate the current policy
            - Log data
        """
        print(
            f"Time for one iteration: {time.time() - self._last_single_iteration_time}"
        )
        self._last_single_iteration_time = time.time()
        self._replay.set_mode("species")
        self.collect_training_experience()

        # TODO Change here to train global only after five designs
        train_pop = self._design_counter > 3
        if self._episode_counter >= self._config["initial_episodes"]:
            self._rl_algo.single_train_step(train_ind=True, train_pop=train_pop)

        self._episode_counter += 1
        self.execute_policy()
        self.save_logged_data()

    def _intial_design_loop(self, iterations):
        """The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.

        """
        self._data_design_type = "Initial"

        if self._initial_model_dir is None:
            for params in self._env.init_sim_params:
                self._design_counter += 1
                self._env.set_new_design(params)
                self.initialize_episode()
                for _ in range(iterations):
                    self.single_rl_iteration()
                self.save_rl_checkpoint()
        else:
            self._design_counter += 1
            self.initialize_episode()
            for _ in range(iterations):
                self.single_rl_iteration()
            self.save_rl_checkpoint()

    def _training_loop(self, iterations, design_cycles, exploration_strategy):
        """The trianing process which optimizes designs and policies.

        The function executes the reinforcement learning loop and the design
        optimization process.

        Args:
            iterations: An integer stating the number of iterations/episodes
                to be used per design during the reinforcement learning loop.
            design_cycles: Integer stating how many designs to evaluate.
            exploration_strategy: String which describes which
                design exploration strategy to use. Is not used at the moment,
                i.e. only the (uniform) random exploration strategy is used.

        """
        self.initialize_episode()
        # TODO fix the following
        initial_state = self._env._env.reset()

        self._data_design_type = "Optimized"

        optimized_params = self._env.get_random_design()
        q_network = self._rl_algo_class.get_q_network(self._networks["population"])
        policy_network = self._rl_algo_class.get_policy_network(
            self._networks["population"]
        )

        optimized_params = self._do_alg.optimize_design(
            design=optimized_params,
            q_network=q_network,
            policy_network=policy_network,
            weights=self._weights_pref,
        )

        optimized_params = list(optimized_params)

        for i in range(design_cycles):
            self._design_counter += 1
            self._env.set_new_design(optimized_params)

            # Reinforcement Learning
            for _ in range(iterations):
                self.single_rl_iteration()
            self.save_rl_checkpoint()

            # Design Optimization
            if i % 2 == 1:
                self._data_design_type = "Optimized"
                q_network = self._rl_algo_class.get_q_network(
                    self._networks["population"]
                )
                policy_network = self._rl_algo_class.get_policy_network(
                    self._networks["population"]
                )
                optimized_params = self._do_alg.optimize_design(
                    design=optimized_params,
                    q_network=q_network,
                    policy_network=policy_network,
                    weights=self._weights_pref,
                )  # Need weights for MORL optimizer
                optimized_params = list(optimized_params)
            else:
                self._data_design_type = "Random"
                optimized_params = self._env.get_random_design()
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
        self._intial_design_loop(self._config["iterations_init"])
        self._training_loop(
            iterations=self._config["iterations"],
            design_cycles=self._config["design_cycles"],
            exploration_strategy=self._config["exploration_strategy"],
        )
