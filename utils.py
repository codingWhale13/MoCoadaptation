import argparse
import csv
import cv2
import os
from shutil import copyfile, move
import time

import json
import numpy as np
import pybullet as p

import rlkit.torch.pytorch_util as ptu


# ========== PYTORCH HELPERS ==========


def move_to_cpu():
    """Set device to cpu for torch"""
    ptu.set_gpu_mode(False)


def move_to_cuda(cuda_device=None):
    """Set device to gpu for torch"""
    if cuda_device is None:
        cuda_device = 0
    ptu.set_gpu_mode(True, cuda_device)


def copy_pop_to_ind(networks_pop, networks_ind):
    """Function used to copy params from pop. networks to individual networks.

    The parameters of all networks in network_ind will be set to the parameters
    of the networks in networks_ind.

    Args:
        networks_pop: Dictonary containing the population networks.
        networks_ind: Dictonary containing the individual networks. These
            networks will be updated.
    """
    for key in networks_pop:
        state_dict = networks_pop[key].state_dict()
        networks_ind[key].load_state_dict(state_dict)
        networks_ind[key].eval()


def copy_network(network_to, network_from, cuda_device=None, force_cpu=False):
    """Copies networks and set them to device or cpu.

    Args:
        networks_to: Networks to which we want to copy (destination).
        networks_from: Networks from which we want to copy (source). These
            networks will be changed.
        force_cpu: Boolean, if True the destination networks will be placed on
            the cpu. If not, the current device will be used.
    """
    network_from_dict = network_from.state_dict()
    if force_cpu:
        for key, val in network_from_dict.items():
            network_from_dict[key] = val.cpu()
    else:
        move_to_cuda(cuda_device)
    network_to.load_state_dict(network_from_dict)
    if force_cpu:
        network_to = network_to.to("cpu")
    else:
        network_to.to(ptu.device)
    network_to.eval()
    return network_to


# ========== VIDEO RECORDING ==========


class BestEpisodesVideoRecorder:
    def __init__(self, path=None, max_videos=1, record_evy_n_episodes=5):
        self._vid_path = "/tmp/videos" if path is None else path

        self._folder_counter = 0
        self._keep_n_best = max(max_videos, 1)
        self._record_evy_n_episodes = record_evy_n_episodes

        self._frame_width = 200
        self._frame_height = 200
        self._fps_per_frame = 0

        self.increase_folder_counter()
        self._create_vid_stream()
        self._time_start = time.time()

    def _episodic_reset(self):
        self._current_episode_reward = 0
        self._did_at_least_one_step = False
        self._step_counter = 1

    def reset_recorder(self):
        self._episode_counter = 0
        # self._episodic_rewards = [-float('inf')] * self._keep_n_best # ORIG
        self._episodic_rewards = np.full(
            self._keep_n_best, -np.inf
        )  # Need for numpy array since rewards has two parts and it is numpy array
        self._episodic_reset()

    def increase_folder_counter(self):
        self._current_vid_path = os.path.join(self._vid_path, str(self._folder_counter))
        self.reset_recorder()
        self._folder_counter += 1

    def step(self, env, state, reward, done):
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._current_episode_reward += reward
            env.camera_adjust()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)
            self._did_at_least_one_step = True
        proc_time = (time.time() - self._time_start) * 1000
        proc_time = 1000 / proc_time
        self._time_start = time.time()
        self._fps_per_frame += proc_time
        self._step_counter += 1

    def _do_video_file_rotation(self):
        for idx, elem in enumerate(self._episodic_rewards):
            if idx > 1:
                try:
                    move(
                        os.path.join(self._current_vid_path, f"video_{idx-1}.avi"),
                        os.path.join(self._current_vid_path, f"video_{idx-2}.avi"),
                    )
                except:
                    pass
            if (
                self._current_episode_reward < elem
            ).any():  # changed for MORL since rewards are not scalar
                # self._episodic_rewards = self._episodic_rewards[1:idx] + [self._current_episode_reward] + self._episodic_rewards[idx:] # ORIG
                self._episodic_rewards = np.concatenate(
                    [
                        self._episodic_rewards[1:idx],
                        self._current_episode_reward.ravel(),
                        self._episodic_rewards[idx:],
                    ]
                )  # Fix for dimensions

                copyfile(
                    os.path.join(self._current_vid_path, "current_video.avi"),
                    os.path.join(
                        self._current_vid_path, "video_{}.avi".format(idx - 1)
                    ),
                )
                break
            # Will only be true in last iteration and only be hit if last element is to be moved
            if idx == len(self._episodic_rewards) - 1:
                try:
                    move(
                        os.path.join(
                            self._current_vid_path, "video_{}.avi".format(idx)
                        ),
                        os.path.join(
                            self._current_vid_path, "video_{}.avi".format(idx - 1)
                        ),
                    )
                except:
                    pass
                # self._episodic_rewards = self._episodic_rewards[1:] + [self._current_episode_reward] #ORIG
                self._episodic_rewards = np.concatenate(
                    [self._episodic_rewards[1:], self._current_episode_reward.ravel()]
                )  # Fix for dimensions

                copyfile(
                    os.path.join(self._current_vid_path, "current_video.avi"),
                    os.path.join(self._current_vid_path, "video_{}.avi".format(idx)),
                )

    def reset(self, env, state, reward, done, verbose=False):
        # Final processing of data from previous episode
        if self._episode_counter % self._record_evy_n_episodes == 0:
            env.camera_adjust()
            self._vid_writer.release()
            os.makedirs(self._current_vid_path, exist_ok=True)

            # If self._did_at_least_one_step and min(self._episodic_rewards) < self._current_episode_reward: # ORIG
            if self._did_at_least_one_step and np.min(self._episodic_rewards) < np.min(
                self._current_episode_reward
            ):  # ('comparison')).any(): # changed for MORL comparison for two rewards
                self._do_video_file_rotation()
            if verbose:
                print(
                    f"Average FPS of last episode: {self._fps_per_frame / self._step_counter}"
                )

        self._episode_counter += 1
        self._episodic_reset()
        # Set up everything for this episode if we record
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._create_vid_stream()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)

            self._time_start = time.time()
            self._fps_per_frame = 0
            self._step_counter = 1

    def _create_vid_stream(self):
        os.makedirs(self._current_vid_path, exist_ok=True)
        self._vid_writer = cv2.VideoWriter(
            os.path.join(self._current_vid_path, "current_video.avi"),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            30,
            (self._frame_width, self._frame_height),
        )


class SimpleVideoRecorder:
    def __init__(self, env, save_dir="", file_name="single_episode"):
        self._env = env
        self._frame_width = 200
        self._frame_height = 200

        os.makedirs(save_dir, exist_ok=True)
        self._vid_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{file_name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (self._frame_width, self._frame_height),
        )

        frame = env.render_camera_image((self._frame_width, self._frame_height))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._vid_writer.write(frame)

        # Connect to PyBullet server (for some reason important when recording videos with only a few frames)
        if p.isConnected() == 0:
            p.connect(p.GUI)

    def step(self):
        self._env.camera_adjust()
        frame = self._env.render_camera_image((self._frame_width, self._frame_height))
        frame = frame * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._vid_writer.write(frame)

    def save_video(self):
        self._env.camera_adjust()
        self._vid_writer.release()


# ========== FILE AND FOLDER HANDLING ==========

# Constant names to avoid confusion due to typos
CONFIG = "config.json"  # Used for training
ORIGINAL_CONFIG = "original_config.json"  # Used for testing
DO_CHECKPOINTS = "do_checkpoints"  # Folder in run_dir
RL_CHECKPOINTS = "rl_checkpoints"  # Folder in run_dir
DESIGN_TYPE = "Design Type"  # Used in CSV file
DESIGN_PARAMETERS = "Design Parameters"  # Used in CSV file


def load_config(load_dir, filename=CONFIG):
    with open(os.path.join(load_dir, filename)) as file:
        config = json.load(file)

    # Backward compatibility: Add fields if they don't exists
    if "previous_weight_preferences" not in config:
        config["previous_weight_preferences"] = []
    if "old_replay_portion" not in config:
        config["old_replay_portion"] = 0

    return config


def save_config(config: dict, save_dir, filename=CONFIG):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w") as file:
        file.write(json.dumps(config, indent=2))


def exp_dir_to_run_dirs(exp_dir):
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(exp_dir):
        if (
            CONFIG in filenames
            and DO_CHECKPOINTS in dirnames
            and RL_CHECKPOINTS in dirnames
        ):  # Make sure experiment did not immediately crash
            run_dirs.append(dirpath)

    return run_dirs


def get_cycle_count(run_dir):
    do_dir = os.path.join(run_dir, DO_CHECKPOINTS)
    rl_dir = os.path.join(run_dir, RL_CHECKPOINTS)

    # Strip last four characters to get rid of filename ending (.csv or .chk)
    do_cycle_count = max(map(lambda x: int(x.split("_")[-1][:-4]), os.listdir(do_dir)))
    rl_cycle_count = max(map(lambda x: int(x.split("_")[-1][:-4]), os.listdir(rl_dir)))
    assert do_cycle_count == rl_cycle_count, f"Found inconsistent experiment: {run_dir}"

    return do_cycle_count


def load_csv(filepath) -> dict:
    """Read a CSV file and return a dictionary containing design and rewards.

    The new CSV format is consistent across files in the `do_checkpoins` folder as well as test results during evaluation:
    1. First row contains two strings:                  DESIGN_TYPE, {design type, e.g. "Optimized"}
    2. Second row contains one string and some floats:  DESIGN_PARAMETERS, {value}, {value}, ...
    3. Third row contains one string and some floats:   {name of first reward dimension, e.g. "Reward Speed"}, {value}, {value}, ...
    4. Fourth row contains one string and some floats:  {name of second reward dimension, e.g. "Reward Energy"}, {value}, {value}, ...
    5. ... (more reward dimensions)

    However, this function can also handle CSV files in previous format, accounting for:
    * Implicit reward names for HalfCheetah and
    * Test results missing row with design type information

    Returns dictionary of format:
    {
        DESIGN_TYPE: {some string}
        DESIGN_PARAMETERS: {list of design parameters (e.g. limb lengths)}
        {reward name 1}: {list of episodic rewards}
        {reward name 2}: {list of episodic rewards}
        ...
    }
    """

    values = {}
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

        if rows[0][0].startswith(DESIGN_TYPE):  # Watching out for "Design Type:"
            values[DESIGN_TYPE] = rows[0][1]
            values[DESIGN_PARAMETERS] = rows[0][1]
            if rows[1][0] == DESIGN_PARAMETERS:
                # Format is up-to-date, reward names will be explicit
                values[DESIGN_PARAMETERS] = np.array(rows[1][1:], dtype=float)
                for row_idx in range(2, len(rows)):
                    row = rows[row_idx]
                    row_name = row[0]
                    row_data = np.array(row[1:], dtype=float)  # Reward values
                    values[row_name] = row_data
            else:
                values[DESIGN_PARAMETERS] = np.array(rows[1], dtype=float)
                values["Reward Speed"] = np.array(rows[2], dtype=float)
                values["Reward Energy"] = np.array(rows[3], dtype=float)
        else:
            # Old format for test results: 3 rows of floats, no description at all
            values[DESIGN_TYPE] = "Unknown"
            values[DESIGN_PARAMETERS] = np.array(rows[0], dtype=float)
            values["Reward Speed"] = np.array(rows[1], dtype=float)
            values["Reward Energy"] = np.array(rows[2], dtype=float)

    return values


def save_csv(
    save_dir: str,
    filename: str,
    design_type: str,
    design_parameters: list[float],
    rewards: dict[str, list[float]],
):
    """Saves design and rewards (logged during training or testing) as CSV.

    The first row states if the design was one of the initial designs
    (as given by the environment), a random design or an optimized design.
    The second row gives the design parameters. The third (and
    following rows) contains all subsequent cumulative rewards achieved by
    the policy throughout the RL process on the current design.
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w") as file:
        cwriter = csv.writer(file)
        cwriter.writerow([DESIGN_TYPE, design_type])
        cwriter.writerow([DESIGN_PARAMETERS] + list(design_parameters))
        for reward_name, rewards in rewards.items():
            cwriter.writerow([reward_name] + list(rewards))


def get_reward_names_from_csv(filepath) -> list[str]:
    """Get reward names in the same order as in the CSV (and thus in the env definition)"""
    assert filepath.endswith(".csv"), f"Expected CSV file, but got '{filepath}'"
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
        if rows[0][0] == DESIGN_TYPE and rows[1][0] == DESIGN_PARAMETERS:
            # Format is up-to-date, reward names will be explicit
            reward_names = []
            for row_idx in range(2, len(rows)):
                reward_names.append(rows[row_idx][0])
            return reward_names
    return ["Reward Speed", "Reward Energy"]  # Support legacy CSV files


def filenamify(s):
    """Turn a string into a lowercase fileanme without spaces"""
    return s.lower().replace(" ", "_")


def pref2str(pref: list[float]) -> str:
    return "-".join(map(str, pref))


def str2bool(val: str) -> bool:
    """
    Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else. Similar to deprecated distutils.util.strtobool.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"Invalid truth value '{val}'")


def int_or_last(value):
    if value == "last":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value} is not a valid integer or "last"')


def add_argparse_arguments(parser, arguments):
    """
    Adds arguments to parser, supplying consistent help messages.

    :param parser: The parser to add the arguments to (will be modified)
    :param arguments: should be a collection of tuples of form (A, B):
        A: String, the argument name (with dashes between words and no leading dashes, e.g. "arg-name")
        B: Boolean (required? yes/no) or other type (default value)
           NOTE: In case of boolean default value, use string "true" or "false"
    """

    # Define immutable keyword arguments for `parser.add_argument()`, in particular type and help
    fixed_kwargs = {
        # GENERAL ARGUMENTS (USEFUL IN MULTIPLE CONTEXTS)
        "exp-dir": {
            "type": str,
            "help": "Path to folder containing one or more runs (which can be nested in subfolders)",
        },
        "run-dir": {
            "type": str,
            "help": "Path to individual run folder (containing a config.json file)",
        },
        "run-dirs": {
            "type": str,
            "help": "One or more paths to individual run folders (each containing a config.json file)",
            "nargs": "+",
        },
        "random-seed": {
            "type": int,
            "help": "Random seed for reproducibility (if not specified, the seed will be randomly generated)",
        },
        "n-tests": {
            "type": int,
            "help": "Number of tests to run",
        },
        "n-iters": {
            "type": int,
            "help": "Number of iterations",
        },
        "use-gpu": {
            "type": str2bool,
            "help": "Use True for running the code on GPU, use False for CPU",
        },
        "save-dir": {
            "type": str,
            "help": "Output folder to save results",
        },
        "design-cycle": {
            "type": int_or_last,
            "help": 'Specifies design cycle of interest as an integer (or "last")',
        },
        "verbose": {
            "type": str2bool,
            "help": "Use True for more verbose console output",
        },
        # TRAINING ARGUMENTS
        "config-id": {
            "type": str,
            "help": "Name of config file, to specify which config to load",
            "choices": (
                "sac_pso_batch_halfcheetah",
                "sac_pso_sim_halfcheetah",
                "sac_pso_batch_vec",
                "sac_pso_batch_walker2d",
                "sac_pso_batch_hopper",
            ),
        },
        "data-folder": {
            "type": str,
            "help": "Path to folder where the data of the experiment run is saved (will contain a config.json file)",
        },
        "run-name": {
            "type": str,
            "help": "Human-readable name of the experiment, part of experiment folder name and used as wandb run name",
        },
        "initial-model-dir": {
            "type": str,
            "help": "If specified, the latest checkpoint from this experiment is loaded at the beginning of training",
        },
        "load-replay-buffer": {
            "type": str2bool,
            "help": "Use True to load the most recent RL replay buffer (can be a few GB large, only takes effect if --initial-model-dir is specified)",
        },
        "old-replay-portion": {
            "type": float,
            "help": "Value between 0 and 1, specifying fraction of samples from old replay buffer (only takes effect with --load-replay-buffer set to True)",
        },
        "save-replay-buffer": {
            "type": str2bool,
            "help": "Use True to save the most recent RL replay buffer (can be a few GB large)",
        },
        "weight-preference": {
            "type": float,
            "help": "Objective preference in MORL setting (one non-negative value per objective, should add up to 1)",
            "nargs": "+",
        },
        "condition-on-preference": {
            "type": str2bool,
            "help": "Use True to condition policy and Q-networks on the preference",
        },
        "use-vector-q": {
            "type": str2bool,
            "help": "Use True for vector output of Q-network, use False for scalar Q-values",
        },
        "use-wandb": {
            "type": str2bool,
            "help": 'Use True to log the run with wandb ("Weights and Biases")',
        },
        # EVALUATION AND PLOTTING ARGUMENTS
        "skip-initial-designs": {
            "type": str2bool,
            "help": "Use True to skip testing on initial designs",
        },
        "skip-random-designs": {
            "type": str2bool,
            "help": "Use True to skip testing on random designs",
        },
        "test-results-dir": {
            "type": str,
            "help": "Path to folder with test results (each subfolder should contain 1+ CSV files and `original_config.json`)",
        },
        "direct-test-dir": {
            "type": str,
            "help": "Path to folder with baseline test results",
        },
        "steered-test-dirs": {
            "type": str,
            "help": "One or more paths to folders with test results of steered agent",
            "nargs": "+",
        },
        "common-name": {
            "type": str,
            "help": "Human-readable name for comparing across multiple experiment runs, will be included e.g. in plot title and file name",
        },
        "marker-size": {
            "type": int,
            "help": "Size of markers used for plotting",
        },
        "video-dirs": {
            "type": str,
            "help": "One or more paths to folders containing videos of single runs, generated by `video(s)_from_checkpoint.py`",
            "nargs": "+",
        },
    }

    for arg_name, req_def in arguments:
        if arg_name not in fixed_kwargs:
            raise ValueError(f'Argument name "{arg_name}" not found')

        parser.add_argument(
            f"--{arg_name}",
            **fixed_kwargs[arg_name],
            **(
                {"required": req_def}
                if isinstance(req_def, bool)
                else {"default": req_def}
            ),
        )
