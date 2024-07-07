import argparse
from datetime import datetime
import hashlib
import json
import os
import random

import numpy as np
import torch
import wandb

import coadapt
from configs import all_configs


def strtobool(val):
    """Convert a string representation of truth to True or False.
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


def parse_args():
    parser = argparse.ArgumentParser()

    # NOTE: For default values, see the specific config files

    parser.add_argument(
        "--config-id",
        type=str,
        help="Name of config file, to specify which config to load",
        choices=("sac_pso_batch", "sac_pso_sim", "sac_pso_batch_vec"),
        default="sac_pso_batch",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Human-readable name of the experiment, part of experiment folder name and used as wandb run name",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        help="Path to parent folder of experiment run",
    )
    parser.add_argument(
        "--save-replay-buffer",
        type=str,
        help="Use True to save the most recent RL replay buffer (can be a few GB large)",
    )
    parser.add_argument(
        "--initial-model-dir",
        type=str,
        help="Use True to load the latest checkpoint from this experiment folder at the beginning of training",
    )
    parser.add_argument(
        "--weight-preference",
        type=float,
        nargs="+",
        help="Objective preference in MORL setting (one non-negative value per objective, should add up to 1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility (if None, the seed will be randomly generated)",
    )
    parser.add_argument(
        "--use-gpu",
        type=strtobool,
        help="Use True to train with GPU and False to train with CPU",
    )
    parser.add_argument(
        "--verbose",
        type=strtobool,
        help="Use True for more verbose console output",
    )
    parser.add_argument(
        "--use-wandb",
        type=strtobool,
        help='Use True to log the run with wandb ("weights and biases")',
    )
    return parser.parse_args()


def load_config(args):
    # Load requested config
    config = all_configs[args.config_id]

    # Overwrite config values if specified with argparse
    for arg_name in [
        "run_name",
        "data_folder",
        "initial_model_dir",
        "weight_preference",
        "random_seed",
        "use_gpu",
        "verbose",
        "use_wandb",
    ]:
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[arg_name] = arg_value

    # Add experiment ID to config (NOTE: os.urandom is independent of random seeding)
    config["run_id"] = hashlib.md5(os.urandom(128)).hexdigest()[:8]

    # Add current timestamp to config (without ":" for cross-platform compatibility)
    timestamp = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
    config["timestamp"] = timestamp

    # Create run folder
    run_name = config["run_name"]
    run_folder = os.path.join(config["data_folder"], f"run_{timestamp}_{run_name}")
    config["run_folder"] = run_folder
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)

    # Store config
    with open(os.path.join(run_folder, "config.json"), "w") as file:
        file.write(json.dumps(config, indent=2))

    # Apply random seeding
    seed = config["random_seed"]
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        config["random_seed"] = seed  # Save generated seed to config
        if config["verbose"]:
            print(f"Setting random_seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config["verbose"]:
        print("\n=== CONFIG FILE USED FOR EXPERIMENT ===")
        print(json.dumps(config, sort_keys=True, indent=4))
        print("=======================================\n")

    return config


def launch_experiment(config):
    co = coadapt.Coadaptation(config)
    co.run()  # run training loop

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    launch_experiment(config)
