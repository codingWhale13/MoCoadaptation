import argparse
from datetime import datetime
import hashlib
import json
import os
import random

import numpy as np
import torch
from utils import add_argparse_arguments, load_config, save_config
import wandb

from coadapt import Coadaptation
from configs import all_configs


# Specify options (NOTE: For default values, see `configs` folder)
ARGS = [
    ("config-id", "sac_pso_batch_halfcheetah"),
    ("data-folder", True),
    ("run-name", False),
    ("load-replay-buffer", False),
    ("save-replay-buffer", False),
    ("old-replay-portion", False),
    ("initial-model-dir", False),
    ("weight-preference", False),
    ("condition-on-preference", False),
    ("use-vector-q", False),
    ("random-seed", False),
    ("use-gpu", False),
    ("verbose", False),
    ("use-wandb", False),
]


def parse_args():
    parser = argparse.ArgumentParser()
    add_argparse_arguments(parser, ARGS)

    return parser.parse_args()


def prepare_config(args):
    # Load requested config
    config = all_configs[args.config_id]

    # Overwrite config values if specified with argparse
    for arg_name, _ in ARGS:
        if arg_name == "config-id":
            continue

        arg_name = arg_name.replace("-", "_")
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[arg_name] = arg_value

    # Assert validity of weight preference
    assert (
        sum(config["weight_preference"]) == 1 and min(config["weight_preference"]) >= 0
    ), "Weight preference must consist of non-negative values, adding up to 1"

    # Add experiment ID to config (NOTE: os.urandom is independent of random seeding)
    config["run_id"] = hashlib.md5(os.urandom(128)).hexdigest()[:8]

    # Add current timestamp to config (without ":" for cross-platform compatibility)
    timestamp = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
    config["timestamp"] = timestamp

    # Append previous weight preferences
    if config["initial_model_dir"] is not None:
        prev_config = load_config(config["initial_model_dir"])
        prev_pref = prev_config["weight_preference"]
        prev_id = prev_config["run_id"]
        config["previous_weight_preferences"].append((prev_pref, prev_id))

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

    # Specify run folder and save config
    run_name = config["run_name"]
    run_folder = os.path.join(config["data_folder"], f"run_{timestamp}_{run_name}")
    config["run_folder"] = run_folder
    save_config(config, run_folder)

    if config["verbose"]:
        print("\n=== CONFIG FILE USED FOR EXPERIMENT ===")
        print(json.dumps(config, sort_keys=True, indent=2))
        print("=======================================\n")

    return config


def launch_experiment(config):
    co = Coadaptation(config)
    co.run()  # Run training loop

    if config["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = prepare_config(args)

    launch_experiment(config)
