import argparse
from datetime import datetime
import hashlib
import json
import os
import random

import numpy as np
import torch
from utils import add_argparse_arguments
import wandb

import coadapt
from configs import all_configs

ARGS = [
    ("config-id", "sac_pso_batch"),
    ("run-name", False),
    ("data-folder", False),
    ("load-replay-buffer", False),
    ("save-replay-buffer", False),
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
    # NOTE: For default values, see the specific config files
    parser = argparse.ArgumentParser()
    add_argparse_arguments(parser, ARGS)

    return parser.parse_args()


def load_config(args):
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
