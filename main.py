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
import experiment_configs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-name",
        type=str,
        help="Type of experiment (choose 'sac_pso_batch' or 'sac_pso_sim')",
        choices=("sac_pso_batch", "sac_pso_sim"),
        default="sac_pso_sim",
    )
    parser.add_argument(
        "--weight-index",
        type=int,
        help="Weight index to use (0, 1, ..., 10)",
        choices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        default=0,
    )
    parser.add_argument(
        "--project-name",
        type=str,
        help="Name to identify the project",
        default="Co-Adaptation",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name to identify the run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (if None, random seed is used)",
    )
    parser.add_argument(
        "--use-wandb",
        type=bool,
        help="Whether or not training should be logged with wandb",
        default=False,
    )
    parser.add_argument(
        "--initial-model-path",
        type=str,
        help="If specified, loads model at beginning of training (bootstrapping)",
    )

    return parser.parse_args()


def main(config):
    weight_index = config["weight_index"]
    use_wandb = config["use_wandb"]

    # random seeding
    seed = config["seed"]
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        config["seed"] = seed  # save generated seed to config
        print(f"Custom seed set not set, using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Custom seed set: {seed}")

    # generate unique experiment name and create folder if not exists
    data_folder = config["data_folder"]
    timestamp = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]  # unique identifier
    weight_str = "-".join([str(i) for i in config["weights"][weight_index]])
    exp_id = f"run_{timestamp}_{rand_id}_{weight_str}"
    data_folder_experiment = os.path.join(data_folder, exp_id)
    config["data_folder_experiment"] = data_folder_experiment
    if not os.path.exists(data_folder_experiment):
        os.makedirs(data_folder_experiment)

    # store config
    with open(os.path.join(data_folder_experiment, "config.json"), "w") as fd:
        fd.write(json.dumps(config, indent=2))

    # start training
    co = coadapt.Coadaptation(config)
    co.run()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    config = experiment_configs.config_dict[args.config_name]

    run_name = args.run_name
    if run_name is None:
        run_name = f"default-run-weight-{config['weights'][args.weight_index]}"

    config["project_name"] = args.project_name
    config["run_name"] = args.run_name
    config["weight_index"] = args.weight_index
    config["seed"] = args.seed
    config["use_wandb"] = args.use_wandb
    config["initial_model_path"] = args.initial_model_path
    config["load_model"] = args.initial_model_path is not None

    print(json.dumps(config, sort_keys=True, indent=4))

    main(config)
