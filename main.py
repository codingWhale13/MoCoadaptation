import os, sys, time
import hashlib
import coadapt
import experiment_configs as cfg
import json
import wandb
import torch
import numpy as np
import random


def main(config_name, weight_index):
    # Random seeding
    if seed is None:
        print(f"Custom seed set not set, using random seed")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Custom seed set: {seed}")

    # Create foldr in which to save results
    folder = config_name["data_folder"]
    # generate random hash string - unique identifier if we start
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]

    file_str = f"./{folder}/{time.ctime().replace(' ', '_')}__{rand_id}{str(config_name['weights'][weight_index])}"
    if seed is not None:
        file_str += f"_{seed}"

    config_name["data_folder_experiment"] = file_str
    # Create experiment folder
    if not os.path.exists(file_str):
        os.makedirs(file_str)
    # Store config
    with open(os.path.join(file_str, "config.json"), "w") as fd:
        fd.write(json.dumps(config_name, indent=2))

    # Start training
    co = coadapt.Coadaptation(config_name, weight_index, project_name, run_name)
    co.run()
    # wandb.finish()


if __name__ == "__main__":
    if len(sys.argv) > 5:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = int(sys.argv[5])
        print(run_name)
        print(project_name)
        print(f"index w : {weight_index}")
        print(config_name["weights"][weight_index])
    elif len(sys.argv) > 4:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = False
        print(run_name)
        print(project_name)
        print(f"index w : {weight_index}")
        print(config_name["weights"][weight_index])
    else:
        # config_name = cfg.config_dict['sac_pso_sim']
        config_name = cfg.config_dict["sac_pso_sim"]
        weight_index = 0
        seed = False
        project_name = "coadapt-save-testing"
        run_name = f"default-run-weight-{config_name['weights'][weight_index]}"
        print(config_name["weights"][weight_index])
    main(config_name, weight_index)
