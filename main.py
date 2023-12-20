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
    # Create foldr in which to save results
    folder = config_name['data_folder']
    #generate random hash string - unique identifier if we startexi
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    if seed:
        file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + str(config_name['weights'][weight_index]) + '_' + str(seed)# add seed to filename
    else:
        file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + str(config_name['weights'][weight_index]) # do not add the seed to filename
    config_name['data_folder_experiment'] = file_str
    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)
    # Store config
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config_name,indent=2))
    co = coadapt.Coadaptation(config_name, weight_index, project_name, run_name)
    # Set custom seed
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Custom seed set: {seed}")
    else:
        print(f"Custom seed set not set, using random seed")
    co.run()
    wandb.finish()


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
        print(config_name['weights'][weight_index])
    elif len(sys.argv) > 4:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = False
        print(run_name)
        print(project_name)
        print(f"index w : {weight_index}")
        print(config_name['weights'][weight_index]) 
    else:
        config_name = cfg.config_dict['sac_pso_sim']
        weight_index = 0
        seed = False
        project_name="coadapt-save-testing"
        run_name="default-run-weight" + f"-{config_name['weights'][weight_index]}"
        print(config_name['weights'][weight_index])
    main(config_name, weight_index)