import os, sys, time
import hashlib
import coadapt
import experiment_configs as cfg
import json
import wandb
import torch
import numpy as np
import random

#def main(config): # ORIG'
def main(config_name, weight_index): # MORL

    # Create foldr in which to save results
    #folder = config['data_folder'] #ORIG
    folder = config_name['data_folder'] #MORL
    #generate random hash string - unique identifier if we startexi
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    if seed:
        file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + str(config_name['weights'][weight_index]).replace(' ', '') + '_' + str(seed)# add seed to filename
    else:
        file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + str(config_name['weights'][weight_index]).replace(' ', '') # do not add the seed to filename
    #config['data_folder_experiment'] = file_str # ORIG
    config_name['data_folder_experiment'] = file_str # MORL

    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)
    
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config_name,indent=2))                  #MORL
    
    #co = coadapt.Coadaptation(config) # ORIG 
    co = coadapt.Coadaptation(config_name, weight_index, project_name, run_name, model_path, track) # MORL
    # Set custom seed
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Custom seed set: {seed}")
    else:
        print(f"Custom seed set not set, using random seed")
    #run coadapt
    co.run()
    #wandb.finish()


if __name__ == "__main__":
    if len(sys.argv) > 7:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = int(sys.argv[5])
        model_path = sys.argv[6]
        track = eval(sys.argv[7])
        print(run_name)
        print(project_name)
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(f"seed : {seed}")
        print(config_name['weights'][weight_index])
        print(f"Previous model provided for starting point: {model_path}")
        print(f"Model tracking : {track}")
    if len(sys.argv) > 6:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = int(sys.argv[5])
        model_path = sys.argv[6]
        track = False
        print(run_name)
        print(project_name)
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(f"seed : {seed}")
        print(config_name['weights'][weight_index])
        print(f"Previous model provided for starting point: {model_path}")
    elif len(sys.argv) > 5:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        seed = int(sys.argv[5])
        model_path = None
        track = False
        print(run_name)
        print(project_name)
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(config_name['weights'][weight_index])
    elif len(sys.argv) > 4:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        model_path = None
        seed = False
        track = False
        print(run_name)
        print(project_name)
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(config_name['weights'][weight_index]) 
    else:
        config_name = cfg.config_dict['sac_pso_batch']
        weight_index = 7
        seed = False
        project_name="coadapt-save-testing"
        run_name="default-run-weight" + f"-{config_name['weights'][weight_index]}"
        print(config_name['weights'][weight_index])
        seed = False
        model_path = None
        track = False
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(f"seed : {seed}")
        print(config_name['weights'][weight_index])
    main(config_name, weight_index) # MORL