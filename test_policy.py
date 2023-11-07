import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb


#path='/home/oskar/Thesis/checkpoint_design_60' #data.pkl
#path='/home/oskar/Thesis/Results_scalarized/results_with_wandb/Wed_Oct_18_18:29:14_2023__49c4bc4c/checkpoints/checkpoint_design_60.chk' # works
path='/home/oskar/Thesis/Results_scalarized/results_with_wandb/Fri_Oct_20_17:52:03_2023__fb9514fe_.5_.5/checkpoints/checkpoint_design_60.chk' 
experiment_config = experiment_configs.sac_pso_sim #MORL
weight_index = 8 #Set as what the data was for # in this case 8
project_name = "coadapt-testing-video"
run_name = "0.5, 0.5"

if __name__ == "__main__":

    folder = experiment_config['data_folder'] #MORL
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id
    experiment_config['data_folder_experiment'] = file_str # MORL

    #initiliaze the class
    coadapt_test = coadapt.Coadaptation(experiment_config, weight_index, project_name, run_name)
    #load the networks
    coadapt_test.load_networks(path)
    #simulate the model
    for _ in range(30):
        coadapt_test.initialize_episode()
        coadapt_test.execute_policy()
    
    wandb.finish()