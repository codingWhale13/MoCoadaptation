import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb
import csv
import numpy as np

#Add path to checkpoint and path to model folder
#path_model_checkpoint='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/Thu_Nov_16_21:54:23_2023__cbea1a7e_1._.0/checkpoints/checkpoint_design_46.chk' 
path_model_checkpoint='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/Wed_Nov_22_18:10:44_2023__07f5af99_0.01_0.99/checkpoints/checkpoint_design_60.chk' 
path_model_morphology='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/Wed_Nov_22_18:10:44_2023__07f5af99_0.01_0.99'
model_checkpoint_num = path_model_checkpoint.split('_')[-1][:-4]
morphology_number = model_checkpoint_num  + ".csv"
newline=''

experiment_config = experiment_configs.sac_pso_sim #MORL dictiornary need for batch or sim
weight_index = 5 # dummy value for creating the class, we dont update the network so the weight doesnt matter here
project_name = "test" #"coadapt-scaled-test"#"coadapt-testing-scaling-tests"
run_name = "test" #"0, 1, 5"#"1, 0, 7"

def read_morphology(path) -> list:
    
    """ Returns a list of values read from cvs file per row
    
    Returns:
        a list containing csv file values
    """    
    rows = []
    for filename in os.listdir(path):
        if filename.endswith(morphology_number):
            filepath = os.path.join(path, filename)
            with open(filepath, newline=newline) as file:
                reader = csv.reader(file)
                for row in reader:
                    rows.append(row)
    return rows

if __name__ == "__main__": 

    model_file = read_morphology(path_model_morphology) # read model csv file as list
    link_lengths = np.array(model_file[1], dtype=float) # index link lengths from the file
    #print(link_lengths)
    
    folder = experiment_config['data_folder'] #MORL
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + '_test_' + model_checkpoint_num 
    experiment_config['data_folder_experiment'] = file_str # MORL

    #Create directory when not using video recording, turn off when you do, sloppy I know
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    #initiliaze the class
    coadapt_test = coadapt.Coadaptation(experiment_config, weight_index, project_name, run_name)
    #load the networks
    coadapt_test.load_networks(path_model_checkpoint)
    #filepath
    file_path = coadapt_test._config['data_folder_experiment']
    #iterations
    n = 30
    #print(coadapt_test._env.get_current_design())
    coadapt_test._env.set_new_design(link_lengths) # Set new link lenghts
    #print(coadapt_test._env.get_current_design())
    with open(
            os.path.join(file_path,
                'episodic_rewards_run_{}.csv'.format(run_name)
                ), 'w') as fd:
        #simulate the model
        for i in range(n):
            cwriter = csv.writer(fd)
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()
            cwriter.writerow([coadapt_test._data_reward_1[0], coadapt_test._data_reward_2[0]])
            print(f"Iteration done: {i}, Progress: {round((i/n)*100)}%")
    
    wandb.finish()