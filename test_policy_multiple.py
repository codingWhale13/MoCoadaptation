import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb
import csv
import numpy as np

##### IMPORTANT if you want to use this testing you cannot use the render or video capture feature in the 'experiment_configs.py' #####

#DOES NOT WORK PROPERLY WITH VIDEO SAVING#

# put path to folder of model here, seed, weight and model folder name, aka last three parts from path
# example -> set_seed/0.0_1.0/Thu_Dec__7_20:55:59_2023__0f1677df[0.0, 1.0]
# -> set_seed/<weight folder name>/<model_name>

#change <insert> to priori or inter based what you use or rename the folders '/Thesis/<insert>/ rest of the path

path_to_folder = '/home/oskar/Thesis/priori/Model_scalarized_batch/results_with_rescaling/set_seed/test' # in path_to_folder have models in folders per each unique weight

newline=''
experiment_config = experiment_configs.sac_pso_batch #MORL dictiornary need for batch or sim
weight_index = 5 # dummy value for creating the class, we dont update the network so the weight doesnt matter here
project_name = "test" #"coadapt-scaled-test"#"coadapt-testing-scaling-tests"
run_name = "test" #"0, 1, 5"#"1, 0, 7"

def find_checkpoint(path_to_directory):
    """ Find the checkpoint for the model

    Returns: returns the int value of the last checkpoint or None
    """    
    checkpoints = []
    for file in os.listdir(path_to_directory):    
        if file.endswith('.csv'):
            checkpoint = int(file.split('_')[-1][:-4])
            checkpoints.append(checkpoint)
    if checkpoints:
        return max(checkpoints)
    else:
        return None
    
    
def read_morphology(path, checkpoint) -> list:
    
    """ Returns a list of values read from cvs file per row
    
    Returns: a list containing csv file values
    """    
    rows = []
    for filename in os.listdir(path):
        if filename.endswith(checkpoint):
            filepath = os.path.join(path, filename)
            with open(filepath, newline=newline) as file:
                reader = csv.reader(file)
                for row in reader:
                    rows.append(row)
    return rows


def run_tests(tests : int, save_mode):
    for weight_folder in os.listdir(path_to_folder):
        weight_folder_path = os.path.join(path_to_folder, weight_folder)
        for model_folder in os.listdir(weight_folder_path):
            model_folder_path = os.path.join(weight_folder_path, model_folder)
            print(f"***Running tests for model: {model_folder_path} ***")
            for j in range(tests):
                testing(model_folder_path, j, save_mode)
                
                
def testing(model_path, count, save_returns_mode):
    
    save_returns = save_returns_mode
    
    #check if coadapt object exists, destroy it if it does and create a new one
    if 'coadapt_test' in locals():
        del coadapt_test
    
    #Use to get last model checkpoint
    last_model_checkpoint_num = find_checkpoint(model_path)#-1 # checkpoint
    last_model_checkpoint = f'checkpoint_design_{last_model_checkpoint_num}.chk'
    
    print("path_to_folder:", model_path)
    print("last_model_checkpoint_num:", last_model_checkpoint_num)
    
    last_model_checkpoint = os.path.join(model_path, 'checkpoints', last_model_checkpoint) # Set model path for correct checkpoint
    print(f"Last model checkpoint: {last_model_checkpoint}")
    
    #Load morphology -> link lenghts
    morphology_number = str(last_model_checkpoint_num)  + ".csv"
    model_file = read_morphology(model_path, morphology_number) # read model csv file as list
    link_lengths = np.array(model_file[1], dtype=float) # index link lengths from the file
    print(f"Link lenghts: {link_lengths}")
    
    folder = experiment_config['data_folder'] #MORL
    #rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    model_name = model_path.split('/')[-1:]
    model_name = str(model_name[0])
    #file_str = './' + folder + '/' + '__' + model_name + '_chkpnt_' + str(last_model_checkpoint_num) + '_run_' + str(count+1)
    file_str = './' + folder + '/' + model_name
    experiment_config['data_folder_experiment'] = file_str # MORL
    
    #Create directory for files
    if not os.path.exists(file_str):
        os.makedirs(file_str)
        
    #initiliaze the class
    coadapt_test = coadapt.Coadaptation(experiment_config, weight_index, project_name, run_name)
    #load the networks
    coadapt_test.load_networks(last_model_checkpoint)
    
    file_path = coadapt_test._config['data_folder_experiment'] #filepath
    n = 30 #iterations
    
    coadapt_test._env.set_new_design(link_lengths) # Set new link lenghts
    
        #csv file name
    if save_returns:
        file_name ='episodic_rewards_run_'
    else:
        file_name ='states_actions_run_'
    
    with open(
            os.path.join(file_path,
                #'episodic_rewards_run_{}.csv'.format(run_name)
                '{}{}_{}.csv'.format(file_name, run_name, str(count+1))
                ), 'w') as fd:
        #simulate the model
        
        if save_returns:
            running_speed = []
            energy_saving = []
        else:
            states = []
            actions = []
        
        for i in range(n):
            cwriter = csv.writer(fd)
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()
            #append iteration results to lists
            if save_returns:
                running_speed.append(coadapt_test._data_reward_1[0])
                energy_saving.append(coadapt_test._data_reward_2[0])
            else:
                states.append(coadapt_test._states[0])
                actions.append(coadapt_test._actions[0])
            print(f"Iteration done: {i}, Progress: {round((i/n)*100)}%")
        #save results to csv file
        #Maybe too many if's here?
        if save_returns:
            cwriter.writerow(link_lengths)
            cwriter.writerow(running_speed)
            cwriter.writerow(energy_saving)
        else:
            states_transposed = list(map(list, zip(*states)))
            actions_transposed = list(map(list, zip(*actions)))
            cwriter.writerow(["States"])
            cwriter.writerows(states_transposed)
            cwriter.writerow(["Actions"])
            cwriter.writerows(actions_transposed)
    #wandb.finish()   Needed for wandb tracking

if __name__ == "__main__": 

    # Turn to True IF you want to save episodic returns, 
    # Turn to False when saving states and actions to csv file
    save_mode = False # True to episodic returns, False to states and actions
    test = 1 # Amount of test runs done per model
    print(f"***Running {test} tests per each model***")
    run_tests(test, save_mode)
