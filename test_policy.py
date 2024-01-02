import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb
import csv
import numpy as np

# put path to folder of model here, seed, weight and model folder name, aka last three parts from path
# example -> set_seed/0.0_1.0/Thu_Dec__7_20:55:59_2023__0f1677df[0.0, 1.0]

<<<<<<< Updated upstream
path_to_folder = '/home/oskar/Thesis/Model_scalarized/results_with_rescaling/set_seed/0.8_0.2/Thu_Dec__7_20:56:48_2023__878baf28[0.8, 0.2]' 

=======
path_to_folder = '/home/oskar/Thesis/inter/models/results_with_rescaling/set_seed/0.7_0.3/Fri_Dec_29_22:47:42_2023__1186cf06[0.7, 0.3]_1' 

seed = path_to_folder[-1]
>>>>>>> Stashed changes
newline=''

experiment_config = experiment_configs.sac_pso_sim #MORL dictiornary need for batch or sim
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

if __name__ == "__main__": 

    #Use to get last model checkpoint
    last_model_checkpoint_num = find_checkpoint(path_to_folder)#-1 # checkpoint
    last_model_checkpoint = f'checkpoint_design_{last_model_checkpoint_num}.chk'
    print("path_to_folder:", path_to_folder)
    print("last_model_checkpoint_num:", last_model_checkpoint_num)
    last_model_checkpoint = os.path.join(path_to_folder, 'checkpoints', last_model_checkpoint) # Set model path for correct checkpoint
    
    print(f"Last model checkpoint: {last_model_checkpoint}")
    
    #Load morphology -> link lenghts
    morphology_number = str(last_model_checkpoint_num)  + ".csv"
    model_file = read_morphology(path_to_folder, morphology_number) # read model csv file as list
    #print(model_file)
    link_lengths = np.array(model_file[1], dtype=float) # index link lengths from the file
    print(f"Link lenghts: {link_lengths}")
    
    folder = experiment_config['data_folder'] #MORL
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
<<<<<<< Updated upstream
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id + '_test_' + str(last_model_checkpoint_num)
=======
    model_name = path_to_folder.split('/')[-1:]
    model_name = str(model_name[0])
    file_str = './' + folder + '/' + '__' + rand_id + '_' + model_name + '_test_' + str(last_model_checkpoint_num) + "_" + str(seed)
>>>>>>> Stashed changes
    experiment_config['data_folder_experiment'] = file_str # MORL

    #Create directory when not using video recording, turn off when you do, sloppy I know
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    #initiliaze the class
    coadapt_test = coadapt.Coadaptation(experiment_config, weight_index, project_name, run_name)
    #load the networks
    coadapt_test.load_networks(last_model_checkpoint)
    file_path = coadapt_test._config['data_folder_experiment'] #filepath
    n = 30 #iterations
    #print(coadapt_test._env.get_current_design())
    coadapt_test._env.set_new_design(link_lengths) # Set new link lenghts
    #print(coadapt_test._env.get_current_design())
    with open(
            os.path.join(file_path,
                'episodic_rewards_run_{}.csv'.format(run_name)
                ), 'w') as fd:
        #simulate the model
        running_speed = []
        energy_saving = []
        for i in range(n):
            cwriter = csv.writer(fd)
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()
            
            # Quick note, it would be more efficient to have the results within the link lenghts themselves within results and write everything on 3 rows
            # but that would make the previous test run files obsolete since you couldnt read the easily in the same way. This should be the way things are done later on.
            #cwriter.writerow([coadapt_test._data_reward_1[0], coadapt_test._data_reward_2[0]]) # original
            
            #append iteration results to lists
            running_speed.append(coadapt_test._data_reward_1[0])
            energy_saving.append(coadapt_test._data_reward_2[0])
            print(f"Iteration done: {i}, Progress: {round((i/n)*100)}%")
        #save results to csv file
        cwriter.writerow(link_lengths)
        cwriter.writerow(running_speed)
        cwriter.writerow(energy_saving)
        
    wandb.finish()