import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb
import csv


#path='/home/oskar/Thesis/checkpoint_design_60' #data.pkl
#path='/home/oskar/Thesis/Results_scalarized/results_with_wandb/Wed_Oct_18_18:29:14_2023__49c4bc4c/checkpoints/checkpoint_design_60.chk' # works
path='/home/oskar/Thesis/Results_scalarized/results_with_rescaling/Thu_Nov_16_21:54:23_2023__cbea1a7e_1._.0/checkpoints/checkpoint_design_46.chk' 
experiment_config = experiment_configs.sac_pso_sim #MORL dictiornary need for batch or sim
weight_index = 8 #Set as what the data was for # in this case 8
project_name = "test" #"coadapt-scaled-test"#"coadapt-testing-scaling-tests"
run_name = "test" #"0, 1, 5"#"1, 0, 7"

if __name__ == "__main__": 

    folder = experiment_config['data_folder'] #MORL
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id
    experiment_config['data_folder_experiment'] = file_str # MORL

    #Create directory when not using video recording, turn off when you do, sloppy I know
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    #initiliaze the class
    coadapt_test = coadapt.Coadaptation(experiment_config, weight_index, project_name, run_name)
    #load the networks
    coadapt_test.load_networks(path)
    #simulate the model
    
    file_path = coadapt_test._config['data_folder_experiment']
    
    #iterations
    n = 30
    
    print(coadapt_test._data_design_type)
    print(coadapt_test._env.get_current_design())
    #coadapt_test._env.load_design_dimensions([0.1, 1, 1, 1, 1, 1])
    #coadapt_test._env.load_design_dimensions([1.16513724, 0.83658113, 1.11166463, 0.96295299, 0.83459136, 1.27074335])
    coadapt_test._env.load_design_dimensions([2, 2, 2, 2, 2, 2])
    #coadapt_test._env.load_design_dimensions([0, 0, 0, 0, 0, 0])
    #print(coadapt_test._env.get_current_design())
    
    with open(
            os.path.join(file_path,
                'episodic_rewards_run_{}.csv'.format(run_name)
                ), 'w') as fd:
        
        for i in range(n):
            cwriter = csv.writer(fd)
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()
            cwriter.writerow([coadapt_test._data_reward_1[0], coadapt_test._data_reward_2[0]])
            print(f"Iteration done: {i}, Progress: {(i/n)*100}")
    
    wandb.finish()