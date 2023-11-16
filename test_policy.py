import coadapt
import experiment_configs

import os, sys, time
import hashlib
import json
import wandb
import csv


#path='/home/oskar/Thesis/checkpoint_design_60' #data.pkl
#path='/home/oskar/Thesis/Results_scalarized/results_with_wandb/Wed_Oct_18_18:29:14_2023__49c4bc4c/checkpoints/checkpoint_design_60.chk' # works
path='/home/oskar/Thesis/Results_scalarized/results_with_wandb/Wed_Oct_18_18:29:24_2023__a18577fe_1_0/checkpoints/checkpoint_design_60.chk' 
experiment_config = experiment_configs.sac_pso_sim #MORL dictiornary need for batch or sim
weight_index = 8 #Set as what the data was for # in this case 8
project_name = "test"#"coadapt-testing-scaling-tests"
run_name = "test"#"1, 0, 7"

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
    
    with open(
            os.path.join(file_path,
                'episodic_rewards_run_{}.csv'.format(run_name)
                ), 'w') as fd:
        cwriter = csv.writer(fd)
        for i in range(200):
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()
            #run_speed_reward = coadapt_test._data_reward_1
            #energy_consumption_reward = coadapt_test._data_reward_2
            
            
            #current_design = coadapt_test._env.get_current_design()
            
            #cwriter.writerow(['Design Type:', coadapt_test._data_design_type])
            #cwriter.writerow(current_design)
            #cwriter.writerow(self._data_rewards) # SORL - needs to be changed for MORL, we have two rewards instead of one
            cwriter.writerow([coadapt_test._data_reward_1[0], coadapt_test._data_reward_2[0]])
            #cwriter.writerow(coadapt_test._data_reward_2)
            print(f"Iteration done: {i}, Progress: {i/200*0.5}")
    
    wandb.finish()