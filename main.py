import os, sys, time
import hashlib
import coadapt
import experiment_configs as cfg
import json
import wandb

#def main(config): # ORIG'
def main(config_name, weight_index): # MORL

    # Create foldr in which to save results
    #folder = config['data_folder'] #ORIG
    folder = config_name['data_folder'] #MORL
    #generate random hash string - unique identifier if we startexi
    # multiple experiments at the same time
    rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
    file_str = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id
    #config['data_folder_experiment'] = file_str # ORIG
    config_name['data_folder_experiment'] = file_str # MORL

    # Create experiment folder
    if not os.path.exists(file_str):
      os.makedirs(file_str)

    # Store config
    #with open(os.path.join(file_str, 'config.json'), 'w') as fd:
    #        fd.write(json.dumps(config,indent=2))                  #ORIG
    
    with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config_name,indent=2))                  #MORL
    
    #co = coadapt.Coadaptation(config) # ORIG 
    co = coadapt.Coadaptation(config_name, weight_index, project_name, run_name) # MORL
    co.run()
    wandb.finish()



if __name__ == "__main__":
    # We assume we call the program only with the name of the config we want to run
    # nothing too complex
    #if len(sys.argv) > 1:
    #    config = cfg.config_dict[sys.argv[1]]
    #else:
    #    config = cfg.config_dict['sac_pso_sim'] #['sac_pso_batch'] # for debugging the code
    #    #config = cfg.config_dict['base']
    #main(config)
    
    # MORL
    if len(sys.argv) > 4:
        config_name = cfg.config_dict[sys.argv[1]]
        weight_index = int(sys.argv[2])
        project_name = sys.argv[3]
        run_name = sys.argv[4]
        print(run_name)
        print(project_name)
        #Later on needs to changed to give you a option to give trackable names for wandb tracking
        print(f"index w : {weight_index}")
        print(config_name['weights'][weight_index])
    else:
        config_name = cfg.config_dict['sac_pso_sim']
        weight_index = 10
        project_name="coadapt-save-testing"
        run_name="default-run-weight" + f"-{config_name['weights'][weight_index]}"
        print(config_name['weights'][weight_index])
    #main(config) ORIG
    main(config_name, weight_index) # MORL