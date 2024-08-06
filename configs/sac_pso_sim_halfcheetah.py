from copy import deepcopy

from configs import base_config


sac_pso_sim_halfcheetah = deepcopy(base_config)
sac_pso_sim_halfcheetah["config_id"] = "sac_pso_sim_halfcheetah"
sac_pso_sim_halfcheetah["data_folder"] = "experiments/sac_pso_sim/halfcheetah"
sac_pso_sim_halfcheetah["design_optim_method"] = "pso_sim"
sac_pso_sim_halfcheetah["env"]["env_name"] = "HalfCheetah"
