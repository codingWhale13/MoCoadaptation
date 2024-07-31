from copy import deepcopy

from configs import base_config


sac_pso_batch_halfcheetah = deepcopy(base_config)
sac_pso_batch_halfcheetah["config_name"] = "sac_pso_batch_halfcheetah"
sac_pso_batch_halfcheetah["data_folder"] = "experiments/sac_pso_batch/halfcheetah"
sac_pso_batch_halfcheetah["design_optim_method"] = "pso_batch"
sac_pso_batch_halfcheetah["env"]["env_name"] = "HalfCheetah"
