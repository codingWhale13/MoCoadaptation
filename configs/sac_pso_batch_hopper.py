from copy import deepcopy

from configs import base_config


sac_pso_batch_hopper = deepcopy(base_config)
sac_pso_batch_hopper["config_id"] = "sac_pso_batch_hopper"
sac_pso_batch_hopper["data_folder"] = "experiments/sac_pso_batch/hopper"
sac_pso_batch_hopper["design_optim_method"] = "pso_batch"
sac_pso_batch_hopper["env"]["env_name"] = "Hopper"
