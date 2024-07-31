from copy import deepcopy

from configs import base_config


sac_pso_batch_walker2d = deepcopy(base_config)
sac_pso_batch_walker2d["config_name"] = "sac_pso_batch_walker2d"
sac_pso_batch_walker2d["data_folder"] = "experiments/sac_pso_batch/walker2d"
sac_pso_batch_walker2d["design_optim_method"] = "pso_batch"
sac_pso_batch_walker2d["env"]["env_name"] = "Walker2d"
