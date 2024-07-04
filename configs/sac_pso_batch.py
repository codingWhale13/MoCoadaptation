from copy import deepcopy

from configs import base_config


sac_pso_batch = deepcopy(base_config)
sac_pso_batch["config_name"] = "sac_pso_batch"
sac_pso_batch["data_folder"] = "/scratch/work/kielen1/experiments/sac_pso_batch/"
sac_pso_batch["design_optim_method"] = "pso_batch"
