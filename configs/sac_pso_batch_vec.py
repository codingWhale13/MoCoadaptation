from copy import deepcopy

from configs import base_config


sac_pso_batch_vec = deepcopy(base_config)
sac_pso_batch_vec["config_name"] = "sac_pso_batch"
sac_pso_batch_vec["data_folder"] = "/scratch/work/kielen1/experiments/sac_pso_batch/"
sac_pso_batch_vec["design_optim_method"] = "pso_batch"
sac_pso_batch_vec["use_vector_q"] = True
