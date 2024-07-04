from copy import deepcopy

from configs import base_config


sac_pso_sim = deepcopy(base_config)
sac_pso_sim["config_name"] = "sac_pso_sim"
sac_pso_sim["data_folder"] = "/scratch/work/kielen1/experiments/sac_pso_sim/"
sac_pso_sim["design_optim_method"] = "pso_sim"
