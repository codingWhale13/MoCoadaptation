from copy import deepcopy

from configs import sac_pso_batch


sac_pso_batch_walker2d = deepcopy(sac_pso_batch)

sac_pso_batch_walker2d["env"]["env_name"] = "Walker2d"
