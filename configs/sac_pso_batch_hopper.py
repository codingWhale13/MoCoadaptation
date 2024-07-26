from copy import deepcopy

from configs import sac_pso_batch


sac_pso_batch_hopper = deepcopy(sac_pso_batch)

sac_pso_batch_hopper["env"]["env_name"] = "Hopper"
sac_pso_batch_hopper["env"]["record_video"] = True  # TODO: remove this
