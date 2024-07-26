from configs.base_config import base_config
from configs.sac_pso_batch import sac_pso_batch
from configs.sac_pso_sim import sac_pso_sim
from configs.sac_pso_batch_vec import sac_pso_batch_vec
from configs.sac_pso_batch_walker2d import sac_pso_batch_walker2d
from configs.sac_pso_batch_hopper import sac_pso_batch_hopper


# wrap all configs in one dictionary for convenience
all_configs = {
    "base_config": base_config,
    "sac_pso_batch": sac_pso_batch,
    "sac_pso_sim": sac_pso_sim,
    "sac_pso_batch_vec": sac_pso_batch_vec,
    "sac_pso_batch_walker2d": sac_pso_batch_walker2d,
    "sac_pso_batch_hopper": sac_pso_batch_hopper,
}

__all__ = [
    all_configs,
    sac_pso_batch,
    sac_pso_sim,
    sac_pso_batch_vec,
    sac_pso_batch_walker2d,
    sac_pso_batch_hopper,
]
