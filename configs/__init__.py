from configs.base_config import base_config
from configs.sac_pso_batch_halfcheetah import sac_pso_batch_halfcheetah
from configs.sac_pso_sim_halfcheetah import sac_pso_sim_halfcheetah
from configs.sac_pso_batch_walker2d import sac_pso_batch_walker2d
from configs.sac_pso_batch_hopper import sac_pso_batch_hopper


# wrap all configs in one dictionary for convenience
all_configs = {
    "base_config": base_config,
    "sac_pso_batch_halfcheetah": sac_pso_batch_halfcheetah,
    "sac_pso_sim_halfcheetah": sac_pso_sim_halfcheetah,
    "sac_pso_batch_walker2d": sac_pso_batch_walker2d,
    "sac_pso_batch_hopper": sac_pso_batch_hopper,
}

__all__ = [
    all_configs,
    sac_pso_batch_halfcheetah,
    sac_pso_sim_halfcheetah,
    sac_pso_batch_walker2d,
    sac_pso_batch_hopper,
]
