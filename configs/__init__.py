from configs.base_config import base_config
from configs.sac_pso_batch import sac_pso_batch
from configs.sac_pso_sim import sac_pso_sim


# wrap all configs in one dictionary for convenience
all_configs = {
    "base_config": base_config,
    "sac_pso_batch": sac_pso_batch,
    "sac_pso_sim": sac_pso_sim,
}

__all__ = [
    all_configs,
    sac_pso_batch,
    sac_pso_sim,
]
