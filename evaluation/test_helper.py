import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coadapt import Coadaptation
from utils import (
    DESIGN_PARAMETERS,
    DESIGN_TYPE,
    DO_CHECKPOINTS,
    load_config,
    get_cycle_count,
    load_csv,
    save_csv,
)


def create_test_run(run_dir, design_iter_to_load, n_iters, save_dir, filename):
    """Executes `n_iters` test episodes and logs the rewards as CSV"""

    # Modify config before test runs (no replay, wandb, or GPU needed for testing)
    config = load_config(run_dir)
    config["old_replay_portion"] = 0
    config["use_wandb"] = False
    config["use_gpu"] = False
    config["initial_model_dir"] = run_dir

    # Create agent with design and model loaded from checkpoint
    coadapt_test = Coadaptation(config, design_iter_to_load=design_iter_to_load)
    env = coadapt_test.env

    # Run test iterations
    coadapt_test.initialize_episode()
    for _ in range(n_iters):
        coadapt_test.execute_policy()

    # Load design info (NOTE: The training rewards in run_dir are *not* used)
    filename_design = f"data_design_{get_cycle_count(run_dir)}.csv"
    design = load_csv(os.path.join(run_dir, DO_CHECKPOINTS, filename_design))

    # Save test result as CSV
    save_csv(
        save_dir=save_dir,
        filename=filename,
        design_type=design[DESIGN_TYPE],
        design_parameters=list(design[DESIGN_PARAMETERS]),
        rewards={
            env.reward_names[i]: [reward[i] for reward in coadapt_test.rewards]
            for i in range(env.reward_dim)
        },
    )
