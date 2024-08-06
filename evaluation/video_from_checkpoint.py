import argparse
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coadapt import Coadaptation
from utils import add_argparse_arguments, load_config, get_cycle_count


def parse_args():
    parser = argparse.ArgumentParser()

    add_argparse_arguments(
        parser,
        [
            ("run-dir", True),
            ("design-cycle", "last"),
            ("common-name", "episode_video"),
            ("save-dir", "videos"),
        ],
    )

    return parser.parse_args()


def create_video(run_dir, design_cycle, common_name, save_dir):
    config = load_config(run_dir)
    config["initial_model_dir"] = run_dir  # Enable model loading
    config["use_gpu"] = False  # No need for GPU
    config["use_wandb"] = False  # No need for wandb
    config["old_replay_portion"] = 0  # No need for replay buffer
    config["env"]["record_video"] = False  # Instead: see create_video_of_episode

    # Load checkpoint (happens in Coadaptation.__init__) and create video of episode
    co = Coadaptation(config, design_iter_to_load=design_cycle)
    co.create_video_of_episode(save_dir, filename=f"{common_name}_{design_cycle}")


if __name__ == "__main__":
    args = parse_args()
    create_video(
        run_dir=args.run_dir,
        design_cycle=args.design_cycle,
        common_name=args.common_name,
        save_dir=args.save_dir,
    )
