# LOADS CHECKPOINT AND SHOWS POLICY AS A VIDEO
import argparse

import coadapt
from utils import add_argparse_arguments, get_config, get_cycle_count


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


def generate_video(run_dir, design_cycle, common_name, save_dir):
    config = get_config(run_dir)
    if design_cycle == "last":
        design_cycle = get_cycle_count(run_dir)
    else:
        design_cycle = int(design_cycle)

    config["initial_model_dir"] = run_dir  # Enable model loading
    config["use_gpu"] = False  # No need for GPU when creating videos
    config["use_wandb"] = False  # No need for wandb when creating videos
    config["env"]["record_video"] = False  #  Done in create_video_of_episode
    config["load_replay_buffer"] = False

    # Load checkpoint (in Coadaptation.__init__) and create video of episode
    co = coadapt.Coadaptation(config, design_iter_to_load=design_cycle)
    co.create_video_of_episode(save_dir, f"{common_name}_{design_cycle}")


if __name__ == "__main__":
    args = parse_args()
    generate_video(
        run_dir=args.run_dir,
        design_cycle=args.design_cycle,
        common_name=args.common_name,
        save_dir=args.save_dir,
    )
