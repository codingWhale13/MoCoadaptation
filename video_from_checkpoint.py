# LOADS CHECKPOINT AND SHOWS POLICY AS A VIDEO

import argparse
import json
import os

import coadapt
from utils import add_argparse_arguments


def parse_args():
    parser = argparse.ArgumentParser()

    parser = add_argparse_arguments(
        parser,
        [
            ("data-folder", True),
            ("save-dir", False),
        ],
    )

    return parser.parse_args()


def generate_video(data_folder, save_dir=None):
    # load config
    with open(os.path.join(data_folder, "config.json")) as file:
        config = json.load(file)

    config["initial_model_dir"] = data_folder  # enable model loading
    config["use_gpu"] = False  # no need for GPU when creating videos
    config["use_wandb"] = False  # no need for wandb when creating videos

    config["env"]["record_video"] = True  # enable video recording
    config["video"]["record_evy_n_episodes"] = 1  # we'll just run a single episode
    if save_dir is not None:
        config["video"]["video_save_dir"] = save_dir

    # temporary stuff for backward compatibility
    config["use_vector_q"] = config["use_vector_Q"]
    config["scalarize_before_q_loss"] = False
    config["condition_on_preference"] = False

    co = coadapt.Coadaptation(config)  # checkpoint is loaded in __init__

    # run an episode using the loaded model
    co.initialize_episode()
    co.execute_policy()

    # actually save the video to disk
    co._env.reset()


if __name__ == "__main__":
    args = parse_args()

    generate_video(args.data_folder, args.save_dir)
