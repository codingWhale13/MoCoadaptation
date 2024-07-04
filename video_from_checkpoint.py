# LOADS CHECKPOINT AND SHOWS POLICY AS A VIDEO

import argparse
import json
import os

import coadapt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-dir",
        type=str,
        help="Experiment folder (should contain config.json)",
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Output folder for video",
        default="videos",
    )

    return parser.parse_args()


def generate_video(exp_dir, save_dir=None):
    # load config
    with open(os.path.join(exp_dir, "config.json")) as file:
        config = json.load(file)

    config["initial_model_dir"] = exp_dir  # enable model loading
    config["use_gpu"] = False  # no need for GPU when creating videos

    config["env"]["record_video"] = True  # enable video recording
    config["video"]["record_evy_n_episodes"] = 1  # we'll just run a single episode
    if save_dir is not None:
        config["video"]["video_save_dir"] = save_dir

    co = coadapt.Coadaptation(config)  # checkpoint is loaded in __init__

    # run an episode using the loaded model
    co.initialize_episode()
    co.execute_policy()

    # actually save the video to disk
    co._env.reset()


if __name__ == "__main__":
    args = parse_args()

    generate_video(args.exp_dir, args.save_dir)
