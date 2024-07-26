import argparse
import json
import os
import csv

import numpy as np

import coadapt
from utils import (
    add_argparse_arguments,
    exp_dir_to_run_dirs,
    get_config,
    get_cycle_count,
    read_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()

    add_argparse_arguments(
        parser,
        [
            ("exp-dir", False),  # Option for convenience
            ("run-dirs", False),  # Option for more control
            ("skip-initial-designs", "true"),
            ("skip-random-designs", "true"),
            ("n-tests", 1),  # Number of test runs for each iteration
            ("n-iters", 30),  # Number of test iterations
            ("save-dir", "test_data/per_iteration"),
        ],
    )

    return parser.parse_args()


def find_checkpoint(path_to_directory):
    """Find the checkpoint for the model

    Returns: returns the int value of the last checkpoint or None
    """
    checkpoints = []
    for file in os.listdir(path_to_directory):
        if file.endswith(".csv"):
            checkpoint = int(file.split("_")[-1][:-4])
            checkpoints.append(checkpoint)
    if checkpoints:
        return max(checkpoints)
    else:
        return None


def read_morphology(run_dir) -> list:
    """Returns a list of values read from CSV file per row"""

    rows = []
    filename = f"data_design_{get_cycle_count(run_dir)}.csv"
    filepath = os.path.join(run_dir, "do_checkpoints", filename)
    with open(filepath, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)

    design_type = rows[0]
    morphology = np.array(rows[1], dtype=float)

    return design_type, morphology


def create_test_run(
    run_dir: str,
    design_cycle: int,
    test_run_id: int,
    n_iters: int,
    save_dir: str,
):
    # Load config and modify it before test runs
    config = get_config(run_dir)
    config["load_replay_buffer"] = False  # Determinstic policy -> no replay buffer
    config["use_wandb"] = False
    config["use_gpu"] = False
    config["initial_model_dir"] = run_dir
    config["data_folder"] = save_dir

    # Create agent with design and RL checkpoint loaded from
    print(f"LOADING {design_cycle}")
    coadapt_test = coadapt.Coadaptation(config, design_iter_to_load=design_cycle)

    # Run test iterations
    coadapt_test.initialize_episode()
    coadapt_test.execute_policy()
    for _ in range(n_iters):
        coadapt_test.initialize_episode()
        coadapt_test.execute_policy()

    # Run test iterations
    test_iteration_count = 30
    filename = f"episodic_rewards_run_{design_cycle}_{test_run_id}.csv"
    with open(os.path.join(save_dir, filename), "w") as file:
        running_speed = []
        energy_saving = []

        for _ in range(test_iteration_count):
            cwriter = csv.writer(file)
            coadapt_test.initialize_episode()
            coadapt_test.execute_policy()

            # Append iteration results to lists
            running_speed.append(coadapt_test._rewards[-1][0])
            energy_saving.append(coadapt_test._rewards[-1][1])

        # Save results to CSV file
        design_type, link_lengths = read_morphology(run_dir)
        cwriter.writerow(design_type)
        cwriter.writerow(link_lengths)
        cwriter.writerow(running_speed)
        cwriter.writerow(energy_saving)


if __name__ == "__main__":
    args = parse_args()

    if args.exp_dir is None and args.run_dirs is None:
        raise ValueError("Either --exp-dir or --run-dirs must be specified")

    run_dirs = [] if args.run_dirs is None else args.run_dirs
    if args.exp_dir is not None:
        run_dirs.extend(exp_dir_to_run_dirs(args.exp_dir))

    save_dir = args.save_dir

    for run_number, run_dir in enumerate(run_dirs):
        print(f"Creating test runs (directory {run_number + 1}/{len(run_dirs)})...")

        if args.save_dir == "test_data/per_iteration" and run_number + 1 < 12:
            continue

        # Load config and save it for reference
        config = get_config(run_dir)
        test_run_dir = os.path.join(save_dir, f"{config['run_name']}_TEST_PER_ITER")
        if not os.path.exists(test_run_dir):
            os.makedirs(test_run_dir)
        with open(os.path.join(test_run_dir, "original_config.json"), "w") as file:
            file.write(json.dumps(config, indent=2))

        cycle_count = get_cycle_count(run_dir)
        for design_cycle in range(1, cycle_count + 1):
            rows = read_csv(
                os.path.join(
                    run_dir, "do_checkpoints", f"data_design_{design_cycle}.csv"
                )
            )

            design_type = rows[0][1]
            if (args.skip_initial_designs and design_type.lower() == "initial") or (
                args.skip_random_designs and design_type.lower() == "random"
            ):
                continue

            for run_id in range(1, args.n_tests + 1):
                create_test_run(
                    run_dir,
                    design_cycle=design_cycle,
                    test_run_id=run_id,
                    n_iters=args.n_iters,
                    save_dir=test_run_dir,
                )
