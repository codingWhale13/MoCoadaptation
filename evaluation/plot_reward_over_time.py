import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    DESIGN_TYPE,
    ORIGINAL_CONFIG,
    add_argparse_arguments,
    filenamify,
    get_reward_names_from_csv,
    load_config,
    pref2str,
    load_csv,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_argparse_arguments(
        parser,
        [
            ("direct-test-dir", True),
            ("steered-test-dirs", True),
            ("skip-initial-designs", "true"),
            ("skip-random-designs", "true"),
            ("common-name", "Default"),
            ("save-dir", "plots/reward_over_time"),
        ],
    )

    return parser.parse_args()


def get_reward_from_dir(run_dir, skip_initial_designs, skip_random_designs):
    """Return rewards for all desired and available design iterations (`x_values`)

    Expect filenames in this format: f'{some text}_{design iteration}_{test number}.csv'
    """
    x_values_set = set()
    all_rewards = {}  # Format: design_cycle -> {test_number -> reward_dict}
    csv_files = [f for f in os.listdir(run_dir) if f.endswith(".csv")]
    reward_names = get_reward_names_from_csv(os.path.join(run_dir, csv_files[0]))

    for filename in sorted(csv_files, key=lambda x: int(x.split("_")[-2])):
        csv_data = load_csv(os.path.join(run_dir, filename))
        if skip_initial_designs and csv_data[DESIGN_TYPE].lower() == "initial":
            continue  # Skip the warm-up phase
        if skip_random_designs and csv_data[DESIGN_TYPE].lower() == "random":
            continue  # Don't plot rewards of random designs => smoother plot

        design_cycle = int(filename.split("_")[-2])
        test_number = int(filename.split("_")[-1][:-4])

        x_values_set.add(design_cycle)
        if design_cycle not in all_rewards:
            all_rewards[design_cycle] = {}
        all_rewards[design_cycle][test_number] = {
            reward_name: csv_data[reward_name] for reward_name in reward_names
        }

    # Turn x-values relevant for plotting into a list
    x_values = sorted(x_values_set)

    # Take mean of rewards for each design cycle (format: reward_name -> list of mean rewards per iteration)
    rewards = {reward_name: [] for reward_name in reward_names}
    for design_cycle in x_values:
        for reward_name in reward_names:
            values = []
            for test_number in all_rewards[design_cycle].keys():
                values.extend(all_rewards[design_cycle][test_number][reward_name])
            rewards[reward_name].append(np.mean(values))

    return rewards, reward_names, x_values


def plot_reward_over_time(
    direct_test_dir: str,
    steered_test_dirs: list[str],
    skip_initial_designs: bool,
    skip_random_designs: bool,
    save_dir: str,
    common_name: str,
):
    # Get rewards of baseline (directly trained to target preference, no steering)
    rewards_baseline, reward_names, x_baseline = get_reward_from_dir(
        run_dir=direct_test_dir,
        skip_initial_designs=skip_initial_designs,
        skip_random_designs=skip_random_designs,
    )

    # Create plot
    n_steered_runs = len(steered_test_dirs)
    n_objectives = len(rewards_baseline)
    fig, axes = plt.subplots(n_steered_runs, n_objectives, figsize=(10, 10))
    fig.suptitle(f"{common_name} (Test Results per Design Cycle)")
    alpha = 0.8
    label_baseline = f"Baseline ({pref2str(load_config(direct_test_dir, ORIGINAL_CONFIG)['weight_preference'])})"

    # Plot training reward history from all steered runs and overlay baseline
    for i, steered_test_dir in enumerate(steered_test_dirs):
        rewards_steered, _, x_steered = get_reward_from_dir(
            run_dir=steered_test_dir,
            skip_initial_designs=skip_initial_designs,
            skip_random_designs=skip_random_designs,
        )

        config = load_config(steered_test_dir, ORIGINAL_CONFIG)
        initial_model_dir = config["initial_model_dir"]
        initial_config = load_config(initial_model_dir)
        label_steered = f"Steered from {pref2str(initial_config['weight_preference'])}"

        for j, reward_name in enumerate(reward_names):
            axes[i, j].plot(
                x_steered,
                rewards_steered[reward_name],
                color="darkorange",
                alpha=alpha,
                label=label_steered,
                linewidth=2,
            )
            axes[i, j].plot(
                x_baseline,
                rewards_baseline[reward_name],
                color="green",
                alpha=alpha,
                label=label_baseline,
                linewidth=2,
            )
            axes[i, j].legend()
            axes[i, j].set_ylabel(reward_name)
            if i == n_steered_runs - 1:
                axes[i, j].set_xlabel("Design Cycle")

    # Save plot as PDF
    os.makedirs(save_dir, exist_ok=True)
    filename = f"reward_over_time_{filenamify(common_name)}.pdf"
    plt.savefig(os.path.join(save_dir, filename), format="pdf")


if __name__ == "__main__":
    args = parse_args()

    plot_reward_over_time(
        direct_test_dir=args.direct_test_dir,
        steered_test_dirs=args.steered_test_dirs,
        skip_initial_designs=args.skip_initial_designs,
        skip_random_designs=args.skip_random_designs,
        save_dir=args.save_dir,
        common_name=args.common_name,
    )
