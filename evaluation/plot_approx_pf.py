import argparse
import os
import sys


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
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
            ("steered-test-dirs", False),
            ("marker-size", 100),
            ("common-name", "Default"),
            ("save-dir", "plots/approx_pf"),
        ],
    )

    return parser.parse_args()


def read_reward_data(direct_test_dir, steered_test_dirs):
    """Returns mean and std of rewards

    Expected input structure
    - test_results_dir
        - run_name_1
            - original_config.json
            - episodic_rewards_run_1.csv
            - episodic_rewards_run_2.csv
            - ...
        - ...

    Return format:
    rewards_mean = {
        {target preference}: {
            {source preference or "Baseline ({pref})"}: {
                {run with some seed}: {
                    {reward_name_1}: {some value},
                    {reward_name_2}: {some other value},
                    ...
                }
            },
        },
    }.
    """

    # Add test data of direct run ("baseline")
    config = load_config(direct_test_dir, ORIGINAL_CONFIG)
    run_id = config["run_id"]
    target_pref = pref2str(config["weight_preference"])
    source_pref = f"Baseline ({target_pref})"

    csv_files = [f for f in os.listdir(direct_test_dir) if f.endswith(".csv")]
    reward_names = get_reward_names_from_csv(
        os.path.join(direct_test_dir, csv_files[0])
    )
    rewards = {reward_name: [] for reward_name in reward_names}
    for filename in csv_files:
        csv_data = load_csv(os.path.join(direct_test_dir, filename))
        for reward_name in reward_names:
            rewards[reward_name].extend(csv_data[reward_name])

    rewards_mean = {target_pref: {source_pref: {run_id: {}}}}
    for reward_name in reward_names:
        reward_mean = np.mean(rewards[reward_name])
        rewards_mean[target_pref][source_pref][run_id][reward_name] = reward_mean

    # Add test data of steered runs
    for test_dir in steered_test_dirs:
        config = load_config(test_dir, ORIGINAL_CONFIG)
        if config["initial_model_dir"] is None:
            raise ValueError(f"Missing initial model in steered run '{run_id}'")
        initial_config = load_config(config["initial_model_dir"])

        run_id = config["run_id"]
        target_pref = pref2str(config["weight_preference"])
        source_pref = pref2str(initial_config["weight_preference"])

        rewards = {reward_name: [] for reward_name in reward_names}
        for filename in os.listdir(test_dir):
            if not filename.endswith(".csv"):
                continue
            csv_data = load_csv(os.path.join(test_dir, filename))
            for reward_name in reward_names:
                rewards[reward_name].extend(csv_data[reward_name])

        if target_pref not in rewards_mean:
            rewards_mean[target_pref] = {}
        if source_pref not in rewards_mean[target_pref]:
            rewards_mean[target_pref][source_pref] = {}
        if run_id not in rewards_mean[target_pref][source_pref]:
            rewards_mean[target_pref][source_pref][run_id] = {}
        for reward_name in reward_names:
            reward_mean = np.mean(rewards[reward_name])
            rewards_mean[target_pref][source_pref][run_id][reward_name] = reward_mean

    return rewards_mean, reward_names


def create_pf_scatter_plot(
    direct_test_dir: str,
    steered_test_dirs: list[str],
    marker_size: int,
    common_name: str,
    save_dir: str,
):
    rewards_mean, reward_names = read_reward_data(direct_test_dir, steered_test_dirs)

    source_prefs = set()
    reward_names_set = set()
    for target_pref in rewards_mean:
        for source_pref in rewards_mean[target_pref]:
            source_prefs.add(source_pref)
            for run_id in rewards_mean[target_pref][source_pref]:
                for reward_name in rewards_mean[target_pref][source_pref][run_id]:
                    reward_names_set.add(reward_name)

    # Turn reward dicts into numpy arrays
    assert len(reward_names) == 2, f"Only 2D supported (found {len(reward_names)} dims)"
    rewards_mean_np = np.array(
        [
            (
                rewards_mean[target_pref][source_pref][run_id][reward_names[0]],
                rewards_mean[target_pref][source_pref][run_id][reward_names[1]],
            )
            for target_pref in rewards_mean.keys()
            for source_pref in rewards_mean[target_pref].keys()
            for run_id in rewards_mean[target_pref][source_pref].keys()
        ]
    )

    # Create figure and set axes labels
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_xlabel(reward_names[0])
    ax2.set_ylabel(reward_names[1])

    # Create source-pref -> shape mapping
    marker_shapes = "o^s*pPXD<>v"
    source_pref_to_marker = {  # reverse sorting -> baseline first -> always same symbol
        i: marker_shapes[idx] for idx, i in enumerate(sorted(source_prefs)[::-1])
    }

    # Generate source->color mapping (direct->green, steered->orange)
    source_pref_to_color = {}
    for pref in source_prefs:
        if pref.lower().startswith("baseline"):
            source_pref_to_color[pref] = "green"
        else:
            source_pref_to_color[pref] = "darkorange"

    # Plot data
    max_x, max_y = 0, 0
    for index, (source_pref, target_pref, _) in enumerate(
        [
            (source_pref, target_pref, run_id)
            for target_pref in rewards_mean.keys()
            for source_pref in rewards_mean[target_pref].keys()
            for run_id in rewards_mean[target_pref][source_pref].keys()
        ]
    ):
        max_x = max(max_x, rewards_mean_np[index, 0])
        max_y = max(max_y, rewards_mean_np[index, 1])
        ax2.scatter(
            rewards_mean_np[index, 0],
            rewards_mean_np[index, 1],
            s=marker_size,
            color=source_pref_to_color[source_pref],
            marker=source_pref_to_marker[source_pref],
            alpha=0.8,
        )

    # Generate legend handles
    legend_handles = [mpatches.Patch(visible=False, label="$\\bf{Preferences}$")]
    for pref in source_pref_to_marker:
        label = pref if pref.lower().startswith("baseline") else f"Steered from {pref}"
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=source_pref_to_marker[pref],
                markerfacecolor=source_pref_to_color[pref],
                markeredgecolor=source_pref_to_color[pref],
                linestyle="",
                label=label,
            )
        )
    ax2.legend(handles=legend_handles)

    # Save plot as PDF
    ax2.set_xlim(0, max_x * 1.2)
    ax2.set_ylim(0, max_y * 1.2)
    plt.title(f"{common_name} (Approx. PF)")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"pf_{filenamify(common_name)}.pdf"
    ax2.figure.savefig(os.path.join(save_dir, filename), format="pdf")


if __name__ == "__main__":
    args = parse_args()

    create_pf_scatter_plot(
        direct_test_dir=args.direct_test_dir,
        steered_test_dirs=args.steered_test_dirs,
        marker_size=args.marker_size,
        common_name=args.common_name,
        save_dir=args.save_dir,
    )
