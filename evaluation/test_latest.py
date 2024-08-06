import argparse
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.test_helper import create_test_run
from utils import (
    ORIGINAL_CONFIG,
    add_argparse_arguments,
    exp_dir_to_run_dirs,
    load_config,
    save_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    add_argparse_arguments(
        parser,
        [
            ("exp-dir", False),  # Option for convenience
            ("run-dirs", False),  # Option for more control
            ("n-tests", 5),  # Number of test runs per final model
            ("n-iters", 30),  # Number of iterations per test
            ("save-dir", "../test_data/latest"),
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine run_dirs to be tested
    if args.exp_dir is None and args.run_dirs is None:
        raise ValueError("Either --exp-dir or --run-dirs must be specified")
    run_dirs = [] if args.run_dirs is None else args.run_dirs
    if args.exp_dir is not None:
        run_dirs.extend(exp_dir_to_run_dirs(args.exp_dir))

    run_dirs = exp_dir_to_run_dirs(args.exp_dir)
    for run_number, run_dir in enumerate(run_dirs):
        print(f"Testing directory {run_number + 1}/{len(run_dirs)}")

        # Load config and save it for reference
        config = load_config(run_dir)
        test_dir_name = f"{config['run_name']}_{config['run_id']}"
        test_dir = os.path.join(args.save_dir, test_dir_name)
        save_config(config, test_dir, ORIGINAL_CONFIG)

        for test_number in range(1, args.n_tests + 1):
            create_test_run(
                run_dir,
                design_iter_to_load="last",
                n_iters=args.n_iters,
                save_dir=test_dir,
                filename=f"episodic_rewards_run_{test_number}",
            )
