import argparse
import subprocess

from utils import add_argparse_arguments, exp_dir_to_run_dirs, get_config


def parse_args():
    parser = argparse.ArgumentParser()

    add_argparse_arguments(
        parser,
        [
            ("run-dir", True),
            ("design-cycle", False),
            ("common-name", False),
            ("save-dir", False),
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dirs = exp_dir_to_run_dirs(args.run_dir)
    for run_dir in run_dirs:
        common_name = (
            get_config(run_dir)["run_name"]
            if args.common_name is None
            else args.common_name
        )

        design_cycles = (
            [1, 10, 30, "last"] if args.design_cycle is None else [args.design_cycle]
        )
        for design_cycle in design_cycles:
            command = [
                "python",
                "video_from_checkpoint.py",
                "--run-dir",
                run_dir,
                "--design-cycle",
                str(design_cycle),
            ]

            command.extend(["--common-name", common_name])

            if args.save_dir is not None:
                command.extend(["--save-dir", args.save_dir])

            print(f"\nRUNNING COMMAND {command}\n")
            subprocess.run(command)
