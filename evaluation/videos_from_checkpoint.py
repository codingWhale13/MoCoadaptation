import argparse
import os
import subprocess
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import add_argparse_arguments, exp_dir_to_run_dirs, load_config


def parse_args():
    parser = argparse.ArgumentParser()

    add_argparse_arguments(
        parser,
        [
            ("exp-dir", True),
            ("design-cycle", False),
            ("common-name", False),
            ("save-dir", False),
        ],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_dirs = exp_dir_to_run_dirs(args.exp_dir)
    if args.design_cycle is None:
        design_cycles = [1, 10, 30, "last"]
    else:
        design_cycles = [args.design_cycle]
    common_name = args.common_name

    videos_want = len(design_cycles) * len(run_dirs)
    videos_got = 0

    for run_dir in run_dirs:
        if args.common_name is None:
            common_name = load_config(run_dir)["run_id"]

        for design_cycle in design_cycles:
            command = [
                "python",
                "video_from_checkpoint.py",
                "--run-dir",
                run_dir,
                "--design-cycle",
                str(design_cycle),
                "--common-name",
                common_name,
            ]
            if args.save_dir is not None:
                command.extend(["--save-dir", os.path.join(args.save_dir, common_name)])

            videos_got += 1
            print(f"\nCreate video {videos_got}/{videos_want}, `{' '.join(command)}`")
            subprocess.run(command)
