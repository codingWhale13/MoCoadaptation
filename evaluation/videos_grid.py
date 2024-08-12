import argparse
import ffmpeg
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import add_argparse_arguments


def parse_args():
    parser = argparse.ArgumentParser()

    add_argparse_arguments(
        parser,
        [
            ("video-dirs", True),
            ("common-name", "episode_video"),
            ("save-dir", "videos/grid"),
        ],
    )

    return parser.parse_args()


def add_text_to_video(input_file, output_file, text):
    # Add text to the top of the video
    (
        ffmpeg.input(input_file)
        .drawtext(
            text=text,
            fontfile="fonts/dejavu-sans-bold.ttf",
            fontsize=10,
            fontcolor="black",
            x="(w-text_w)/2",
            y=10,
            borderw=2,
            bordercolor="white",
        )
        .output(output_file)
        .run(overwrite_output=True)
    )


def stack_videos_horizontally(video_files, output_file):
    # Concatenate videos horizontally
    inputs = [ffmpeg.input(video) for video in video_files]
    ffmpeg.filter(inputs, "hstack", inputs=len(inputs)).output(
        output_file,
        vcodec="libx264",  # Set video codec
        pix_fmt="yuv420p",  # Set pixel format
    ).run(overwrite_output=True)


def stack_videos_vertically(video_files, output_file):
    # Concatenate videos vertically
    inputs = [ffmpeg.input(video) for video in video_files]
    ffmpeg.filter(inputs, "vstack", inputs=len(inputs)).output(output_file).run(
        overwrite_output=True
    )


def sort_filenames_by_design_iter(filename):
    if filename.endswith("last.mp4"):
        return float("inf")
    return int(filename.split("_")[-1][:-4])


def process_folder(folder_path, temp_folder):
    temp_files = []
    for f in sorted(os.listdir(folder_path), key=sort_filenames_by_design_iter):
        video_path = os.path.join(folder_path, f)
        temp_file = os.path.join(temp_folder, f"temp_{f}")

        run_id, design_cycle = f.split("_")
        design_cycle = design_cycle[:-4]
        text = f"{run_id} @ design cycle {design_cycle}"

        add_text_to_video(video_path, temp_file, text)
        temp_files.append(temp_file)

    return temp_files


def create_video_grid(video_dirs, common_name):
    # Determine amound of n_rows and n_cols needed for video grid
    n_rows = len(video_dirs)
    n_cols = None
    for video_folder in video_dirs:
        video_count_in_folder = len(os.listdir(video_folder))
        if n_cols is None:
            n_cols = video_count_in_folder
        elif n_cols != video_count_in_folder:
            raise ValueError(
                "Video folders don't all contain the same amount of videos"
            )

    print(f"VIDEO GRID SIZE: {n_rows}x{n_cols}")

    temp_folder = "temp_videos"
    os.makedirs(temp_folder, exist_ok=True)

    all_temp_files = []
    for folder in video_dirs:
        temp_files = process_folder(folder, temp_folder)
        all_temp_files.extend(temp_files)

    row_files = []
    for i in range(n_rows):
        col_files = all_temp_files[i * n_cols : (i + 1) * n_cols]
        row_path = os.path.join(temp_folder, f"row_{i}.mp4")
        stack_videos_horizontally(col_files, row_path)
        row_files.append(row_path)

    stack_videos_vertically(row_files, f"{common_name}.mp4")

    # Clean up temporary files
    for temp_file in all_temp_files + row_files:
        os.remove(temp_file)
    os.rmdir(temp_folder)


if __name__ == "__main__":
    args = parse_args()

    create_video_grid(video_dirs=args.video_dirs, common_name=args.common_name)
