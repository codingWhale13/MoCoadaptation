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
            fontsize=16,
            fontcolor="black",
            x="(w-text_w)/2",
            y=10,
            borderw=2,
            bordercolor="white",
        )
        .output(output_file)
        .run(overwrite_output=True)
    )


def concatenate_videos_horizontally(video_files, output_file):
    # Concatenate videos horizontally
    inputs = [ffmpeg.input(video) for video in video_files]
    print("\n\n\n\n\n\n\n\n\n\n\n\nHSTACK", inputs)
    ffmpeg.filter(
        inputs,
        "hstack",
        inputs=4,
    ).output(
        output_file,
        vcodec='libx264',       # Set video codec
        pix_fmt='yuv420p',      # Set pixel format
    ).run(overwrite_output=True)


def concatenate_videos_vertically(video_files, output_file):
    # Concatenate videos vertically
    inputs = [ffmpeg.input(video) for video in video_files]
    ffmpeg.filter(inputs, "vstack").output(
        output_file,
    ).run(overwrite_output=True)


def process_folder(folder_path, temp_folder):
    temp_files = []
    for f in sorted(os.listdir(folder_path), key=lambda x: int(x.split("_")[-1][:-4])):
        video_path = os.path.join(folder_path, f)
        temp_file = os.path.join(temp_folder, f"temp_{f}")
        add_text_to_video(video_path, temp_file, folder_path)
        temp_files.append(temp_file)

    return temp_files


def create_video_grid(video_dirs, common_name):
    # Determine amound of rows and cols needed for video grid
    rows = len(video_dirs)
    cols = None
    for video_folder in video_dirs:
        video_count_in_folder = len(os.listdir(video_folder))
        if cols is None:
            cols = video_count_in_folder
        elif cols != video_count_in_folder:
            raise ValueError(
                "Video folders don't all contain the same amount of videos"
            )

    print(f"VIDEO GRID SIZE: {rows}x{cols}")

    temp_folder = "temp_videos"
    os.makedirs(temp_folder, exist_ok=True)

    all_temp_files = []
    for folder in video_dirs:
        temp_files = process_folder(folder, temp_folder)
        all_temp_files.extend(temp_files)

    row_files = []
    for i in range(rows):
        col_files = all_temp_files[i * cols : (i + 1) * cols]
        row_path = os.path.join(temp_folder, f"row_{i}.mp4")
        concatenate_videos_horizontally(col_files, row_path)
        row_files.append(row_path)

    # ==============

    concatenate_videos_vertically(row_files, f"{common_name}.mp4")

    # Clean up temporary files
    # for temp_file in all_temp_files + row_files:
    #    os.remove(temp_file)
    # os.rmdir(temp_folder)


if __name__ == "__main__":
    args = parse_args()

    create_video_grid(video_dirs=args.video_dirs, common_name=args.common_name)
