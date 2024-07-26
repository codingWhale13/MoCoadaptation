import os

import cv2
import numpy as np

import pybullet as p


class SimpleVideoRecorder:
    def __init__(self, env, save_dir="", file_name="single_episode"):
        self._env = env
        self._frame_width = 200
        self._frame_height = 200

        # SET UP THINGS
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._vid_writer = cv2.VideoWriter(
            os.path.join(save_dir, f"{file_name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (self._frame_width, self._frame_height),
        )

        frame = env.render_camera_image((self._frame_width, self._frame_height))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._vid_writer.write(frame)

        # connect to PyBullet server (for some reason important when recording videos with only a few frames)
        if p.isConnected() == 0:
            p.connect(p.GUI)

    def step(self):
        self._env.camera_adjust()
        frame = self._env.render_camera_image((self._frame_width, self._frame_height))
        frame = frame * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._vid_writer.write(frame)

    def save_video(self):
        self._env.camera_adjust()
        self._vid_writer.release()
