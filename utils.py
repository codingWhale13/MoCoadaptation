import cv2
import os
from shutil import copyfile, move
import time

import numpy as np

import rlkit.torch.pytorch_util as ptu


def move_to_cpu():
    """Set device to cpu for torch"""
    ptu.set_gpu_mode(False)


def move_to_cuda(cuda_device=None):
    """Set device to gpu for torch"""
    if cuda_device is None:
        cuda_device = 0
    ptu.set_gpu_mode(True, cuda_device)


def copy_pop_to_ind(networks_pop, networks_ind):
    """Function used to copy params from pop. networks to individual networks.

    The parameters of all networks in network_ind will be set to the parameters
    of the networks in networks_ind.

    Args:
        networks_pop: Dictonary containing the population networks.
        networks_ind: Dictonary containing the individual networks. These
            networks will be updated.
    """
    for key in networks_pop:
        state_dict = networks_pop[key].state_dict()
        networks_ind[key].load_state_dict(state_dict)
        networks_ind[key].eval()


def copy_network(network_to, network_from, cuda_device=None, force_cpu=False):
    """Copies networks and set them to device or cpu.

    Args:
        networks_to: Networks to which we want to copy (destination).
        networks_from: Networks from which we want to copy (source). These
            networks will be changed.
        force_cpu: Boolean, if True the destination networks will be placed on
            the cpu. If not, the current device will be used.
    """
    network_from_dict = network_from.state_dict()
    if force_cpu:
        for key, val in network_from_dict.items():
            network_from_dict[key] = val.cpu()
    else:
        move_to_cuda(cuda_device)
    network_to.load_state_dict(network_from_dict)
    if force_cpu:
        network_to = network_to.to("cpu")
    else:
        network_to.to(ptu.device)
    network_to.eval()
    return network_to


class BestEpisodesVideoRecorder:
    def __init__(self, path=None, max_videos=1, record_evy_n_episodes=5):
        self._vid_path = "/tmp/videos" if path is None else path

        self._folder_counter = 0
        self._keep_n_best = max(max_videos, 1)
        self._record_evy_n_episodes = record_evy_n_episodes

        self._frame_width = 200
        self._frame_height = 200
        self._fps_per_frame = 0

        self.increase_folder_counter()
        self._create_vid_stream()
        self._time_start = time.time()

    def _episodic_reset(self):
        self._current_episode_reward = 0
        self._did_at_least_one_step = False
        self._step_counter = 1

    def reset_recorder(self):
        self._episode_counter = 0
        # self._episodic_rewards = [-float('inf')] * self._keep_n_best # ORIG
        self._episodic_rewards = np.full(
            self._keep_n_best, -np.inf
        )  # need for numpy array since rewards has two parts and it is numpy array
        self._episodic_reset()

    def increase_folder_counter(self):
        self._current_vid_path = os.path.join(self._vid_path, str(self._folder_counter))
        self.reset_recorder()
        self._folder_counter += 1

    def step(self, env, state, reward, done):
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._current_episode_reward += reward
            env.camera_adjust()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)
            self._did_at_least_one_step = True
        proc_time = (time.time() - self._time_start) * 1000
        proc_time = 1000 / proc_time
        self._time_start = time.time()
        self._fps_per_frame += proc_time
        self._step_counter += 1

    def _do_video_file_rotation(self):
        for idx, elem in enumerate(self._episodic_rewards):
            if idx > 1:
                try:
                    move(
                        os.path.join(self._current_vid_path, f"video_{idx-1}.avi"),
                        os.path.join(self._current_vid_path, f"video_{idx-2}.avi"),
                    )
                except:
                    pass
            if (
                self._current_episode_reward < elem
            ).any():  # changed for MORL since rewards are not scalar
                # self._episodic_rewards = self._episodic_rewards[1:idx] + [self._current_episode_reward] + self._episodic_rewards[idx:] # ORIG
                self._episodic_rewards = np.concatenate(
                    [
                        self._episodic_rewards[1:idx],
                        self._current_episode_reward.ravel(),
                        self._episodic_rewards[idx:],
                    ]
                )  # Fix for dimensions

                copyfile(
                    os.path.join(self._current_vid_path, "current_video.avi"),
                    os.path.join(
                        self._current_vid_path, "video_{}.avi".format(idx - 1)
                    ),
                )
                break
            # Will only be true in last iteration and only be hit if last element is to be moved
            if idx == len(self._episodic_rewards) - 1:
                try:
                    move(
                        os.path.join(
                            self._current_vid_path, "video_{}.avi".format(idx)
                        ),
                        os.path.join(
                            self._current_vid_path, "video_{}.avi".format(idx - 1)
                        ),
                    )
                except:
                    pass
                # self._episodic_rewards = self._episodic_rewards[1:] + [self._current_episode_reward] #ORIG
                self._episodic_rewards = np.concatenate(
                    [self._episodic_rewards[1:], self._current_episode_reward.ravel()]
                )  # Fix for dimensions

                copyfile(
                    os.path.join(self._current_vid_path, "current_video.avi"),
                    os.path.join(self._current_vid_path, "video_{}.avi".format(idx)),
                )

    def reset(self, env, state, reward, done, verbose=False):
        # final processing of data from previous episode
        if self._episode_counter % self._record_evy_n_episodes == 0:
            env.camera_adjust()
            self._vid_writer.release()

            if not os.path.exists(self._current_vid_path):
                os.makedirs(self._current_vid_path)

            # if self._did_at_least_one_step and min(self._episodic_rewards) < self._current_episode_reward: # ORIG
            if self._did_at_least_one_step and np.min(self._episodic_rewards) < np.min(
                self._current_episode_reward
            ):  # ('comparison')).any(): # changed for MORL comparison for two rewards
                self._do_video_file_rotation()
            if verbose:
                print(
                    f"Average FPS of last episode: {self._fps_per_frame / self._step_counter}"
                )

        self._episode_counter += 1
        self._episodic_reset()
        # set up everything for this episode if we record
        if self._episode_counter % self._record_evy_n_episodes == 0:
            self._create_vid_stream()
            frame = env.render_camera_image((self._frame_width, self._frame_height))
            frame = frame * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._vid_writer.write(frame)

            self._time_start = time.time()
            self._fps_per_frame = 0
            self._step_counter = 1

    def _create_vid_stream(self):
        if not os.path.exists(self._current_vid_path):
            os.makedirs(self._current_vid_path)
        self._vid_writer = cv2.VideoWriter(
            os.path.join(self._current_vid_path, "current_video.avi"),
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            30,
            (self._frame_width, self._frame_height),
        )


"""
TODO (if actually needed): upgrade config from old version ("0") to proper version 1
def upgrade_config(config):
    if "config_version" in config and config["config_version"] == 1:
        return config  # nothing to do, already up to date

    config = deepcopy(config)

    config["config_version"] = 1  # upgrading to config version 1
    config["config_name"] = config["name"]

    config["random_seed"] = None
    config["timestamp"] = None
    config["run_id"] = None
    config["run_name"] = None
    config["initial_model_dir"] = None
    # TODO: explicitly store information in config, something like this:
    if "data_folder_experiment" in config:
        folder_name = config["data_folder_experiment"].split("/")[-1]
        config["random_seed"] = int(folder_name.split("_")[-1])
        config["timestamp"] = " ".join(folder_name.split("__")[0:2]).replace("_", " ")
        config["run_id"] = folder_name.split("__")[2].split("[")[0]
        config["run_name"] = config["data_folder_experiment"].split("/")[-2]
        config["weight_preference"] = (0.7, 0.3)

    config["run_folder"] = config["data_folder_experiment"]
    config["save_replay_buffer"] = False  # this was not in option in the original repos

    # create key->None entries for other newly introduced parameters
    config["video"] = dict()
    config["video"]["video_save_dir"] = None
    config["video"]["record_evy_n_episodes"] = None

    # remove old keys
    for key in [
        "data_folder_experiment",
        "use_cpu_for_rollout",
        "nmbr_random_designs",
        "iterations_random",
        "weights",
        "name",
        "load_model",
    ]:
        if key in config:
            del config[key]

    return config
"""
