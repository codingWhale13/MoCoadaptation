class RL_algorithm:
    def __init__(self, config, env, replay, networks):
        self._config = config
        self.file_str = config["run_folder"]

        self._env = env
        self._replay = replay
        self._networks = networks

        if "use_only_global_networks" in config.keys():
            self._use_only_global_networks = config["use_only_global_networks"]
        else:
            self._use_only_global_networks = False

    def episode_init(self):
        raise NotImplementedError
