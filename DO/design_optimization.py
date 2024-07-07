class DesignOptimization:
    def __init__(self, config, replay, env):
        raise NotImplementedError

    def optimize_design(self, design, q_network, policy_network, weights, verbose):
        raise NotImplementedError
