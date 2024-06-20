sac_pso_batch = {
    "config_name": "Experiment 1: PSO Batch",  # Name of the experiment. Can be used just for reference later
    "data_folder": "/scratch/work/kielen1/experiments/data_exp_sac_pso_batch/",  # Name of the folder in which data generated during the experiments are saved
    "iterations_init": 300,  # Number of episodes for all initial designs as provided by the environment class
    "iterations": 100,  # Number of training episodes to run for all designs, AFTER the initial iterations
    "design_cycles": 55,  # Number of design adaptations after the initial designs
    "state_batch_size": 32,  # Size of the batch used during the design optimization process to estimate fitness of design
    "initial_episodes": 3,  # Number of initial episodes for each design before we start the training of the individual networks. Useful if steps per episode is low and we want to fill the replay.
    "use_gpu": True,  # Use True when GPU should be used for training and inference
    "cuda_device": 0,  # Which cuda device to use. Only relevant if you have more than one GPU
    "exploration_strategy": "random",  # Type of design exploration to use - we do usually one greedy design optimization and one random selection of a design
    "design_optim_method": "pso_batch",  # Which design optimization method to use
    "steps_per_episodes": 1000,  # Number of steps per episode
    "save_networks": True,  # If True networks are checkpointed and saved for each design
    "save_replay_buffer": True,  # If True replay buffers are checkpointed and saved for each design
    "rl_method": "SoftActorCritic",  # Which reinforcement learning method to use.
    "weights": [
        [0.0, 1.0],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1],
        [1.0, 0.0],
    ],  # Needed for MORL weigh preferences
    "rl_algorithm_config": dict(  # Dictonary which contains the parameters for the RL algorithm
        algo_params=dict(  # Parameters for the RL learner for the individual networks
            discount=0.99,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        algo_params_pop=dict(  # Parameters for the RL learner for the individual networks
            discount=0.99,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        net_size=200,  # Number of neurons in hidden layer
        network_depth=3,  # Number of hidden layers
        copy_from_gobal=True,  # Shall we pre-initialize the individual network with the global network?
        indiv_updates=1000,  # Number of training updates per episode for individual networks
        pop_updates=250,  # Number of training updates per episode for population networks
        batch_size=256,  # Batch size
    ),
    "env": dict(  # Parameters and which environment to use
        env_name="HalfCheetah",  # Name of environment
        render=False,  # Use True if you want to visualize/render the environment
        record_video=False,  # Use True if you want to record videos
    ),
}

sac_pso_sim = {
    "config_name": "Experiment 2: PSO using Simulations",
    "data_folder": "/scratch/work/kielen1/experiments/data_exp_sac_pso_sim/",  # Name of the folder in which data generated during the experiments are saved
    "iterations_init": 300,  # 300
    "iterations": 100,  # 100
    "design_cycles": 55,  # 55
    "state_batch_size": 32,  #
    "initial_episodes": 3,
    "use_gpu": True,
    "cuda_device": 0,
    "exploration_strategy": "random",
    "design_optim_method": "pso_sim",
    "steps_per_episodes": 1000,  # Number of steps per episode
    "save_networks": True,
    "save_replay_buffer": True,
    "rl_method": "SoftActorCritic",
    "weights": [
        [0.0, 1.0],
        [0.1, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6],
        [0.5, 0.5],
        [0.6, 0.4],
        [0.7, 0.3],
        [0.8, 0.2],
        [0.9, 0.1],
        [1.0, 0.0],
    ],  # Needed for MORL weigh preferences
    "rl_algorithm_config": dict(
        algo_params=dict(
            discount=0.99,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        algo_params_pop=dict(
            discount=0.99,
            reward_scale=1.0,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        net_size=200,
        network_depth=3,
        copy_from_gobal=True,
        indiv_updates=10,
        pop_updates=250,
        batch_size=256,
    ),
    "env": dict(
        env_name="HalfCheetah",
        render=False,
        record_video=False,
    ),
}

config_dict = {
    "sac_pso_batch": sac_pso_batch,
    "sac_pso_sim": sac_pso_sim,
}
