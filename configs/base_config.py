# NOTE: Parameters in config files should not be modified to avoid confusion.
# Instead of changing this file, create a copy with a unique config_id and apply changes there.
# Exceptions, when overwriting parameters programmatically (i.e. from a Python file) is fine:
# - run_name (to identify the experiment in a human-friendly way)
# - verbose and use_gpu (to give the option between debugging vs. running many experiments using SLURM)
# - random_seed (to allow running the same experiment with different random seeds)
# - data_folder (e.g. using subfolders for more clear structure)
# - initial_model_dir (to allow loading different starting points and seeing how training with the same setting performs)
# - weight_preference (to allow running multiple different weight preferences with the same setting)
# - env/record_video and video/* (e.g. overwritten by video_from_checkpoint.py)

base_config = {
    # GENERAL OPTIONS
    "config_version": 1,  # Specifies version number of config (0 is from old repo, 1 is from this repo (codingWhale13/MoCoadaptation)
    "config_id": "base_config",  # Identifier of of this config, for later reference
    "project_name": "MO Co-Adaptation",  # Name of this project, used as project name in wandb
    "run_name": "default",  # Human-readable name of the experiment, part of experiment folder name and used as wandb run name
    "timestamp": None,  # Timestamp of experiment, created automatically (e.g. "2024-06-25T09-52-02")
    "run_folder": None,  # Folder for data generated by experiment, created automatically: f"{data_folder}/run_{timestamp}_{run_name}"
    "run_id": None,  # Unique ID for future reference, created randomly (independent of random seed)
    "random_seed": None,  # Integer used for random seeding (if not specified, generated randomly)
    "verbose": False,  # Use True for more verbose console output (e.g. printing the config values at the beginning)
    "use_wandb": True,  # Use True to log the run with wandb ("weights and biases")
    # LOADING/SAVING OPTIONS
    "data_folder": "/scratch/work/kielen1/experiments/",  # Path to parent folder of experiment run
    "initial_model_dir": None,  # If specified, loads the latest checkpoint from this experiment folder at the beginning of training",
    "save_networks": True,  # Use True to save checkpoints (RL network and design specification) for each design
    "save_replay_buffer": False,  # Use True to save the most recent RL replay buffer (can be a few GB large)
    # GPU OPTIONS
    "use_gpu": True,  # Use True to use GPU for training and inference
    "cuda_device": 0,  # Specifies which cuda device to use (only relevant in case of multiple GPUs)
    # ENVIRONMENT PARAMETERS
    "env": dict(
        env_name="HalfCheetah",  # Name of the environment to train the agent in
        render=False,  # Use True to render the environment
        record_video=False,  # Use True to record and save videos using the BestEpisodesVideoRecorder
    ),
    # VIDEO PARAMETERS (only take effect if `record_video` is True)
    "video": dict(
        video_save_dir=None,  # Where to save videos (if not specified, f"{experiment_folder}/videos" is used)
        record_evy_n_episodes=5,  # Specifies the interval at which videos should be created in terms of env episodes
    ),
    # PARAMETERS RELEVANT FOR BOTH DO AND RL
    "iterations_init": 300,  # Number of episodes for all initial designs as provided by the environment class
    "design_cycles": 55,  # Number of design adaptations after the initial designs
    "iterations": 100,  # Number of training episodes to run for each design, *after* the initial iterations
    "initial_episodes": 3,  # Number of initial episodes per design before training of the individual networks (useful for filling replay buffer when steps_per_episodes is low)
    "steps_per_episodes": 1000,  # Number of steps per episode
    "weight_preference": (
        0.5,
        0.5,
    ),  # Objective preference in MORL setting (one non-negative value per objective, should add up to 1)
    # DESIGN OPTIMIZATION (DO) PARAMETERS
    "design_optim_method": "pso_batch",  # Which design optimization method to use
    "state_batch_size": 32,  # Size of the batch used during the design optimization process to estimate fitness of design
    # REINFORCEMENT LEARNING (RL) PARAMETERS
    "rl_method": "SoftActorCritic",  # Which reinforcement learning method to use
    "condition_on_preference": True,  # Use True to condition the policy and Q networks on the weight preference (input grows by the number of objectives)
    "scalarize_before_q_loss": False,  # Use True to scalarize Q-target and Q-prediction before loss calculation, use False to feed original values (vectors if use_vector_q, else scalars) to loss function
    "use_vector_q": False,  # Use True to use Q-functions with vector output (one value per objective), use False for scalar Q-values
    "rl_algorithm_config": dict(
        algo_params=dict(  # Parameters for the RL learner for the individual networks
            discount=0.99,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        algo_params_pop=dict(  # Parameters for the RL learner for the population network
            discount=0.99,
            soft_target_tau=0.005,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            alpha=0.01,
        ),
        batch_size=256,
        net_size=200,  # Number of neurons in hidden layer
        network_depth=3,  # Number of hidden layers
        copy_from_gobal=True,  # Use True to pre-initialize the individual network with the global network
        indiv_updates=1000,  # Number of training updates per episode for individual networks
        pop_updates=250,  # Number of training updates per episode for population networks
    ),
}
