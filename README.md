# Multi-Objective Fast Evolution through Actor-Critic Reinforcement Learning

## Acknowledgement of Previous Work

The work is a continuation of the [`Coadaptation`](https://github.com/ksluck/Coadaptation) repository, developed by [Kevin Luck](https://github.com/ksluck), which was extended by [Oskar Rönnberg](https://github.com/psyberprimate) in a forked repository called [`MoCoadaptation`](https://github.com/psyberprimate/MoCoadaptation) during his master thesis. This repository, [`codingWhale13/MoCoadaptation`](https://github.com/codingWhale13/MoCoadaptation), is a fork of a fork.

In short, here's what happened:

1. [`Coadaptation`](https://github.com/ksluck/Coadaptation):
    - Implement the algorithm _Fast Evolution through Actor-Critic Reinforcement Learning_, optimizing both morphology (= design parameters) and behavior (= policy) of an agent.
    - PSO (_Particle Swarm Optimization_) is used to learn the morphology of the agent (e.g. limb lengths)
    - SAC (_Soft Actor Critic_) is used to learn the agent's policy
    - Focus on one environment: PyBullet implementation of HalfCheetah 
2. [`psyberprimate/MoCoadaptation`](https://github.com/psyberprimate/MoCoadaptation):
    - Adapt the HalfCheetah environment to a multi-objective (MO) setting, the reward now being a vector of two dimensions: speed and energy conservation
    - Adapt the agent to work with this multi-objective environment by scalarizing the reward based on a weight preference (e.g. 90% speed and 10% energy conservation) and training only on one such preference during one experiment
    - Allow "steering" experiments which load a model trained on a weight preference and then continue learning with a different preference
    - Add plotting utility to visualize the approximate Pareto front and the reward dimensions over time
3. [`codingWhale13/MoCoadaptation`](https://github.com/codingWhale13/MoCoadaptation):
    - Implement additional features:
        - Replay buffer saving/loading
        - Option to condition the policy and Q-network on the weight preference
        - Make option to use scalar/vector Q values easily usable (existed in previous repository, but on different branches)
    - Add two multi-objective environments: Walker2D and Hopper
    - Make code more user-friendly by writing argparse-based scripts for training and plotting
    - Refactor code and remove external dependency of rlkit (see [Code Structure](#code-structure))
    - Add interactive visualization using Plotly Dash (WIP)

### Citation
The original paper [_Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning_](https://research.fb.com/publications/data-efficient-co-adaptation-of-morphology-and-behaviour-with-deep-reinforcement-learning/) was presented at CoRL (Conference on Robot Learning) in 2019.

If you use this method in your research, please cite:
```
@inproceedings{luck2019coadapt,
  title={Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning},
  author={Luck, Kevin Sebastian and Ben Amor, Heni and Calandra, Roberto},
  booktitle={Conference on Robot Learning},
  year={2019}
}
```

### Acknowledgment From Original Repository

> This project would have been harder to implement without the great work of
> the developers behind rlkit and pybullet.
>
> The reinforcement learning loop makes extensive use of rlkit, a framework developed
> and maintained by Vitchyr Pong. You find this repository [here](https://github.com/vitchyr/rlkit).
> We made slight adaptations to the Soft-Actor-Critic algorithm used in this repository.
>
> Tasks were simulated in [PyBullet](https://pybullet.org/wordpress/), the
> repository can be found [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet).
> Adaptations were made to the files found in pybullet_evo to enable the dynamic adaptation
> of design parameters during the training process."

## Installation

Create a conda environment using [environment.yml](environment.yml). Unlike in the parent repositories, `rlkit` is no longer a dependency. 

## Code Structure

All relevant code is on the `master` branch; there is no need to switch to any other branch.

Instead of the previous dependency to [`rlkitMO`](https://github.com/psyberprimate/rlkitMO), the relevant code from there is included in the folder [rlkit](/rlkit) of this repository.

### Previous Code Structure

For the SAC agent, the `Coadaptation` repository uses an implementation from [`rlkit`](https://github.com/rail-berkeley/rlkit). Analogously, `MoCoadaptation` uses [`rlkitMO`](https://github.com/psyberprimate/rlkitMO).

The original `MoCoadaptation` has two branches which coincide with two branches of the dependency `rlkitMO`:

1. `MoCoadaptation/master` introduces the MO setting by using MO environments and scalarizing their vector returns a priori. It goes together with `rlkitMO/coadapt`.
2. `MoCoadaptation/interactive` extends this by vectorizing the Q-function and allowing to load a model from a checkpoint to fine-tune it "interactively". This branch goes together with `rlkitMO/coadapt_interactive`.

To compare this with the code structure in this repository: The functionality from the second scenario is now on the `master` branch, without a direct dependency to `rlkitMO` because the rlkit implementation of SAC  is included in this repository.

## Major Changes

- Conditioning on weight preference: Both policy and Q-network of the agent now take the weight preference into account. This itself does not change the (state, action, reward, next state) tuple (reward is a vector here). It only effects the loss calculation which takes the scalarized reward (using the weight preference) into account.
    - The idea behind this is to allow faster adaptation after loading a checkpoint and steering to a different preference.
- For more steering control, an agent now has two replay buffers (which appear as one, see [`MixedEvoReplayLocalGlobalStart`](RL/replay_mix.py)). We now have
    1. One replay buffer with the old replay data, loaded from a previous checkpoint and
    2. A fresh replay buffer that will be populated with new experience.
- The `old_replay_portion` option handles how much percentage of samples are from the old / new replay buffer. At the moment, this is a constant for one experiment, but extending it to follow some schedule should be straightforward.
- The replay buffer (meaning only one of the two [`EvoReplayLocalGlobalStart`](RL/replay_evo.py)s, populated with new experience) can be saved to disk. Because of its size (~10GB), only the latest replay buffer is stored, separately of the model checkpoints as an individual file.

## Config Arguments

- Different experiments are specified by their config files which are in the [configs](/configs/) folder. Compared to the previous config, some things have changed:

**Introduced new arguments:**

- `config_version`: Did not exist before, now set to 1
- `run_name`: Human-readable name of the run
- `timestamp`: Filled automatically
- `run_id`: Unique random hash, filled automatically (was previously included in file name)
- `run_folder`: Filled automatically as `f"{data_folder}/run_{timestamp}_{run_name}"`
- `random_seed`: Random seed used for experiment (was not explicitly stored in the config file before)
- `verbose`: To specify more verbose console output
- `use_wandb`: To specify if wandb is used for tracking or not (was not explicit before)
- `initial_model_dir`: Optional path to model checkpoint that should be loaded before training
- `save_replay_buffer`: Saving the replay buffer allows loading it from a checkpoint later on
- `video`/*: More control for video recording
- `condition_on_preference`: Give option to condition the policy and Q networks on the weight preference
- `use_vector_q`: Give explicit option to use scalar or vectorized Q function (one value per objective) during training
- `old_replay_portion`: Value between 0 and 1, specifying how much experience should be sampled from a previously loaded replay buffer
- `previous_weight_preferences`: Filled automatically as list of previously trained weight preferences (if any), each entry having format [weight_preference, run id]

**Changed arguments for more clarity:**

- `name` is now `config_id`, highlighting the idea that this is an identifier which comes with certain unchangeable parameters
- `data_folder_experiment` is now `run_folder` (see above)
- Instead of specifying all considered preferences in `weights` and then choosing the desired index with `weight_index`, there now only is `weight_preference` which directly specifies the desired weight preference as a tuple

**Removed unused arguments:**

- `use_cpu_for_rollout`
- `nmbr_random_designs`
- `iterations_random`
- `load_model`: Can be deducted from `initial_model_dir` being None or specified
- `exploration_strategy`: String declaring which design exploration strategy to use. Was an unused argument in `_training_loop`. Only the (uniform) random exploration strategy is used.
- `rl_algorithm_config`/*/`reward_scale`: Was always set to 1, removed to avoid confusion with energy reward scaling which was introduced in MORL setting.

## Experiment Folder Structure

A single training run is stored in a folder containing these elements:

- `config.json`
- `do_checkpoints` folder, containing one CSV file per design cycle:
    1. First row contains two strings:                  "Design Type", {design type, e.g. "Optimized"}
    2. Second row contains one string and some floats:  "Design Parameters", {value}, {value}, ...
    3. Third row contains one string and some floats:   {name of first reward dimension, e.g. "Reward Speed"}, {value}, {value}, ...
    4. Fourth row contains one string and some floats:  {name of second reward dimension, e.g. "Reward Energy"}, {value}, {value}, ...
    5. ... (more reward dimensions, depending on environment)
- `rl_checkpoints` folder, containing one PyTorch checkpoint of the trained RL policy per design cycle

These training runs (referred to as `--run-dir` in argparse scripts and `run_folder` in Python code) are contained in a `data_folder` (or `--exp-dir` for "experiment directory") which can have arbitrary subfolder structure. The steering history is automatically recollected, given that all config files and required run folders have not been deleted or moved.

One suggestion for keeping track of multiple runs within one steering experiment is given here:

- my_experiment (CONTAINS ALL EXPERIMENTS RELEVANT FOR ANALYZING STEERING PERFORMANCE)
   - 0.8-0.2 (TARGET PREFERENCE)
       - run_direct_{HASH}
       - run_direct_{HASH} (same setting as folder above, but with different seed)
       - run_steered_from_0.7-0.3_{HASH}
       - run_steered_from_1.0-0.0
       - ...
    - ...

 where the HASH entries are automatically generated unique run IDs and each folder starting with "run" has the following files:
 - `config.json`
 - `do_checkpoints` (contains `.csv` files)
 - `rl_checkpoints` (contains `.chk` files)