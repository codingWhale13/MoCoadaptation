# Multi-objective Fast Evolution through Actor-Critic Reinforcement Learning
This repository is originally from: https://github.com/ksluck/Coadaptation

The original paper was presented in [**Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning**](https://research.fb.com/publications/data-efficient-co-adaptation-of-morphology-and-behaviour-with-deep-reinforcement-learning/).
This paper was presented in the Conference on Robot Learning in 2019.

## Citation
If you use the original code in your research, please cite
```
@inproceedings{luck2019coadapt,
  title={Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning},
  author={Luck, Kevin Sebastian and Ben Amor, Heni and Calandra, Roberto},
  booktitle={Conference on Robot Learning},
  year={2019}
}
```

## Acknowledgements of Previous Work
The work is a continuation of the [`Coadaptation`](https://github.com/ksluck/Coadaptation) repository, developed by [Kevin Luck](https://github.com/ksluck), and extended by [Oskar Rönnberg](https://github.com/psyberprimate) in a [forked repository](https://github.com/psyberprimate/MoCoadaptation) during his master thesis.

### Acknowledgments From Original Repository

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

This repository is a fork of [`MoCoadaptation`](https://github.com/psyberprimate/MoCoadaptation) which in turn is forked from [`Coadaptation`](https://github.com/ksluck/Coadaptation). As the name suggests, `MoCoadaptation` extends `Coadaptation` by introducing a multi-objective ("MO") setting. It does so by using MO environements and scalarizing the returns (which are now vectors) a priori via a weight preference. The code is limited to the bi-objective scenario fo the HalfCheetah environment, using Soft Actor Critic (SAC) as the RL agent and PSO for the design optimization.

For the SAC agent, `Coadaptation` uses an implementation from [`rlkit`](https://github.com/rail-berkeley/rlkit). Analogously, `MoCoadaptation` uses [`rlkitMO`](https://github.com/psyberprimate/rlkitMO).

The original `MoCoadaptation` has two branches which coincide with two branches of the dependency `rlkitMO`:

1. `MoCoadaptation/master` introduces the MO setting by using MO environments and scalarizing their vector returns a priori. It goes together with `rlkitMO/coadapt`.
2. `MoCoadaptation/interactive` extends this by vectorizing the Q-function and allowing to load a model from a checkpoint to fine-tune it "interactively". This branch goes together with `rlkitMO/coadapt_interactive`.

To compare this with the code structure in this repository: The functionality from the second scenario is now on the `master` branch, without a direct dependency to `rlkitMO` because the SAC implementation is included in this repository.
