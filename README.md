# Multi-objective Fast Evolution through Actor-Critic Reinforcement Learning
This repository is orignally from: https://github.com/ksluck/Coadaptation

The original paper was presented in [**Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning**](https://research.fb.com/publications/data-efficient-co-adaptation-of-morphology-and-behaviour-with-deep-reinforcement-learning/).
This paper was presented on the Conference on Robot Learning in 2019.

## Citation
If you use this code in your research, please cite
```
@inproceedings{luck2019coadapt,
  title={Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning},
  author={Luck, Kevin Sebastian and Ben Amor, Heni and Calandra, Roberto},
  booktitle={Conference on Robot Learning},
  year={2019}
}
```

## Acknowledgements of Previous Work
The work is heavily based on the previous work of Luck. Many thanks to him and the rest. You should definitely read his repository page. By his own words:

"This project would have been harder to implement without the great work of
the developers behind rlkit and pybullet.

The reinforcement learning loop makes extensive use of rlkit, a framework developed
and maintained by Vitchyr Pong. You find this repository [here](https://github.com/vitchyr/rlkit).
We made slight adaptations to the Soft-Actor-Critic algorithm used in this repository.

Tasks were simulated in [PyBullet](https://pybullet.org/wordpress/), the
repository can be found [here](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet).
Adaptations were made to the files found in pybullet_evo to enable the dynamic adaptation
of design parameters during the training process."

## Installation

Read Luck's reposity for the instrutions on installation. In this repository, there's a requirement.txt that has the needed modules.

## Branches
'main' is the priori scalarized returns and 'interactive' is vectorized Q-function with priori scalarized returns.

## Videos

### Chapter 3 videos

PSO sim:
[0.0-0.4_priori_sim](https://vimeo.com/915549793?share=copy),
[0.5-1.0_priori_sim](https://vimeo.com/915549793?share=copy)

PSO batch:
[0.0-0.4_priori_batch](https://vimeo.com/915549753?share=copy),
[0.5-1.0_priori_batch](https://vimeo.com/915549772?share=copy)

### Chapter 4 videos

Vectorized:
[0.3-1.0_vectorized](https://vimeo.com/924973475?share=copy)

Vectorized with bootstrapping:

[0.3-1.0_bootstrapped_seed_1](https://vimeo.com/924968081?share=copy),
[0.3-1.0_bootstrapped_seed_3](https://vimeo.com/924969157?share=copy)





