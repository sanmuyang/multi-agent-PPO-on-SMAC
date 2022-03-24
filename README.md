# Multi-Agent-PPO-on-SMAC
Implementations of IPPO and MAPPO on SMAC, the multi-agent StarCraft environment. What we implemented is a simplified version, without complex tricks. This repository is based on https://github.com/starry-sky6688/StarCraft. 

## Corresponding Papers
[IPPO: Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?](https://arxiv.org/abs/2011.09533)

[MAPPO: Benchmarking Multi-agent Deep Reinforcement Learning Algorithms](https://arxiv.org/abs/2006.07869)

## Requirements
+ pytorch
+ [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
+ [pysc2](https://github.com/deepmind/pysc2)

## Quick Start
	$ python main.py --map=3m --alg=ippo


