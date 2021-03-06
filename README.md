# MineRL RLlib Benchmark [WIP]

Here we benchmark various reinforcement learning algorithms available in [RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) on the [MineRL](https://minerl.io/docs/) environment.

[RLlib](https://docs.ray.io/en/releases-0.8.6/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. 
RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.

## Installation

OS: Linux  
If you want to run on Windows, you can install the ray nightly version, but some RL algorithms may break.

Make sure you have JDK 1.8 on your system for MineRL.

Install [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) with correct cuda version, then
`pip install -r requirements.txt`

## Experiments

See [Planned Implementation Details](Implementation.md)

Currently experiments do not support headless mode.  
Multiprocessing is available but not merged into minerl ([see this pull](https://github.com/minerllabs/minerl/pull/352)). For now, do
```
git clone https://github.com/minerllabs/minerl.git
cd minerl
git fetch origin pull/352/head:process-safe
git checkout process-safe
pip install -e .
```

For discrete action spaces, make sure you have the environment variable `MINERL_DATA_ROOT` set, otherwise it defaults to `data`. Then, run `python generate_kmeans.py`

### Online Reinforcement Learning [WIP]
This is the standard reinforcement learning agent-environment loop.

Example:
`python train.py --experiment custom_model/impala`

### Offline Reinforcement Learning [TODO]
As human demonstrations available in MineRL, it is possible to increase sample efficiency by using them to learn a better policy. 
This section is dedicated to offline sampling from the dataset only and does not sample from the environment.

### Mixed: Online Reinforcement Learning with Offline Data [TODO]
One can try to get the best of both worlds of online and offline RL, by learning when data is already available and sampling for exploration. 
