# neural-automata

Neural cellular automata implemented in PyTorch, based on https://distill.pub/2020/growing-ca/ (GitHub: https://github.com/google-research/self-organising-systems).

## Additions

The original article showed that automata could be trained using recurrent neural networks to accomplish the following:

1. Reach a target state after a certain number of iterations
2. Maintain such a target state
3. Regenerate from alterations to the target state

This repository modifies the training environment to show some new properties.

### 1. Coupling of seed states to targets

In [1_divergence.ipynb](https://github.com/kevinhu/neural-automata/blob/master/notebooks/1_divergence.ipynb) two different seed states are trained to grow into two different target states under the same update network. The difference between the seeds is a single value at the 5th index of the center vector, which is 0 for one state and 1 for the other. Using just this difference, the network successfully produces entirely different outputs.

Here is the output of the first seed:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_1_colors.gif">
</p>

Here is the output of the second seed, with the exact same perception network:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_2_colors.gif">
</p>

In addition, an average of the two seed states produces the following (with the perception network the same as the previous two):

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_mix_colors.gif">
</p>

As expected, a mix of the two seeds also produces a mix of the two targets. In theory, this approach could be used to couple even more seed states to targets, all under the same perception network.

### 2. Control of growth stages

The original paper showed that by using a pool to store training outputs, automata could be trained to persist and regenerate. Here in [2_metamorphosis.ipynb](https://github.com/kevinhu/neural-automata/blob/master/notebooks/2_metamorphosis.ipynb), the same pooling mechanism is employed to produce automata able to transition between stages, with the example here being cycling between two states:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/metamorphosis_colors.gif">
</p>

It's likely possible to chain even more stages together, and it would be interesting to explore just how many can be used before the network starts failing or forgetting some states.

### 3. Alternative objectives

Besides using images as targets, other more abstract cost functions can be employed. In [3_optimization.ipynb](https://github.com/kevinhu/neural-automata/blob/master/notebooks/3_optimization.ipynb), a network with the same architecture is trained with two objectives:

1. Vitality: maximize the sum of 'alive' state values (the 4th channel).
2. Stability: minimize the discrepancy between the end and penultimate states.

To prevent the trivial solution of a uniform `lawn`, at each iteration a circular mask is applied such that each cell can only have a certain number of neighbors. The end result is an interesting layout that bears some resemblance to a [Turing pattern](https://en.wikipedia.org/wiki/Turing_pattern). Here, all 16 channels are shown in parallel.

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/optimization_channels.gif">
</p>

## Getting started

1. Install dependencies: `poetry install` (from within the repository)
2. Install Jupyter kernel: `poetry run python -m ipykernel install --user`
2. Inititialize virtual environment: `poetry shell`
3. Start Jupyter: `jupyter notebook`
