# neural-automata

Neural cellular automata implemented in PyTorch, based on https://distill.pub/2020/growing-ca/ (GitHub: https://github.com/google-research/self-organising-systems).

## Additions

The original article showed that automata could be trained using recurrent neural networks to accomplish the following:

1. Reach a target state after a certain number of iterations
2. Maintain such a target state
3. Regenerate from alterations to the target state

This repository modifies the training environment to show some new properties.

### 1. Coupling of seed states to targets

In [1_divergence.ipynb](https://github.com/kevinhu/neural-automata/blob/master/notebooks/1_divergence.ipynb) two seed states are trained to grow into two different target states under the same update network. The difference between the seeds is a single value at the 4th index of the center vector, which is 0 for one state and 1 for the other.

Here is the output of the first seed:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_1_colors.gif">
</p>

and here is the output of the second seed, with the exact same perception network:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_2_colors.gif">
</p>

In addition, an average of the two seed states produces the following (with the perception network the same as the previous two):

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/divergence_seed_mix_colors.gif">
</p>

### 2. Control of growth stages

The original paper showed that by using a pool to store training outputs, automata could be trained to persist and regenerate. Here in [2_metamorphosis.ipynb](https://github.com/kevinhu/neural-automata/blob/master/notebooks/2_metamorphosis.ipynb), the same pooling mechanism is employed to produce automata able to transition between stages, with the example here being cycling between two states:

<p align="center">
  <img src="https://github.com/kevinhu/neural-automata/blob/master/videos/metamorphosis_colors.gif">
</p>

It's likely possible to chain even more stages together, and it would be interesting to explore just how many can be used before the network starts failing or forgetting some states.

### 3. Alternative objectives

## Getting started

1. Install dependencies: `poetry install` (from within the repository)
2. Install Jupyter kernel: `poetry run python -m ipykernel install --user`
2. Inititialize virtual environment: `poetry shell`
3. Start Jupyter: `jupyter notebook`
