# neural-automata

Neural cellular automata implemented in PyTorch, based on https://distill.pub/2020/growing-ca/ (GitHub: https://github.com/google-research/self-organising-systems).

## Additions

The original article showed that automata could be trained using recurrent neural networks to accomplish the following:

1. Reach a target state after a certain number of iterations
2. Maintain such a target state
3. Regenerate from alterations to the target state

This repository modifies the training environment to show some new properties.

### 1. Coupling of seed states to targets

### 2. Control of growth stages

### 3. Alternative objectives

## Getting started

1. Install dependencies: `poetry install` (from within the repository)
2. Install Jupyter kernel: `poetry run python -m ipykernel install --user`
2. Inititialize virtual environment: `poetry shell`
3. Start Jupyter: `jupyter notebook`
