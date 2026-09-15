# Reference Notes

This repository is intentionally small. It reuses the public `activation_based` stack from [SpikingJelly](https://github.com/fangwei123456/spikingjelly) as a dependency instead of vendoring the framework into this project.

The soft and hard logic-tree layers are adapted from the local reference project
`/data/linan/MPhil_1/my_convlogic`. The adapted implementation preserves its
16-operator ordering, fixed connectivity, depth-3 convolutional trees, and
argmax hard-gate conversion while adding temporal LIF execution.

## Upstream dependency

- Framework: `spikingjelly`
- Referenced commit: `b049197b9589211418940c50031d3b7acb603bd6`
- Installed via `requirements.txt`

## What this repository borrows

- Spiking neuron implementations such as `LIFNode` and `IFNode`
- Multi-step execution utilities under `activation_based`
- Weight initialization and network reset helpers
- The general Conv-BN-Spike-Pool modeling pattern commonly used in SpikingJelly examples

## What is original in this repository

- A compact, inspection-friendly training pipeline for static-image SNN experiments
- Deterministic train/validation splitting with explicit test-only evaluation
- Timestamped run directories with lightweight experiment logging
- Standalone evaluation script that reconstructs a model from saved run metadata
- Repository-level packaging for reproducibility, CI smoke tests, and experiment summaries

## Why this distinction matters

For a public research portfolio, it is important to separate:

- upstream framework functionality
- experiment-specific design choices
- repository engineering added for reproducibility

This project is therefore best understood as a reproducible SNN experiment package built on top of SpikingJelly, not as a reimplementation of the full upstream framework.
