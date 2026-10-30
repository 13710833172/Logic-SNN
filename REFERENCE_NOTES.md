# References and implementation scope

[Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277) — Felix Petersen, Christian Borgelt, Hilde Kuehne, and Oliver Deussen, NeurIPS 2022.

[Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732) — Felix Petersen, Hilde Kuehne, Christian Borgelt, Julian Welzel, and Stefano Ermon, NeurIPS 2024.

Logic-tree synaptic layers adapt the related [Conv-DLGN project](https://github.com/13710833172/Conv-DLGN), preserving the operator ordering, fixed wiring, convolutional tree structure, and argmax gate export while adding spiking execution. This repository does not claim the underlying differentiable logic-gate formulation as its invention.

Spiking neurons, ATan surrogate gradients, and reset/execution utilities use [SpikingJelly](https://github.com/fangwei123456/spikingjelly), pinned in requirements.txt to commit `b049197b9589211418940c50031d3b7acb603bd6`.

Repository-specific work includes combining the logic layers with LIF/IF temporal dynamics, architecture variants, soft/hard checkpoint handling, distributed training, data preparation, and experiment records. These are implementation contributions and do not by themselves establish novelty or hardware efficiency.

Conventional ConvSNN runs are separate baselines. CIFAR10-DVS currently uses ConvSNN, not Logic-SNN. Dependencies retain their own licenses; the repository LICENSE remains unchanged.
