# Logic-SNN: Differentiable Logic-Gate Spiking Neural Networks

**English** | [简体中文](README.zh-CN.md)

**Learnable logic synapses meet spiking temporal dynamics.**

A PyTorch/SpikingJelly research implementation that integrates differentiable logic-tree synapses with LIF/IF neurons. The logic layers build on [Conv-DLGN](https://github.com/13710833172/Conv-DLGN); spiking neurons and surrogate gradients use [SpikingJelly](https://github.com/fangwei123456/spikingjelly). See [attribution](REFERENCE_NOTES.md) for the underlying methods.

## Abstract

Logic-gate learning provides trainable Boolean operator choices, while spiking neurons provide temporal state. Logic-SNN brings these two mechanisms together through convolutional and dense logic synapses coupled to LIF/IF dynamics. The model learns operator mixtures with surrogate-gradient training and exports discrete synaptic gates while retaining its spiking state and temporal readout. A representative four-timestep MNIST model achieves **98.05% soft and 97.54% hard test accuracy** on 10,000 examples, with a **0.51 percentage-point** conversion difference. A CIFAR-10 configuration reaches **67.49% soft accuracy**. Paired execution-mode evaluation makes the contribution of discrete gate export explicit for each task.

## Contributions

- **Logic synapses with temporal state.** Local convolutional logic trees and dense logic layers drive LIF/IF neurons, connecting operator learning to time-dependent spike computation.
- **A shared soft/hard spiking architecture.** Gate export preserves input connectivity, spiking dynamics, current gain, and temporal/class-group aggregation, allowing evaluation of the same architecture with discrete synaptic operators.
- **Task-specific evidence for the hybrid model.** MNIST demonstrates 97.54% hard accuracy at four timesteps; paired CIFAR-10 measurements characterize soft and exported operation in a deeper pyramid configuration.

## Methodology

![Logic-SNN couples convolutional and dense logic layers with spiking neurons across repeated timesteps, and aggregates class-group spikes over time.](docs/figures/overview.svg)

*Figure 1. Logic layers compute synaptic outputs; current scaling drives the spiking neurons. Neuron state evolves across repeated presentations of a static image, and time/class-group means produce the final scores. Export changes gate operators while preserving the temporal architecture.*

**Synaptic computation.** A logic gate learns a mixture over 16 two-input Boolean operators. Local depth-3 convolutional trees extract spatial features, followed by dense logic layers. In hard mode, each mixture becomes its argmax operator with fixed connectivity. The [logic layers](snn_minimal/models/logic_layers.py) implement both representations.

**Temporal computation.** At every timestep, logic outputs are multiplied by a current gain and passed to LIF/IF neurons; selected convolutional stages apply max pooling. Static-image inputs are repeated over $T$ timesteps. If $s_{t,c,j}$ denotes an output spike for class $c$ and group member $j$, the readout is

$$r_c=\frac{1}{T M}\sum_{t=1}^{T}\sum_{j=1}^{M}s_{t,c,j}.$$

Here $M$ is the number of outputs per class. The resulting model combines discrete-capable synapses with membrane dynamics and temporal aggregation. See the [shared forward path and export](snn_minimal/models/logic_snn.py).

## Experiments

**Question 1: can discrete synaptic gates support the hybrid spiking classifier?** On MNIST, the large identity-middle configuration reaches **97.54% hard accuracy**, within **0.51 pp** of its soft counterpart at four timesteps.

**Question 2: how do the two execution modes behave on CIFAR-10?** The pyramid-m configuration evaluates spatial logic features and spiking computation on natural images, with both modes reported explicitly.

| Logic-SNN / configuration | Timesteps | Soft test accuracy | Hard test accuracy | Test examples |
| --- | ---: | ---: | ---: | ---: |
| MNIST / large, identity-middle | 4 | **98.05%** | **97.54%** | 10,000 |
| CIFAR-10 / pyramid-m, 9-channel input | 4 | **67.49%** | 28.76% | 10,000 |

*Table 1. Representative completed local configurations. Hard evaluation uses the gate export from the corresponding soft checkpoint. Dataset, architecture, and training settings differ between rows.*

**Takeaway.** The MNIST result establishes a concrete operating point where discrete synapses retain the classifier's accuracy closely. CIFAR-10 supports the 67.49% claim specifically for soft execution; its hard-mode score is measured separately. This execution-mode distinction keeps the role of operator learning and the effect of export visible.

**Protocol.** Both configurations use LIF neurons and validation-selected soft checkpoints; each is an individual completed configuration rather than an average across independent seeds. The table measures classification under two synaptic representations. Membrane dynamics, current scaling, and readout remain part of hard inference. Hardware latency and energy are separate measurement targets. Datasets, weights, per-run arguments, and raw experiment histories are maintained outside the repository.

## Supported tasks

| Model family | MNIST | CIFAR-10 | CIFAR10-DVS |
| --- | --- | --- | --- |
| `logic_snn` | Static image, repeated timesteps | Static image, repeated timesteps | — |
| `conv_snn` | Conventional baseline | Conventional baseline | Event-frame baseline |

CIFAR10-DVS experiments use **`conv_snn`** and event-derived frames. They belong to the conventional baseline branch of this repository, separate from the Logic-SNN results in Table 1. The related [Conv-SNN-Baselines](https://github.com/13710833172/Conv-SNN-Baselines) repository presents conventional static-image baselines.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Short pipeline check with restricted batches.
python train.py --dataset mnist --model-family logic_snn \
  --output-dir ./runs/mnist --device cpu --epochs 1 \
  --batch-size 8 --timesteps 2 --num-workers 0 \
  --limit-train-batches 2 --limit-val-batches 2
```

Evaluate the generated run, replacing `RUN_NAME` with its directory name:

```bash
python test.py --run-dir ./runs/mnist/RUN_NAME \
  --inference both --device cpu --num-workers 0 --limit-test-batches 2
```

This checks execution on restricted batches. Full test evaluation omits `--limit-test-batches`; the short command is separate from the four-timestep configurations in Table 1. See the [technical guide](README.technical.md) for architecture options and full training commands.

```bash
# Static CIFAR-10 logic-family training entry point.
python train.py --dataset cifar10 --model-family logic_snn --device cuda

# Export a checkpoint generated by your training run.
python discretize.py --checkpoint ./runs/mnist/RUN_NAME/checkpoint_best.pt
python -m unittest discover -s tests -v
```

Specify `--model-family` explicitly: MNIST defaults to `logic_snn`, while CIFAR-10 and DVS default to `conv_snn`. Distributed training is available through `train_ddp.py`, with physical batch size specified per GPU.

<details>
<summary>Documentation and result attribution</summary>

- [Technical guide](README.technical.md): variants, export, and distributed training.
- [REFERENCE_NOTES.md](REFERENCE_NOTES.md): method attribution and implementation scope.
- Existing `results/experiment_summary.*` files describe legacy ConvSNN runs. The summary script ranks by test accuracy without separating model families or soft/hard modes. Table 1 identifies the Logic-SNN configurations directly; the legacy ranking is a different selection protocol.

</details>

## Conclusion

Logic-SNN connects learned logic synapses to spiking temporal state and evaluates the exported hybrid directly. Its four-timestep MNIST model demonstrates **97.54% hard accuracy with a 0.51 pp conversion difference**.

## License

[MIT](LICENSE). Dependencies retain their own licenses.
