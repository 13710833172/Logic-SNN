# Logic-SNN: Differentiable Logic Gates with Spiking Dynamics

A PyTorch/SpikingJelly research implementation combining trainable logic-tree synapses with LIF or IF temporal dynamics. The experiments examine classification quality and the gap between soft gate training and discretized gate inference.

## Model and supported tasks

```text
Encoded image -> convolutional logic tree -> spiking neuron -> pooling
              -> dense logic layers + spiking neurons -> time/class-group mean
Soft gates: mixtures over 16 Boolean operators
Hard gates: argmax operator selection with fixed truth tables
```

Logic layers adapt the related [Conv-DLGN implementation](https://github.com/13710833172/Conv-DLGN). Spiking neurons and surrogate gradients use [SpikingJelly](https://github.com/fangwei123456/spikingjelly). See [REFERENCE_NOTES.md](REFERENCE_NOTES.md) for the implementation boundary.

| Model family | MNIST | CIFAR-10 | CIFAR10-DVS |
| --- | --- | --- | --- |
| `logic_snn` | Supported | Supported | Not implemented |
| `conv_snn` | Supported baseline | Supported baseline | Supported baseline |

MNIST and CIFAR-10 are static-image experiments with repeated timesteps. DVS uses event-derived frames and belongs to the conventional ConvSNN baseline.

## Recorded results

| Model / dataset | Soft test | Hard test | Test samples |
| --- | ---: | ---: | ---: |
| Logic-SNN / MNIST, large identity-middle model | 98.05% | 97.54% | 10,000 |
| Logic-SNN / CIFAR-10, pyramid-m model | 67.49% | 28.76% | 10,000 |
| ConvSNN / MNIST, historical baseline | 99.36% | — | 10,000 |
| ConvSNN / CIFAR-10, historical baseline | 88.04% | — | 10,000 |
| ConvSNN / CIFAR10-DVS | 61.20% | — | 1,000 |

Results are summarized from local metric files and were checked for model-family attribution. They are not matched-parameter comparisons or multi-seed estimates. Datasets, checkpoints, per-run arguments, and experiment-history files are kept outside the repository; experiments were not rerun for this documentation update.

The CIFAR-10 soft-to-hard loss is **38.73 percentage points** and remains an open optimization issue. Current records do not establish superiority over ConvSNN. Hard synaptic operators also do not make the whole system Boolean: membrane dynamics, current scaling, and readout remain part of inference. No hardware energy or latency advantage is claimed.

## Setup and short execution check

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py --dataset mnist --model-family logic_snn   --output-dir ./runs/mnist --device cpu --epochs 1   --batch-size 8 --timesteps 2 --num-workers 0   --limit-train-batches 2 --limit-val-batches 2
```

This command checks the pipeline with limited batches; it does not reproduce the result table. Replace `<run>` below with its generated directory:

```bash
python test.py --run-dir ./runs/mnist/<run>   --inference both --device cpu --num-workers 0 --limit-test-batches 2
```

For full test evaluation, omit `--limit-test-batches`. Checkpoints and per-run training histories are not bundled.

## Training and export

Always specify `--model-family`: MNIST defaults to `logic_snn`, while CIFAR-10 and DVS default to `conv_snn`. The explicit flag avoids silently training a different baseline.

```bash
# Full static CIFAR-10 logic model: see the technical guide for architecture options.
python train.py --dataset cifar10 --model-family logic_snn --device cuda

# Export hard gates from a trained soft checkpoint.
python discretize.py --checkpoint ./runs/mnist/<run>/checkpoint_best.pt

python -m unittest discover -s tests -v
```

Multi-GPU training is available through `train_ddp.py`; physical batch size is per GPU. Large pyramid models have substantially greater resource requirements than the short MNIST execution check.

## Evidence and documentation

- [Technical guide](README.technical.md): architecture variants, full training commands, export, and distributed training.
- [Conventional ConvSNN baseline](https://github.com/13710833172/Conv-SNN-Baselines): related static-image baseline package.
- [Attribution](REFERENCE_NOTES.md): underlying logic-gate methods and SpikingJelly dependency.

The older `results/experiment_summary.*` files describe legacy ConvSNN runs only. The existing summary script ranks by test accuracy and does not separate model families or expose both execution modes. Use the model-labeled result summary in this README rather than interpreting that generated ranking as validation-only model selection. Model and training code were not changed in this documentation update.

## License

[MIT](LICENSE). Dependencies retain their own licenses.
