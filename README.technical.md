# Logic-SNN / Conv-SNN Technical Guide

A small, runnable SNN project: MNIST and CIFAR-10 support trainable soft logic
gates and exportable hard Boolean gates, while the original Conv-SNN remains
available for legacy experiments. Spiking neurons use [SpikingJelly](https://github.com/fangwei123456/spikingjelly) commit `b049197b9589211418940c50031d3b7acb603bd6`.

## Goals

- Keep the codebase runnable and easy to inspect
- Reuse the Conv-SNN training pattern from SpikingJelly
- Avoid copying the whole upstream framework into this repository

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a Logic-SNN (creates a timestamped run directory)
CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset mnist \
  --model-family logic_snn \
  --data-root ./data \
  --output-dir ./runs/mnist \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --device cuda \
  --amp

# 3. Evaluate soft and hard checkpoints on the test split
python test.py --run-dir ./runs/mnist/<timestamped-run> --inference both --device cuda
```

## Project Layout

```
SNN/
├── train.py                 # Training entry point
├── train_ddp.py             # Multi-GPU DistributedDataParallel entry point
├── test.py                  # Soft/hard test-set evaluation
├── discretize.py            # Soft checkpoint to hard-gate bundle
├── requirements.txt
├── REFERENCE_NOTES.md       # Upstream reference mapping
├── data/                    # Downloaded datasets (gitignored)
├── runs/                    # Training outputs (gitignored)
└── snn_minimal/
    ├── data.py              # Datasets and DataLoaders
    ├── engine.py            # Epoch loop, checkpoints
    ├── plotting.py          # train.log, metrics.json, SVG curves
    └── models/
        ├── conv_snn.py      # Unchanged CIFAR-10 Conv-SNN
        ├── factory.py       # Model construction from CLI / args.json
        ├── logic_layers.py  # Soft/hard logic layers
        ├── logic_ops.py     # 16 Boolean operators
        └── logic_snn.py     # Soft/hard MNIST and CIFAR-10 Logic-SNN
```

| File                             | Role                                                                                   |
| -------------------------------- | -------------------------------------------------------------------------------------- |
| `train.py`                       | Train on the training split; each epoch runs validation; saves checkpoints and curves  |
| `train_ddp.py`                   | Multi-GPU training with sharded data, synchronized metrics, and rank-zero checkpoints  |
| `test.py`                        | Evaluate soft and/or hard checkpoints on the official test split                       |
| `discretize.py`                  | Convert `checkpoint_best.pt` into `checkpoint_best_hard.pt`                            |
| `snn_minimal/plotting.py`        | Logging and matplotlib SVG plots (invoked automatically by `train.py`)                 |
| `snn_minimal/engine.py`          | Shared `run_epoch`, checkpoint save/load                                               |
| `snn_minimal/data.py`            | MNIST / CIFAR-10 loaders and transforms                                                |
| `snn_minimal/models/logic_snn.py`| Logic-SNN definition                                                                   |
| `snn_minimal/models/conv_snn.py` | Legacy `ConvSNN` definition                                                            |


## Train vs Validation vs Test


| Phase          | When                      | Data split                          | Script     |
| -------------- | ------------------------- | ----------------------------------- | ---------- |
| **Train**      | Every epoch               | Training set (`train=True`)         | `train.py` |
| **Validation** | Every epoch (after train) | Split from official training set    | `train.py` |
| **Test**       | After training (manual)   | Official test split (`train=False`) | `test.py`  |


During training, validation is created by splitting the official training set with a deterministic random split (`--val-ratio`, `--split-seed`). The official test split is reserved for `test.py` only.

There is **no** separate test phase inside `train.py`; use `test.py` when you need `test_metrics.json`.

## Run Directory

Each new training run creates a **timestamped** folder:

```text
<output-dir>/<dataset>_<YYYYMMDD_HHMMSS>/
```

Example: `runs/mnist_logic_fixed_large_100/mnist_20260911_102623/`


| File                   | Description                                             |
| ---------------------- | ------------------------------------------------------- |
| `args.json`            | Full CLI arguments for this run                         |
| `checkpoint_latest.pt` | Latest epoch weights                                    |
| `checkpoint_best.pt`   | Best validation accuracy (used by `test.py` by default) |
| `checkpoint_best_hard.pt` | Inference-only Boolean gate bundle                   |
| `hard_validation.json` | Soft/hard validation gap and spike rates                |
| `train.log`            | One line per epoch (loss, acc, time, lr, …)             |
| `metrics.json`         | Compact summary: `final` and `best` epoch metrics       |
| `train_loss.svg`       | Training loss vs epoch (written when training finishes) |
| `val_acc.svg`          | Validation accuracy vs epoch                            |
| `confusion_matrix.svg` | Validation confusion matrix of `checkpoint_best.pt`     |
| `test_metrics.json`    | Created by `test.py` (not by `train.py`)                |


**Resume:** pass `--resume path/to/checkpoint_latest.pt`. Training continues in the **same** run directory; optimizer and scheduler states are restored, `train.log` is appended, and curves cover the full history.

`--init-checkpoint` loads **weights only** and starts a new run directory with a fresh optimizer.

TensorBoard is **not** used (no `events.out.tfevents.`* files).

### `metrics.json` format

```json
{
  "final": {
    "epoch": 99,
    "train_loss": 0.048,
    "val_acc": 0.9780
  },
  "best": {
    "epoch": 97,
    "train_loss": 0.048,
    "val_acc": 0.9788
  }
}
```

Full per-epoch curves are in `train.log` and the SVG files; `metrics.json` only stores the last epoch and the best-validation epoch.

### `test_metrics.json` format

```json
{
  "model_family": "logic_snn",
  "inference": "both",
  "soft_test_acc": 0.9805,
  "hard_test_acc": 0.9754,
  "soft_hard_test_acc_gap": 0.0051,
  "test_samples": 10000
}
```

## Latest Results Snapshot

Numbers below come from completed run folders in this workspace. Short
hyperparameter probes are not listed.

### MNIST Logic-SNN (`runs/mnist_logic_fixed_large_100/mnist_20260911_102623`)

- Architecture: 2 conv logic trees (`1→96→192`) plus 3 dense logic layers
  (`9408→8192→8192→20480`), with the middle dense layer initialized as Boolean `A`
- `metrics.json`
  - best val acc: **0.9788** (epoch 97)
  - final val acc: **0.9780** (epoch 99)
- `test_metrics.json`
  - soft test acc: **0.9805**
  - hard test acc: **0.9754**
  - test samples: **10000**
- training setup: 100 epochs, `T=4`, AdamW `lr=0.01`, `weight_decay=0.002`,
  cross-entropy readout, `logic_outputs_per_class=2048`, single GPU

The smaller identity network
`runs/mnist_logic_fixed_identity_100/mnist_20260911_061814` uses
`64→128` conv channels and `4096→4096→10240` dense widths. Soft test acc is
**0.9594**; hard test acc is **0.9375**.

### CIFAR-10 Logic-SNN (`runs/cifar10_logic_pyramid_m_9ch_ddp2_cosine_long/cifar10_20260913_190001`)

- Architecture: pyramid with `k=256`
  (`9→256→1024→4096→8192` conv, then `327680→163840→81920` dense)
- `metrics.json`
  - best val acc: **0.6684** (epoch 292)
  - final val acc: **0.6612** (epoch 299)
- `test_metrics.json`
  - soft test acc: **0.6749**
  - hard test acc: **0.2876**
  - test samples: **10000**
- training setup: 300 epochs, 2-GPU DDP, per-GPU batch 16, `grad_accum_steps=4`
  (effective batch 128), cosine schedule, `min_lr=1e-4`, AdamW `lr=0.02`

A wider pyramid
(`runs/cifar10_logic_pyramid_mid_9ch_ddp3_plateau/cifar10_20260912_050859`,
`k=384`) reached best val acc **0.6452** under ReduceLROnPlateau. Soft-to-hard
validation gap on that run is about 27 points.

### Legacy Conv-SNN

| Dataset | Run | Best Val | Test |
| --- | --- | ---: | ---: |
| MNIST | `runs/mnist/mnist_20260602_052040` | 99.43% | 99.36% |
| CIFAR-10 | `runs/cifar10/cifar10_20260602_072653` | 88.40% | 88.04% |

These use ordinary `Conv2d` synapses, not logic gates.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

- `torch`, `torchvision`
- `matplotlib` (training curves)
- `tqdm`
- `spikingjelly` (pinned git commit)

### GPU (CUDA)

Device selection in `train.py` / `test.py`:

1. `--device` if set (e.g. `cuda`, `cpu`, `mps`)
2. Else CUDA if `torch.cuda.is_available()`
3. Else MPS (Apple Silicon) if available
4. Else CPU

PyTorch must match your NVIDIA driver (e.g. CUDA 12.4 wheels on a 12.4 driver). Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Example: bind a specific GPU and enable mixed precision:

```bash
CUDA_VISIBLE_DEVICES=4 python train.py \
  --dataset cifar10 \
  --model-family logic_snn \
  --logic-cifar-architecture pyramid \
  --data-root ./data \
  --output-dir ./runs/cifar10 \
  --epochs 100 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --timesteps 4 \
  --device cuda \
  --amp
```

`--amp` only takes effect on CUDA.

## Training

### MNIST Logic-SNN

```bash
python train.py \
  --dataset mnist \
  --model-family logic_snn \
  --data-root ./data \
  --output-dir ./runs/mnist \
  --epochs 100 \
  --batch-size 64 \
  --timesteps 4 \
  --channels 96 \
  --hidden-dim 8192 \
  --logic-hidden-dim2 8192 \
  --logic-identity-middle-init \
  --logic-outputs-per-class 2048 \
  --device cuda \
  --amp
```

MNIST defaults to `logic_snn` when `--model-family` is omitted.

### CIFAR-10 Logic-SNN

CIFAR-10 defaults to legacy `conv_snn`. Pass `--model-family logic_snn` for
logic gates:

```bash
python train.py \
  --dataset cifar10 \
  --model-family logic_snn \
  --logic-cifar-architecture pyramid \
  --data-root ./data \
  --output-dir ./runs/cifar10_logic \
  --epochs 300 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --timesteps 4 \
  --channels 256 \
  --hidden-dim 327680 \
  --logic-hidden-dim2 163840 \
  --logic-outputs-per-class 8192 \
  --device cuda \
  --amp
```

### `train.py` arguments


| Argument                | Default    | Description                                           |
| ----------------------- | ---------- | ----------------------------------------------------- |
| `--dataset`             | (required) | `mnist` or `cifar10`                                  |
| `--model-family`        | dataset default | `logic_snn` (MNIST default) or `conv_snn` (CIFAR default) |
| `--data-root`           | `./data`   | Dataset download root                                 |
| `--output-dir`          | `./runs`   | Parent directory for timestamped runs                 |
| `--epochs`              | `5`        | Number of epochs                                      |
| `--batch-size`          | `64`       | Per-process batch size                                |
| `--grad-accum-steps`    | `1`        | Microbatches accumulated per optimizer update         |
| `--num-workers`         | `2`        | DataLoader workers                                    |
| `--timesteps`           | `4`        | SNN time steps `T`                                    |
| `--channels`            | `64`       | Base conv channel width `k`                           |
| `--hidden-dim`          | `256`      | First dense logic width                               |
| `--dropout`             | `0.25`     | Legacy Conv-SNN only                                  |
| `--neuron`              | `lif`      | `lif` or `if`                                         |
| `--tau`                 | `2.0`      | LIF time constant                                     |
| `--optimizer`           | `adam`     | `adam`, `adamw`, or `sgd`                             |
| `--lr`                  | `1e-3`     | Learning rate                                         |
| `--momentum`            | `0.9`      | SGD momentum                                          |
| `--weight-decay`        | `0.0`      | Weight decay                                          |
| `--lr-scheduler`        | `cosine`   | `cosine` or `plateau`                                 |
| `--scheduler-patience`  | `10`       | Plateau: epochs without val-acc improvement           |
| `--scheduler-factor`    | `0.5`      | Plateau: LR multiplier                                |
| `--min-lr`              | `0.0`      | Lower bound for both schedulers                       |
| `--early-stopping-patience` | `0`    | `0` disables; otherwise stop after stalled val acc    |
| `--early-stopping-min-delta` | `0.001` | Minimum val-acc improvement to reset early stopping |
| `--device`              | auto       | `cuda`, `cpu`, `mps`, or omit for auto                |
| `--seed`                | `42`       | Random seed                                           |
| `--amp`                 | off        | Mixed precision (CUDA only)                           |
| `--resume`              | none       | Checkpoint path; continues in same run dir            |
| `--init-checkpoint`     | none       | Load model weights only and start a fresh optimizer/run |
| `--val-ratio`           | `0.1`      | Fraction of official training set used for validation |
| `--split-seed`          | `42`       | Random seed for deterministic train/val split         |
| `--logic-current-gain`  | `2.2`      | Logic-layer-to-LIF current multiplier                  |
| `--logic-connectivity-seed` | `0`    | Fixed logic connectivity seed                          |
| `--logic-channels-per-tree` | `2`    | Input channels available to each fixed 8-leaf tree     |
| `--logic-cifar-architecture` | `original` | `original` or capacity-scalable `pyramid`          |
| `--logic-hidden-dim2`   | none       | Second dense width (MNIST identity middle / CIFAR pyramid) |
| `--logic-identity-middle-init` | off | MNIST: initialize same-width `fc2` as Boolean `A`  |
| `--logic-growth-lr-multiplier` | `1.0` | LR multiplier for a grown MNIST `fc2`              |
| `--logic-gate-init`     | `residual_a` | Gate-logit initialization                            |
| `--logic-gate-mode`     | `soft`     | `soft` or hard-forward `straight_through`               |
| `--logic-gate-temperature` | `1.0`  | Softmax temperature used by logic gates                 |
| `--logic-outputs-per-class` | `64`   | Final logic neurons assigned to each class             |
| `--logic-readout-loss` | dataset-dependent | Default `cross_entropy` for Logic-SNN; `mse` for Conv-SNN |
| `--logic-readout-temperature` | dataset-dependent | Temperature applied after summing class output nodes |
| `--mnist-binarize-threshold` | `0.5` | MNIST input threshold for Logic-SNN                    |
| `--cifar-logic-thresholds` | `0.25 0.5 0.75` | CIFAR-10 RGB thresholds; produces 9 binary channels |
| `--limit-train-batches` | none       | Cap train batches (smoke test)                        |
| `--limit-val-batches`   | none       | Cap val batches (smoke test)                          |
| `--target-accuracy`     | none       | Stop after reaching this validation accuracy           |


### Training loop (per epoch)

1. Train one epoch on the training split
2. Step the scheduler (`CosineAnnealingLR` once per epoch, or
   `ReduceLROnPlateau` on validation accuracy)
3. Validate on the held-out validation split from the official training set
4. Save `checkpoint_latest.pt`; update `checkpoint_best.pt` if val acc improves
5. Append `train.log` and update `metrics.json` via `plotting.py`
6. After all epochs: write `train_loss.svg`, `val_acc.svg`, and `confusion_matrix.svg`
7. Export `checkpoint_best_hard.pt` and `hard_validation.json` for Logic-SNN

## Testing

After training, evaluate the best checkpoint:

```bash
python test.py --run-dir ./runs/mnist/<run> --inference both --device cuda
```

- Reads `args.json` from the run directory to rebuild the model and data settings  
- Default checkpoint: `<run-dir>/checkpoint_best.pt`  
- Writes `test_metrics.json`; limited smoke tests use a separate suffixed file
- Uses the full test set unless `--limit-test-batches` is set

### `test.py` arguments


| Argument               | Default              | Description                      |
| ---------------------- | -------------------- | -------------------------------- |
| `--run-dir`            | (required)           | Path to a training run directory |
| `--checkpoint`         | `checkpoint_best.pt` | Override checkpoint path         |
| `--hard-checkpoint`    | `checkpoint_best_hard.pt` | Override hard bundle path    |
| `--inference`          | model-dependent      | `soft`, `hard`, or `both`         |
| `--data-root`          | from `args.json`     | Override data root               |
| `--batch-size`         | from `args.json`     | Override batch size              |
| `--num-workers`        | from `args.json`     | Override workers                 |
| `--device`             | auto                 | Same as training                 |
| `--limit-test-batches` | none                 | Cap batches (smoke test only)    |


## Smoke Test

Verify install and data paths with a tiny run:

```bash
python train.py \
  --dataset mnist \
  --epochs 1 \
  --batch-size 8 \
  --timesteps 2 \
  --limit-train-batches 2 \
  --limit-val-batches 2

python test.py \
  --run-dir ./runs/mnist/mnist_<timestamp> \
  --limit-test-batches 2
```

Replace `mnist_<timestamp>` with the directory printed at the end of training.

---

## Detailed Network Structure

Logic-SNN is defined in `snn_minimal/models/logic_snn.py`. Legacy Conv-SNN is
in `snn_minimal/models/conv_snn.py`.

### Common Design

- Multi-step SNN using SpikingJelly `activation_based` modules  
- Static images repeated across `T` time steps (no Poisson encoding)  
- MNIST Logic-SNN input: thresholded `[N, 1, 28, 28]`  
- CIFAR-10 Logic-SNN input: 9 binary channels `[N, 9, 32, 32]`  
- After expansion: `[T, N, C, H, W]`  
- Neurons: `LIFNode` (configurable `tau`) or `IFNode`, surrogate `ATan`  
- Readout: mean firing rate over time and over the class output group → `[N, 10]`

```text
output = mean_t(mean_group(spike_logits[t])), t = 1 ... T
```

### SNN Principle and Spike Equations

Both model families are feed-forward SNNs. Their temporal behavior does not come from video or event streams; it comes from the **internal membrane state** of the spiking neurons.

In this repository, one static image is copied to all `T` time steps:

```text
x[1] = x[2] = ... = x[T] = input_image
```

Even though the input frame is the same at every step, the neuron state evolves over time because each spiking neuron keeps an internal membrane potential and performs threshold-and-reset dynamics.

We use the following notation:

- `x[t]`: input current or input feature at time step `t`
- `h[t]`: membrane potential before spike generation at time step `t`
- `s[t]`: output spike at time step `t`
- `v[t]`: membrane potential after spike/reset at time step `t`
- `V_th`: firing threshold
- `V_reset`: reset voltage

For the current code, the experiments use `--neuron lif`, so the actual runs are based on **LIFNode**. The code can also switch to **IFNode** by passing `--neuron if`.

#### 1. LIF neuron used in the current experiments

Logic-SNN constructs:

```text
neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan())
```

with the default SpikingJelly hard-reset behavior (`v_reset=0`). Conceptually, the forward dynamics are:

```text
h[t] = v[t-1] + (x[t] - (v[t-1] - V_reset)) / tau
s[t] = Theta(h[t] - V_th)
v[t] = V_reset, if s[t] = 1
v[t] = h[t],     if s[t] = 0
```

In the default configuration of SpikingJelly:

- `tau = 2.0` in this project unless changed by `--tau`
- `V_th = 1.0`
- `V_reset = 0.0`
- `decay_input = True`

So with the default settings, the charging equation can be viewed as:

```text
h[t] = v[t-1] + (x[t] - v[t-1]) / 2
```

The logic layer emits values in `[0, 1]`. `--logic-current-gain` (default
`2.2`) multiplies that value before it enters the LIF, so a saturated `1`
logic output is a super-threshold current. With `T=4` and hard reset, this
mostly quantizes each logic output into a small number of firing-rate bins
rather than a long analog trajectory.

Interpretation:

- the neuron integrates incoming input `x[t]`
- at the same time, its membrane leaks toward `V_reset`
- once the membrane crosses threshold, it emits a spike
- after emitting the spike, the membrane is reset

This is why LIF is called **Leaky Integrate-and-Fire**.

#### 2. Optional IF neuron

If you run with `--neuron if`, the neuron becomes an ideal integrator without leakage:

```text
h[t] = v[t-1] + x[t]
s[t] = Theta(h[t] - V_th)
v[t] = V_reset, if s[t] = 1
v[t] = h[t],     if s[t] = 0
```

Compared with LIF:

- IF keeps accumulating input without decay
- LIF adds a leak term, which usually gives more stable temporal dynamics

#### 3. Spike generation formula

The spike itself is binary and is produced by the threshold function:

```text
s[t] = Theta(h[t] - V_th)
```

where:

```text
Theta(z) = 1, if z >= 0
Theta(z) = 0, if z < 0
```

So a spike means:

- membrane potential reached threshold
- the neuron "fires" at that time step

No spike means:

- the neuron keeps integrating information
- the membrane state is preserved for the next time step

#### 4. Why surrogate gradient is needed

The true threshold function `Theta(.)` is not suitable for ordinary backpropagation because its derivative is zero almost everywhere and undefined at the threshold.

So during training, the code uses:

```text
surrogate.ATan()
```

This means:

- **forward pass** still uses a binary spike
- **backward pass** replaces the derivative of the hard threshold with a smooth approximation centered near the threshold

Intuitively, this allows gradients to flow through the spike function and makes end-to-end training possible.

#### 5. Temporal readout in this model

The final classifier layer is also a spiking layer. Logic-SNN groups the last
dense outputs into 10 class bins of `outputs_per_class` neurons, then averages
over the group and over time:

```text
y = (1 / T) * sum_t mean_group(s_out[t])
```

where `y` has shape `[N, 10]`. Prediction is `argmax(y)`.

#### 6. Why state reset is necessary

Because spiking neurons carry membrane state across time, each batch must be reset after forward/backward:

```text
functional.reset_net(model)
```

Otherwise:

- the membrane from batch `n` would leak into batch `n+1`
- training and evaluation would become incorrect

So the temporal recurrence exists **within one sample batch across time steps**, but must not continue across different mini-batches.

### Spiking Block Pattern

Logic-SNN (MNIST and CIFAR):

```text
LogicLayer -> CurrentGain -> SpikingNeuron -> optional MaxPool2d
```

CIFAR-10 legacy Conv-SNN:

```text
Conv2d -> BatchNorm2d -> SpikingNeuron -> Pooling
```

CIFAR-10 Conv-SNN classifier:

```text
Flatten -> Dropout -> Linear -> SpikingNeuron -> Dropout -> Linear -> SpikingNeuron
```

### Logic Gates

Each gate mixes or selects among the 16 two-input Boolean operators in
`logic_ops.py`:

```text
FALSE, AND, A AND NOT B, A, NOT A AND B, B, XOR, OR,
NOR, XNOR, NOT B, B=>A, NOT A, A=>B, NAND, TRUE
```

- **Soft training:** softmax over gate logits, then a bilinear mixture of the
  four truth-table values. Connections are frozen.
- **Hard export:** `argmax` on the 16 logits, then a parameter-free lookup.
- **Straight-through mode:** one-hot in the forward pass, softmax gradient in
  the backward pass.

A convolutional logic channel is an eight-leaf, seven-gate depth-3 tree over a
`3×3` unfold. `--logic-channels-per-tree` (default 2) limits how many input
channels each tree may read. A dense logic neuron always selects two fixed
inputs.

`--logic-identity-middle-init` applies only when MNIST `hidden_dim2 == hidden_dim`.
It rewires `fc2` so each output reads its matching input and biases operator
`A` (index 3). The layer can still learn away from identity.

### MNIST Architectures

Default two-dense Logic-SNN (`channels=64`, `hidden_dim=256`,
`outputs_per_class=64`):

| Stage | Layer | Output shape |
| --- | --- | --- |
| Input | thresholded MNIST image | `[N, 1, 28, 28]` |
| Block 1 | `LogicConv(1→64, k=3, pad=1)` → gain → LIF/IF → max pool | `[N, 64, 14, 14]` |
| Block 2 | `LogicConv(64→128, k=3, pad=1)` → gain → LIF/IF → max pool | `[N, 128, 7, 7]` |
| Dense 1 | flatten → `DiffLogic(6272→256)` → gain → LIF/IF | `[N, 256]` |
| Dense 2 | `DiffLogic(256→640)` → gain → LIF/IF | `[N, 640]` |
| Readout | 10 classes × 64 neurons, mean over group and time | `[N, 10]` |

Best current MNIST run (`mnist_logic_fixed_large_100`):

| Stage | Layer | Output shape |
| --- | --- | --- |
| Input | thresholded MNIST image | `[N, 1, 28, 28]` |
| Block 1 | `LogicConv(1→96)` → gain → LIF → pool | `[N, 96, 14, 14]` |
| Block 2 | `LogicConv(96→192)` → gain → LIF → pool | `[N, 192, 7, 7]` |
| Dense 1 | flatten → `DiffLogic(9408→8192)` → gain → LIF | `[N, 8192]` |
| Dense 2 | `DiffLogic(8192→8192)` identity-`A` init → gain → LIF | `[N, 8192]` |
| Dense 3 | `DiffLogic(8192→20480)` → gain → LIF | `[N, 20480]` |
| Readout | 10 classes × 2048 neurons, mean over group and time | `[N, 10]` |

`mnist_logic_fixed_identity_100` is the same topology at `channels=64`,
`hidden_dim=hidden_dim2=4096`, `outputs_per_class=1024`.

The current gain is applied only between a logic layer and its spiking neuron.
Every logic gate therefore receives values in `[0,1]`, while LIF threshold stays
at `1.0`.

### CIFAR-10 Architectures

The default `original` Logic-SNN keeps the legacy Conv-SNN's
four-convolution, two-dense topology. Every convolution/linear operation is
replaced by its logic counterpart and every BatchNorm/Dropout is removed.


| Stage      | Logic-SNN output shape                                      |
| ---------- | ----------------------------------------------------------- |
| Input      | RGB thresholds `0.25/0.5/0.75` → `[N, 9, 32, 32]`          |
| Blocks 1–2 | `[N, 64, 32, 32]` then `[N, 64, 16, 16]` (with pool)        |
| Blocks 3–4 | `[N, 128, 16, 16]` then `[N, 128, 8, 8]` (with pool)        |
| Dense 1    | `DiffLogic(8192→hidden_dim)`                                 |
| Dense 2    | `DiffLogic(hidden_dim→10×outputs_per_class)`             |
| Readout    | class-group mean followed by temporal mean → `[N, 10]`      |

The `pyramid` architecture increases logic capacity without inserting
non-logic feature extractors. Flatten size after the last pool is `128k`
spatial cells (`32k` channels × `2×2`):

| Stage | Pyramid Logic-SNN (`k = --channels`) |
| --- | --- |
| Input | three thresholds per RGB channel → `[N, 9, 32, 32]` |
| Conv 1 | `LogicConv(9→k)` → gain → LIF → pool → `[N, k, 16, 16]` |
| Conv 2 | `LogicConv(k→4k)` → gain → LIF → pool → `[N, 4k, 8, 8]` |
| Conv 3 | `LogicConv(4k→16k)` → gain → LIF → pool → `[N, 16k, 4, 4]` |
| Conv 4 | `LogicConv(16k→32k)` → gain → LIF → pool → `[N, 32k, 2, 2]` |
| Dense 1 | flatten `128k` → `DiffLogic(hidden_dim)` → gain → LIF |
| Dense 2 | `DiffLogic(logic_hidden_dim2)` → gain → LIF |
| Dense 3 | `DiffLogic(10×outputs_per_class)` → gain → LIF |
| Readout | class-group mean followed by temporal mean → `[N, 10]` |

Best current CIFAR Logic-SNN (`pyramid-m`, `k=256`):

```text
9×32×32
→ 256×16×16 → 1024×8×8 → 4096×4×4 → 8192×2×2
→ flatten 32768
→ 327680 → 163840 → 81920
→ 10-way group mean
```

All four pools operate on binary spikes and therefore implement spatial OR.
Each logic convolution channel remains an eight-leaf, seven-gate depth-3 tree.

### Multi-Step Forward

1. Expand `[N, C, H, W]` → `[T, N, C, H, W]`
2. Forward through encoder + classifier
3. Mean over time (and class groups) → `[N, 10]`

## Loss, Optimizer, and SNN State Reset

**Loss:** Legacy Conv-SNN runs use MSE between firing-rate output and one-hot
labels. Logic-SNN runs (MNIST and CIFAR) sum the per-class output-node rates,
divide by the configured temperature, and use cross-entropy.

```text
logits = output_mean_rate * N / temperature
loss = CrossEntropy(logits, label)
pred = argmax(output_mean_rate, dim=1)
```

`N` is `logic_outputs_per_class`. The current best MNIST run uses
`temperature=20`; the current best CIFAR pyramid uses `temperature=40`.

**Optimizer:** `Adam`, `AdamW`, or `SGD`. Scheduler is either
`CosineAnnealingLR` (stepped once per epoch) or `ReduceLROnPlateau` (stepped
on validation accuracy). Gradient accumulation divides each microbatch loss by
the current accumulation-group size, so a final partial group is also scaled
correctly.

For DDP, `--batch-size` is the per-GPU batch size:

```text
effective_batch = batch_size × number_of_GPUs × grad_accum_steps
```

Accumulation steps that do not update the optimizer use DDP `no_sync()` to
avoid unnecessary gradient communication. Validation samples are strided
across ranks without padding, and loss/correct/sample totals are reduced before
the global metrics are calculated.

**State reset:** after every batch, `functional.reset_net(model)` clears membrane state so batches do not leak temporal information.

## Data Processing

From `snn_minimal/data.py`:

- **MNIST Logic-SNN:** `ToTensor` + threshold at `0.5` (configurable)
- **Legacy MNIST Conv-SNN:** normalization is retained when loading old runs
- **CIFAR-10 Logic-SNN train:** crop/flip augmentation, `ToTensor`, then RGB thresholds at `0.25/0.5/0.75` to produce 9 binary channels
- **CIFAR-10 Logic-SNN test:** `ToTensor` + the same 9-channel threshold encoding
- **CIFAR-10 legacy Conv-SNN:** the original normalization is unchanged

DataLoaders: train `shuffle=True`, `drop_last=True`; test/val `shuffle=False`; `pin_memory=True` on CUDA.

Current split policy:

- `train.py`: official training set is split into train/val (default `90% / 10%`)
- `test.py`: official test set is used only for final evaluation

## Checkpoint Contents

Each `.pt` file stores:

- `model`, `optimizer`, `scheduler` (if present)  
- `epoch`, `best_acc`, `args`

Hard bundles additionally store the frozen connectivity and the chosen operator
index per gate. They have no trainable gate logits.

## Simplifications (by design)

- No Poisson encoder, voting layer, or residual blocks
- No event-based datasets
- Multi-GPU training exists (`train_ddp.py`) but is optional
- Focus: correct, readable Logic-SNN experiments for remote GPU runs
