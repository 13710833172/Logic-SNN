from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


def _make_spiking_neuron(neuron_type: str, tau: float):
    neuron_type = neuron_type.lower()
    if neuron_type == "lif":
        return partial(
            neuron.LIFNode,
            tau=float(tau),
            surrogate_function=surrogate.ATan(), #替代梯度函数
        )
    if neuron_type == "if":
        return partial(
            neuron.IFNode,
            surrogate_function=surrogate.ATan(),
        )
    raise ValueError(f"Unsupported neuron type: {neuron_type}")


class ConvSNN(nn.Module):
    def __init__(
        self,
        dataset: str,
        timesteps: int = 4,
        channels: int = 64,
        hidden_dim: int = 256,
        neuron_type: str = "lif",
        tau: float = 2.0,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.timesteps = timesteps

        spike = _make_spiking_neuron(neuron_type=neuron_type, tau=tau)

        if self.dataset == "mnist":
            self.encoder = nn.Sequential(
                layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
                layer.Conv2d(
                    channels, channels * 2, kernel_size=3, padding=1, bias=False
                ),
                layer.BatchNorm2d(channels * 2),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
            )
            classifier_in_dim = channels * 2 * 7 * 7

        elif self.dataset == "cifar10":
            self.encoder = nn.Sequential(
                layer.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spike(),
                layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
                layer.Conv2d(
                    channels, channels * 2, kernel_size=3, padding=1, bias=False
                ),
                layer.BatchNorm2d(channels * 2),
                spike(),
                layer.Conv2d(
                    channels * 2,
                    channels * 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                layer.BatchNorm2d(channels * 2),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
            )
            classifier_in_dim = channels * 2 * 8 * 8

        elif self.dataset == "cifar10_dvs":
            # Same CIFAR-10 stack, with polarity frames resized to 32x32.
            self.encoder = nn.Sequential(
                layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spike(),
                layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
                layer.Conv2d(
                    channels, channels * 2, kernel_size=3, padding=1, bias=False
                ),
                layer.BatchNorm2d(channels * 2),
                spike(),
                layer.Conv2d(
                    channels * 2,
                    channels * 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                layer.BatchNorm2d(channels * 2),
                spike(),
                layer.MaxPool2d(kernel_size=2, stride=2),
            )
            classifier_in_dim = channels * 2 * 8 * 8

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(classifier_in_dim, hidden_dim, bias=False),
            spike(),
            layer.Dropout(dropout),
            layer.Linear(hidden_dim, 10, bias=False),
            spike(),
        )

        functional.kaiming_normal_conv_linear_weight(self)
        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset == "cifar10_dvs":
            if x.dim() != 5 or x.size(1) != self.timesteps or x.size(2) != 2:
                raise ValueError(
                    "CIFAR10-DVS expects input "
                    f"[B, {self.timesteps}, 2, H, W], got {tuple(x.shape)}"
                )
            x_seq = x.transpose(0, 1).contiguous()
        else:
            x_seq = x.unsqueeze(0).expand(self.timesteps, -1, -1, -1, -1)
        x_seq = self.encoder(x_seq)
        x_seq = self.classifier(x_seq)
        return x_seq.mean(0)
