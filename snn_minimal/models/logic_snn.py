"""Image SNNs built from differentiable or discretized logic gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
from torch import Tensor

from .logic_layers import (
    ConvLogicTreeLayer,
    DiffLogicLayer,
    HardConvLogicTreeLayer,
    HardDiffLogicLayer,
)


HARD_CHECKPOINT_VERSION = 1


def _make_spiking_neuron(neuron_type: str, tau: float) -> nn.Module:
    common = {
        "v_threshold": 1.0,
        "surrogate_function": surrogate.ATan(),
    }
    if neuron_type == "lif":
        return neuron.LIFNode(tau=float(tau), **common)
    if neuron_type == "if":
        return neuron.IFNode(**common)
    raise ValueError(f"Unsupported neuron type: {neuron_type}")


class _LogicSNNBase(nn.Module):
    """Shared temporal forward path for soft and hard gate networks."""

    def __init__(
        self,
        dataset: str,
        timesteps: int,
        channels: int,
        hidden_dim: int,
        outputs_per_class: int,
        current_gain: float,
        neuron_type: str,
        tau: float,
        num_classes: int = 10,
        cifar_architecture: str = "original",
        hidden_dim2: int | None = None,
    ):
        super().__init__()
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if min(channels, hidden_dim, outputs_per_class, num_classes) < 1:
            raise ValueError("network dimensions must be positive")
        if current_gain <= 0:
            raise ValueError("current_gain must be positive")

        self.dataset = str(dataset).lower()
        if self.dataset not in {"mnist", "cifar10"}:
            raise ValueError(f"Unsupported Logic-SNN dataset: {dataset}")
        self.timesteps = int(timesteps)
        self.channels = int(channels)
        self.hidden_dim = int(hidden_dim)
        self.outputs_per_class = int(outputs_per_class)
        self.current_gain = float(current_gain)
        self.neuron_type = str(neuron_type).lower()
        self.tau = float(tau)
        self.num_classes = int(num_classes)
        self.cifar_architecture = str(cifar_architecture).lower()
        self.hidden_dim2 = None if hidden_dim2 is None else int(hidden_dim2)
        if self.dataset == "mnist":
            if self.cifar_architecture != "original":
                raise ValueError("cifar_architecture only applies to CIFAR-10")
            if self.hidden_dim2 is not None and self.hidden_dim2 < 1:
                raise ValueError("hidden_dim2 must be positive")
            self.input_shape = (1, 28, 28)
            self.conv_channel_dims = (self.channels, 2 * self.channels)
            self.pool_after_conv = (True, True)
            self.classifier_in_dim = self.conv_channel_dims[-1] * 7 * 7
            self.dense_output_dims = (self.hidden_dim,)
            if self.hidden_dim2 is not None:
                self.dense_output_dims += (self.hidden_dim2,)
            self.dense_output_dims += (
                self.num_classes * self.outputs_per_class,
            )
        else:
            self.input_shape = (9, 32, 32)
            if self.cifar_architecture == "original":
                self.conv_channel_dims = (
                    self.channels,
                    self.channels,
                    2 * self.channels,
                    2 * self.channels,
                )
                self.pool_after_conv = (False, True, False, True)
                self.classifier_in_dim = self.conv_channel_dims[-1] * 8 * 8
                self.hidden_dim2 = None
                self.dense_output_dims = (
                    self.hidden_dim,
                    self.num_classes * self.outputs_per_class,
                )
            elif self.cifar_architecture == "pyramid":
                self.conv_channel_dims = (
                    self.channels,
                    4 * self.channels,
                    16 * self.channels,
                    32 * self.channels,
                )
                self.pool_after_conv = (True, True, True, True)
                self.classifier_in_dim = self.conv_channel_dims[-1] * 2 * 2
                self.hidden_dim2 = (
                    max(1, self.hidden_dim // 2)
                    if self.hidden_dim2 is None
                    else self.hidden_dim2
                )
                if self.hidden_dim2 < 1:
                    raise ValueError("hidden_dim2 must be positive")
                self.dense_output_dims = (
                    self.hidden_dim,
                    self.hidden_dim2,
                    self.num_classes * self.outputs_per_class,
                )
            else:
                raise ValueError(
                    "cifar_architecture must be 'original' or 'pyramid'"
                )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        num_spike_layers = len(self.conv_channel_dims) + len(self.dense_output_dims)
        for index in range(1, num_spike_layers + 1):
            setattr(
                self,
                f"spike{index}",
                _make_spiking_neuron(self.neuron_type, self.tau),
            )

    def conv_layers(self) -> tuple[nn.Module, ...]:
        return tuple(
            getattr(self, f"conv{index}")
            for index in range(1, len(self.conv_channel_dims) + 1)
        )

    def dense_layers(self) -> tuple[nn.Module, ...]:
        return tuple(
            getattr(self, f"fc{index}")
            for index in range(1, len(self.dense_output_dims) + 1)
        )

    def logic_layers(self) -> tuple[nn.Module, ...]:
        return self.conv_layers() + self.dense_layers()

    def _forward_impl(
        self,
        x: Tensor,
        collect_rates: bool,
    ) -> tuple[Tensor, tuple[Tensor, ...] | None]:
        if x.dim() != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Expected {self.dataset} input [B, {', '.join(map(str, self.input_shape))}], "
                f"got {tuple(x.shape)}"
            )

        timestep_outputs: list[Tensor] = []
        logic_layers = self.logic_layers()
        layer_rates: list[list[Tensor]] = [[] for _ in logic_layers]
        for _ in range(self.timesteps):
            z = x
            rate_index = 0
            for layer_index, (logic_layer, should_pool) in enumerate(
                zip(self.conv_layers(), self.pool_after_conv), start=1
            ):
                spike = getattr(self, f"spike{layer_index}")
                z = spike(self.current_gain * logic_layer(z))
                if collect_rates:
                    layer_rates[rate_index].append(z.mean())
                rate_index += 1
                if should_pool:
                    z = self.pool(z)

            z = z.flatten(1)
            spike_offset = len(self.conv_channel_dims)
            for dense_index, logic_layer in enumerate(self.dense_layers(), start=1):
                spike = getattr(self, f"spike{spike_offset + dense_index}")
                z = spike(self.current_gain * logic_layer(z))
                if collect_rates:
                    layer_rates[rate_index].append(z.mean())
                rate_index += 1

            timestep_outputs.append(
                z.view(z.shape[0], self.num_classes, self.outputs_per_class)
            )

        output = torch.stack(timestep_outputs).mean(dim=(0, 3))
        if not collect_rates:
            return output, None
        rates = tuple(torch.stack(values).mean() for values in layer_rates)
        return output, rates

    def forward(self, x: Tensor) -> Tensor:
        output, _ = self._forward_impl(x, collect_rates=False)
        return output

    def forward_with_spike_rates(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        output, rates = self._forward_impl(x, collect_rates=True)
        assert rates is not None
        return output, rates


class LogicSNN(_LogicSNNBase):
    """Trainable image SNN whose synaptic layers are soft logic gates."""

    def __init__(
        self,
        dataset: str = "mnist",
        timesteps: int = 4,
        channels: int = 64,
        hidden_dim: int = 256,
        outputs_per_class: int = 64,
        current_gain: float = 2.2,
        connectivity_seed: int = 0,
        gate_init: str = "residual_a",
        gate_mode: str = "soft",
        gate_temperature: float = 1.0,
        channels_per_tree: int = 2,
        cifar_architecture: str = "original",
        hidden_dim2: int | None = None,
        identity_middle_init: bool = False,
        neuron_type: str = "lif",
        tau: float = 2.0,
        num_classes: int = 10,
    ):
        super().__init__(
            dataset=dataset,
            timesteps=timesteps,
            channels=channels,
            hidden_dim=hidden_dim,
            outputs_per_class=outputs_per_class,
            current_gain=current_gain,
            neuron_type=neuron_type,
            tau=tau,
            num_classes=num_classes,
            cifar_architecture=cifar_architecture,
            hidden_dim2=hidden_dim2,
        )
        self.connectivity_seed = int(connectivity_seed)
        self.gate_init = str(gate_init)
        self.gate_mode = str(gate_mode)
        self.gate_temperature = float(gate_temperature)
        self.channels_per_tree = int(channels_per_tree)
        self.identity_middle_init = bool(identity_middle_init)
        num_logic_layers = len(self.conv_channel_dims) + len(self.dense_output_dims)
        seeds = [
            self.connectivity_seed + i * 1_000_003
            for i in range(num_logic_layers)
        ]

        input_channels = self.input_shape[0]
        for index, output_channels in enumerate(self.conv_channel_dims):
            setattr(
                self,
                f"conv{index + 1}",
                ConvLogicTreeLayer(
                    input_channels,
                    output_channels,
                    receptive_field_size=3,
                    padding=1,
                    connectivity_seed=seeds[index],
                    gate_init=self.gate_init,
                    gate_mode=self.gate_mode,
                    gate_temperature=self.gate_temperature,
                    channels_per_tree=self.channels_per_tree,
                ),
            )
            input_channels = output_channels

        dense_seed_offset = len(self.conv_channel_dims)
        input_dim = self.classifier_in_dim
        for index, output_dim in enumerate(self.dense_output_dims):
            setattr(
                self,
                f"fc{index + 1}",
                DiffLogicLayer(
                    input_dim,
                    output_dim,
                    connectivity_seed=seeds[dense_seed_offset + index],
                    gate_init=self.gate_init,
                    gate_mode=self.gate_mode,
                    gate_temperature=self.gate_temperature,
                ),
            )
            input_dim = output_dim

        if self.identity_middle_init:
            self.initialize_mnist_middle_as_identity()

    def set_gate_mode(self, mode: str, temperature: float | None = None) -> None:
        for layer in self.logic_layers():
            layer.set_gate_mode(mode, temperature)
        self.gate_mode = self.logic_layers()[0].gate_mode
        self.gate_temperature = self.logic_layers()[0].gate_temperature

    @torch.no_grad()
    def initialize_mnist_middle_as_identity(self) -> None:
        """Initialize a same-width MNIST middle layer as Boolean identity A."""
        if self.dataset != "mnist" or self.hidden_dim2 != self.hidden_dim:
            raise ValueError(
                "Identity middle initialization requires MNIST and "
                "hidden_dim2 == hidden_dim"
            )
        identity_layer = self.fc2
        identity_layer.connectivity[:, 0].copy_(
            torch.arange(self.hidden_dim, device=identity_layer.connectivity.device)
        )
        identity_layer.gate_logits.zero_()
        identity_layer.gate_logits[..., 3] = 5.0

    @torch.no_grad()
    def initialize_from_shallow_mnist(
        self, shallow_state_dict: dict[str, Tensor]
    ) -> None:
        """Grow a two-dense-layer MNIST checkpoint by an identity logic layer."""
        if self.dataset != "mnist" or self.hidden_dim2 != self.hidden_dim:
            raise ValueError(
                "MNIST checkpoint growth requires hidden_dim2 == hidden_dim"
            )

        target = self.state_dict()
        for key, value in shallow_state_dict.items():
            if key.startswith("fc2."):
                grown_key = "fc3." + key.removeprefix("fc2.")
                if grown_key not in target or target[grown_key].shape != value.shape:
                    raise ValueError(f"Cannot map shallow parameter {key!r}")
                target[grown_key] = value.detach().clone()
            elif key in target and target[key].shape == value.shape:
                target[key] = value.detach().clone()

        self.initialize_mnist_middle_as_identity()
        self.load_state_dict(target, strict=True)

    def to_hard(self) -> "HardLogicSNN":
        return HardLogicSNN.from_soft_model(self)


class HardLogicSNN(_LogicSNNBase):
    """Inference-only Logic-SNN containing exact Boolean gate lookups."""

    def __init__(
        self,
        *,
        dataset: str = "mnist",
        timesteps: int,
        channels: int,
        hidden_dim: int,
        outputs_per_class: int,
        current_gain: float,
        neuron_type: str,
        tau: float,
        num_classes: int,
        conv_leaf_indices: list[Tensor],
        conv_gate_ids: list[Tensor],
        dense_connectivities: list[Tensor],
        dense_gate_ids: list[Tensor],
        channels_per_tree: int = 2,
        cifar_architecture: str = "original",
        hidden_dim2: int | None = None,
    ):
        super().__init__(
            dataset=dataset,
            timesteps=timesteps,
            channels=channels,
            hidden_dim=hidden_dim,
            outputs_per_class=outputs_per_class,
            current_gain=current_gain,
            neuron_type=neuron_type,
            tau=tau,
            num_classes=num_classes,
            cifar_architecture=cifar_architecture,
            hidden_dim2=hidden_dim2,
        )
        self.channels_per_tree = int(channels_per_tree)
        expected_conv_layers = len(self.conv_channel_dims)
        if len(conv_leaf_indices) != expected_conv_layers or len(
            conv_gate_ids
        ) != expected_conv_layers:
            raise ValueError(
                f"A {self.dataset} hard Logic-SNN requires "
                f"{expected_conv_layers} convolutional layers"
            )
        expected_dense_layers = len(self.dense_output_dims)
        if len(dense_connectivities) != expected_dense_layers or len(
            dense_gate_ids
        ) != expected_dense_layers:
            raise ValueError(
                f"A hard {self.dataset} Logic-SNN requires "
                f"{expected_dense_layers} dense layers"
            )

        input_channels = self.input_shape[0]
        for index, output_channels in enumerate(self.conv_channel_dims):
            setattr(
                self,
                f"conv{index + 1}",
                HardConvLogicTreeLayer(
                    input_channels,
                    output_channels,
                    3,
                    conv_leaf_indices[index],
                    conv_gate_ids[index],
                    padding=1,
                ),
            )
            input_channels = output_channels

        input_dim = self.classifier_in_dim
        for index, output_dim in enumerate(self.dense_output_dims):
            setattr(
                self,
                f"fc{index + 1}",
                HardDiffLogicLayer(
                    input_dim,
                    output_dim,
                    dense_connectivities[index],
                    dense_gate_ids[index],
                ),
            )
            input_dim = output_dim

    @classmethod
    def from_soft_model(cls, soft: LogicSNN) -> "HardLogicSNN":
        return cls(
            dataset=soft.dataset,
            timesteps=soft.timesteps,
            channels=soft.channels,
            hidden_dim=soft.hidden_dim,
            outputs_per_class=soft.outputs_per_class,
            current_gain=soft.current_gain,
            neuron_type=soft.neuron_type,
            tau=soft.tau,
            num_classes=soft.num_classes,
            channels_per_tree=soft.channels_per_tree,
            cifar_architecture=soft.cifar_architecture,
            hidden_dim2=soft.hidden_dim2,
            conv_leaf_indices=[
                layer.leaf_indices.detach().clone()
                for layer in soft.conv_layers()
            ],
            conv_gate_ids=[
                layer.discretized_gate_ids().detach().clone()
                for layer in soft.conv_layers()
            ],
            dense_connectivities=[
                layer.connectivity.detach().clone()
                for layer in soft.dense_layers()
            ],
            dense_gate_ids=[
                layer.discretized_gate_ids().detach().clone()
                for layer in soft.dense_layers()
            ],
        )

    def to_bundle(self, source: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "format_version": HARD_CHECKPOINT_VERSION,
            "model_family": "hard_logic_snn",
            "config": {
                "dataset": self.dataset,
                "input_channels": self.input_shape[0],
                "timesteps": self.timesteps,
                "channels": self.channels,
                "hidden_dim": self.hidden_dim,
                "outputs_per_class": self.outputs_per_class,
                "current_gain": self.current_gain,
                "neuron_type": self.neuron_type,
                "tau": self.tau,
                "num_classes": self.num_classes,
                "channels_per_tree": self.channels_per_tree,
                "cifar_architecture": self.cifar_architecture,
                "hidden_dim2": self.hidden_dim2,
            },
            "conv_leaf_indices": [
                layer.leaf_indices.detach().cpu()
                for layer in self.conv_layers()
            ],
            "conv_gate_ids": [
                layer.gate_ids.detach().cpu()
                for layer in self.conv_layers()
            ],
            "dense_connectivities": [
                layer.connectivity.detach().cpu()
                for layer in self.dense_layers()
            ],
            "dense_gate_ids": [
                layer.gate_ids.detach().cpu()
                for layer in self.dense_layers()
            ],
            "source": source or {},
        }

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> "HardLogicSNN":
        if int(bundle.get("format_version", -1)) != HARD_CHECKPOINT_VERSION:
            raise ValueError("Unsupported hard Logic-SNN checkpoint version")
        if bundle.get("model_family") != "hard_logic_snn":
            raise ValueError("Checkpoint is not a hard Logic-SNN bundle")
        config = dict(bundle["config"])
        config.setdefault("dataset", "mnist")
        config.setdefault("cifar_architecture", "original")
        config.setdefault("hidden_dim2", None)
        input_channels = int(config.pop("input_channels", 1))
        expected_input_channels = 1 if config["dataset"] == "mnist" else 9
        if input_channels != expected_input_channels:
            raise ValueError(
                f"Hard {config['dataset']} checkpoint uses {input_channels} input "
                f"channels, but the current model requires {expected_input_channels}"
            )
        return cls(
            **config,
            conv_leaf_indices=bundle["conv_leaf_indices"],
            conv_gate_ids=bundle["conv_gate_ids"],
            dense_connectivities=bundle["dense_connectivities"],
            dense_gate_ids=bundle["dense_gate_ids"],
        )


def save_hard_checkpoint(
    path: str | Path,
    hard_model: HardLogicSNN,
    source: dict[str, Any] | None = None,
) -> None:
    torch.save(hard_model.to_bundle(source=source), Path(path))


def load_hard_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[HardLogicSNN, dict[str, Any]]:
    bundle = torch.load(Path(path), map_location=map_location, weights_only=False)
    return HardLogicSNN.from_bundle(bundle), bundle
