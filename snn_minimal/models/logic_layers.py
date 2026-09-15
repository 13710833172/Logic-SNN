"""Soft and hard convolutional/dense logic layers used by Logic-SNN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .logic_ops import build_truth_table, hard_logic_lookup, weighted_sum_16_ops


def build_connectivity(input_dim: int, output_dim: int, seed: int) -> Tensor:
    """Choose two fixed inputs for every dense logic neuron."""
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randint(
        0,
        input_dim,
        (output_dim, 2),
        generator=generator,
        dtype=torch.long,
    )


def build_local_tree_connectivity(
    in_channels: int,
    receptive_field_size: int,
    out_channels: int,
    seed: int,
    num_leaves: int = 8,
    channels_per_tree: int = 2,
) -> Tensor:
    """Choose fixed local leaves for each convolutional logic tree."""
    if min(in_channels, receptive_field_size, out_channels) < 1:
        raise ValueError("channel counts and receptive_field_size must be positive")
    spatial = receptive_field_size**2
    observed_channels = min(channels_per_tree, in_channels)
    generator = torch.Generator().manual_seed(int(seed))
    rows: list[Tensor] = []
    for _ in range(out_channels):
        channels = torch.randperm(in_channels, generator=generator)[:observed_channels]
        candidates = (
            channels[:, None] * spatial + torch.arange(spatial)[None, :]
        ).reshape(-1)
        if candidates.numel() >= num_leaves:
            selection = torch.randperm(candidates.numel(), generator=generator)[:num_leaves]
        else:
            selection = torch.randint(
                candidates.numel(),
                (num_leaves,),
                generator=generator,
            )
        rows.append(candidates[selection])
    return torch.stack(rows).long()


def _initial_gate_logits(shape: tuple[int, ...], gate_init: str) -> Tensor:
    if gate_init == "normal":
        return torch.randn(*shape)
    if gate_init == "residual_a":
        logits = torch.zeros(*shape)
        logits[..., 3] = 5.0  # operator 3 is identity A
        return logits
    raise ValueError(f"Unsupported gate initialization: {gate_init}")


def _validate_gate_mode(mode: str, temperature: float) -> tuple[str, float]:
    mode = str(mode).lower()
    if mode not in {"soft", "straight_through"}:
        raise ValueError(f"Unsupported gate mode: {mode}")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("gate_temperature must be positive")
    return mode, temperature


def _gate_probabilities(logits: Tensor, mode: str, temperature: float) -> Tensor:
    soft = F.softmax(logits / temperature, dim=-1)
    if mode == "soft":
        return soft
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).to(soft.dtype)
    return hard + (soft - soft.detach())


class DiffLogicLayer(nn.Module):
    """Differentiable dense logic layer with fixed two-input connectivity."""

    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        connectivity_seed: int,
        gate_init: str = "residual_a",
        gate_mode: str = "soft",
        gate_temperature: float = 1.0,
        connectivity: Tensor | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_neurons = int(num_neurons)
        self.gate_mode, self.gate_temperature = _validate_gate_mode(
            gate_mode, gate_temperature
        )
        selected = (
            build_connectivity(self.input_dim, self.num_neurons, connectivity_seed)
            if connectivity is None
            else connectivity.long()
        )
        if selected.shape != (self.num_neurons, 2):
            raise ValueError("connectivity must have shape [num_neurons, 2]")
        self.register_buffer("connectivity", selected, persistent=True)
        self.gate_logits = nn.Parameter(
            _initial_gate_logits((self.num_neurons, 16), gate_init)
        )

    def gate_probs(self) -> Tensor:
        return _gate_probabilities(
            self.gate_logits, self.gate_mode, self.gate_temperature
        )

    def set_gate_mode(self, mode: str, temperature: float | None = None) -> None:
        value = self.gate_temperature if temperature is None else temperature
        self.gate_mode, self.gate_temperature = _validate_gate_mode(mode, value)

    def discretized_gate_ids(self) -> Tensor:
        return self.gate_logits.argmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected [B, {self.input_dim}] input, got {tuple(x.shape)}"
            )
        a = x[:, self.connectivity[:, 0]]
        b = x[:, self.connectivity[:, 1]]
        return weighted_sum_16_ops(a, b, self.gate_probs().unsqueeze(0))


class ConvLogicTreeLayer(nn.Module):
    """Depth-3 convolutional logic trees with eight fixed local leaves."""

    NUM_LEAVES = 8
    NUM_GATES = 7

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        receptive_field_size: int,
        connectivity_seed: int,
        gate_init: str = "residual_a",
        gate_mode: str = "soft",
        gate_temperature: float = 1.0,
        channels_per_tree: int = 2,
        stride: int = 1,
        padding: int | None = None,
        leaf_indices: Tensor | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.receptive_field_size = int(receptive_field_size)
        self.stride = int(stride)
        self.channels_per_tree = int(channels_per_tree)
        if self.channels_per_tree < 1:
            raise ValueError("channels_per_tree must be positive")
        self.gate_mode, self.gate_temperature = _validate_gate_mode(
            gate_mode, gate_temperature
        )
        self.padding = (
            self.receptive_field_size // 2 if padding is None else int(padding)
        )
        selected = (
            build_local_tree_connectivity(
                self.in_channels,
                self.receptive_field_size,
                self.out_channels,
                connectivity_seed,
                num_leaves=self.NUM_LEAVES,
                channels_per_tree=self.channels_per_tree,
            )
            if leaf_indices is None
            else leaf_indices.long()
        )
        if selected.shape != (self.out_channels, self.NUM_LEAVES):
            raise ValueError("leaf_indices must have shape [out_channels, 8]")
        self.register_buffer("leaf_indices", selected, persistent=True)
        self.gate_logits = nn.Parameter(
            _initial_gate_logits((self.out_channels, self.NUM_GATES, 16), gate_init)
        )

    def gate_probs(self) -> Tensor:
        return _gate_probabilities(
            self.gate_logits, self.gate_mode, self.gate_temperature
        )

    def set_gate_mode(self, mode: str, temperature: float | None = None) -> None:
        value = self.gate_temperature if temperature is None else temperature
        self.gate_mode, self.gate_temperature = _validate_gate_mode(mode, value)

    def discretized_gate_ids(self) -> Tensor:
        return self.gate_logits.argmax(dim=-1)

    def _gate(self, a: Tensor, b: Tensor, probs: Tensor) -> Tensor:
        return weighted_sum_16_ops(
            a,
            b,
            probs.view(1, self.out_channels, 1, 16),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, H, W], got {tuple(x.shape)}"
            )
        batch, _, height, width = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.receptive_field_size,
            padding=self.padding,
            stride=self.stride,
        )
        height_out = (
            height + 2 * self.padding - self.receptive_field_size
        ) // self.stride + 1
        width_out = (
            width + 2 * self.padding - self.receptive_field_size
        ) // self.stride + 1
        leaves = [
            patches[:, self.leaf_indices[:, index], :]
            for index in range(self.NUM_LEAVES)
        ]
        probs = self.gate_probs()
        level_1 = [
            self._gate(leaves[index], leaves[index + 1], probs[:, index // 2])
            for index in range(0, 8, 2)
        ]
        level_2 = [
            self._gate(level_1[0], level_1[1], probs[:, 4]),
            self._gate(level_1[2], level_1[3], probs[:, 5]),
        ]
        root = self._gate(level_2[0], level_2[1], probs[:, 6])
        return root.view(batch, self.out_channels, height_out, width_out)


class HardDiffLogicLayer(nn.Module):
    """Non-trainable dense layer evaluated with Boolean truth-table lookups."""

    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        connectivity: Tensor,
        gate_ids: Tensor,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_neurons = int(num_neurons)
        if connectivity.shape != (self.num_neurons, 2):
            raise ValueError("connectivity must have shape [num_neurons, 2]")
        if gate_ids.shape != (self.num_neurons,):
            raise ValueError("gate_ids must have shape [num_neurons]")
        self.register_buffer("connectivity", connectivity.long(), persistent=True)
        self.register_buffer("gate_ids", gate_ids.long(), persistent=True)
        self.register_buffer(
            "truth_table",
            build_truth_table(device=connectivity.device),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        a = x[:, self.connectivity[:, 0]]
        b = x[:, self.connectivity[:, 1]]
        return hard_logic_lookup(a, b, self.gate_ids.unsqueeze(0), self.truth_table).float()


class HardConvLogicTreeLayer(nn.Module):
    """Non-trainable convolutional logic trees with fixed Boolean gates."""

    NUM_LEAVES = 8
    NUM_GATES = 7

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        receptive_field_size: int,
        leaf_indices: Tensor,
        gate_ids: Tensor,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.receptive_field_size = int(receptive_field_size)
        self.stride = int(stride)
        self.padding = (
            self.receptive_field_size // 2 if padding is None else int(padding)
        )
        if leaf_indices.shape != (self.out_channels, self.NUM_LEAVES):
            raise ValueError("leaf_indices must have shape [out_channels, 8]")
        if gate_ids.shape != (self.out_channels, self.NUM_GATES):
            raise ValueError("gate_ids must have shape [out_channels, 7]")
        self.register_buffer("leaf_indices", leaf_indices.long(), persistent=True)
        self.register_buffer("gate_ids", gate_ids.long(), persistent=True)
        self.register_buffer(
            "truth_table",
            build_truth_table(device=leaf_indices.device),
            persistent=False,
        )

    def _gate(self, a: Tensor, b: Tensor, gate_ids: Tensor) -> Tensor:
        ids = gate_ids.view(1, self.out_channels, 1)
        return hard_logic_lookup(a, b, ids, self.truth_table).float()

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.receptive_field_size,
            padding=self.padding,
            stride=self.stride,
        )
        height_out = (
            height + 2 * self.padding - self.receptive_field_size
        ) // self.stride + 1
        width_out = (
            width + 2 * self.padding - self.receptive_field_size
        ) // self.stride + 1
        leaves = [
            patches[:, self.leaf_indices[:, index], :]
            for index in range(self.NUM_LEAVES)
        ]
        level_1 = [
            self._gate(leaves[index], leaves[index + 1], self.gate_ids[:, index // 2])
            for index in range(0, 8, 2)
        ]
        level_2 = [
            self._gate(level_1[0], level_1[1], self.gate_ids[:, 4]),
            self._gate(level_1[2], level_1[3], self.gate_ids[:, 5]),
        ]
        root = self._gate(level_2[0], level_2[1], self.gate_ids[:, 6])
        return root.view(batch, self.out_channels, height_out, width_out)
