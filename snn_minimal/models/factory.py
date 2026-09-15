"""Dataset-aware model construction with legacy checkpoint compatibility."""

from __future__ import annotations

from torch import nn

from .conv_snn import ConvSNN
from .logic_snn import LogicSNN


def default_model_family(dataset: str) -> str:
    return "logic_snn" if dataset.lower() == "mnist" else "conv_snn"


def build_model(
    *,
    dataset: str,
    timesteps: int = 4,
    channels: int = 64,
    hidden_dim: int = 256,
    neuron_type: str = "lif",
    tau: float = 2.0,
    dropout: float = 0.25,
    model_family: str | None = None,
    logic_current_gain: float = 2.2,
    logic_connectivity_seed: int = 0,
    logic_gate_init: str = "residual_a",
    logic_gate_mode: str = "soft",
    logic_gate_temperature: float = 1.0,
    logic_channels_per_tree: int = 2,
    logic_cifar_architecture: str = "original",
    logic_hidden_dim2: int | None = None,
    logic_identity_middle_init: bool = False,
    logic_outputs_per_class: int = 64,
) -> nn.Module:
    family = model_family or default_model_family(dataset)
    if family == "conv_snn":
        return ConvSNN(
            dataset=dataset,
            timesteps=timesteps,
            channels=channels,
            hidden_dim=hidden_dim,
            neuron_type=neuron_type,
            tau=tau,
            dropout=dropout,
        )
    if family == "logic_snn" and dataset.lower() in {"mnist", "cifar10"}:
        return LogicSNN(
            dataset=dataset,
            timesteps=timesteps,
            channels=channels,
            hidden_dim=hidden_dim,
            outputs_per_class=logic_outputs_per_class,
            current_gain=logic_current_gain,
            connectivity_seed=logic_connectivity_seed,
            gate_init=logic_gate_init,
            gate_mode=logic_gate_mode,
            gate_temperature=logic_gate_temperature,
            channels_per_tree=logic_channels_per_tree,
            cifar_architecture=logic_cifar_architecture,
            hidden_dim2=logic_hidden_dim2,
            identity_middle_init=logic_identity_middle_init,
            neuron_type=neuron_type,
            tau=tau,
        )
    raise ValueError(f"Unsupported model family {family!r} for dataset {dataset!r}")


def build_model_from_args(args: dict, legacy_default: bool = False) -> nn.Module:
    family = args.get("model_family")
    if family is None and legacy_default:
        family = "conv_snn"
    return build_model(
        dataset=str(args["dataset"]),
        timesteps=int(args.get("timesteps", 4)),
        channels=int(args.get("channels", 64)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        neuron_type=str(args.get("neuron", "lif")),
        tau=float(args.get("tau", 2.0)),
        dropout=float(args.get("dropout", 0.25)),
        model_family=family,
        logic_current_gain=float(args.get("logic_current_gain", 2.2)),
        logic_connectivity_seed=int(args.get("logic_connectivity_seed", 0)),
        logic_gate_init=str(args.get("logic_gate_init", "residual_a")),
        logic_gate_mode=str(args.get("logic_gate_mode", "soft")),
        logic_gate_temperature=float(args.get("logic_gate_temperature", 1.0)),
        logic_channels_per_tree=int(args.get("logic_channels_per_tree", 2)),
        logic_cifar_architecture=str(
            args.get("logic_cifar_architecture", "original")
        ),
        logic_hidden_dim2=(
            None
            if args.get("logic_hidden_dim2") is None
            else int(args["logic_hidden_dim2"])
        ),
        logic_identity_middle_init=bool(
            args.get("logic_identity_middle_init", False)
        ),
        logic_outputs_per_class=int(args.get("logic_outputs_per_class", 64)),
    )
