"""Export an inference-only hard Logic-SNN from a trained soft checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from snn_minimal.models import LogicSNN, build_model_from_args, save_hard_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Soft Logic-SNN checkpoint")
    parser.add_argument("--output", default=None, help="Output hard checkpoint path")
    return parser.parse_args()


def discretize_checkpoint(checkpoint_path: Path, output_path: Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint.get("args", {})
    if model_args.get("model_family") != "logic_snn":
        raise ValueError("Only a soft Logic-SNN checkpoint can be discretized")

    soft_model = build_model_from_args(model_args)
    if not isinstance(soft_model, LogicSNN):
        raise TypeError("Checkpoint did not reconstruct a LogicSNN")
    soft_model.load_state_dict(checkpoint["model"], strict=True)
    hard_model = soft_model.to_hard()
    save_hard_checkpoint(
        output_path,
        hard_model,
        source={
            "checkpoint": str(checkpoint_path.resolve()),
            "epoch": int(checkpoint.get("epoch", -1)),
            "best_soft_val_acc": float(checkpoint.get("best_acc", 0.0)),
        },
    )
    return output_path


def main() -> None:
    args = parse_args()
    source = Path(args.checkpoint).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Soft checkpoint not found: {source}")
    output = (
        Path(args.output).resolve()
        if args.output
        else source.with_name(f"{source.stem}_hard.pt")
    )
    discretize_checkpoint(source, output)
    print(f"saved hard Logic-SNN checkpoint: {output}")


if __name__ == "__main__":
    main()
