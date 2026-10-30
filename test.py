from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from snn_minimal.data import (
    CIFAR10_LOGIC_THRESHOLDS,
    build_test_dataloader,
    get_dataset_spec,
    infer_pin_memory,
)
from snn_minimal.engine import load_checkpoint, run_epoch
from snn_minimal.models import build_model_from_args, load_hard_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: <run-dir>/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--hard-checkpoint",
        type=str,
        default=None,
        help="Hard checkpoint path (default: <run-dir>/checkpoint_best_hard.pt)",
    )
    parser.add_argument(
        "--inference",
        choices=["soft", "hard", "both"],
        default=None,
        help="Default: both for Logic-SNN, soft for legacy/CIFAR models",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root. Default uses args.json value.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_args_json(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in run dir: {run_dir}")
    with args_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cli_args = parse_args()

    run_dir = Path(cli_args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    soft_checkpoint_path = (
        Path(cli_args.checkpoint).resolve()
        if cli_args.checkpoint
        else run_dir / "checkpoint_best.pt"
    )
    train_args = _read_args_json(run_dir)
    dataset = train_args["dataset"]
    model_family = train_args.get("model_family", "conv_snn")
    inference = cli_args.inference or (
        "both" if model_family == "logic_snn" else "soft"
    )
    if inference in {"hard", "both"} and model_family != "logic_snn":
        raise ValueError("Hard inference is only available for Logic-SNN runs")

    hard_checkpoint_path = (
        Path(cli_args.hard_checkpoint).resolve()
        if cli_args.hard_checkpoint
        else run_dir / "checkpoint_best_hard.pt"
    )
    if inference in {"soft", "both"} and not soft_checkpoint_path.exists():
        raise FileNotFoundError(f"Soft checkpoint not found: {soft_checkpoint_path}")
    if inference in {"hard", "both"} and not hard_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Hard checkpoint not found: {hard_checkpoint_path}. Run discretize.py first."
        )

    data_root = cli_args.data_root or train_args.get("data_root", "./data")
    batch_size = cli_args.batch_size or int(train_args.get("batch_size", 64))
    num_workers = (
        cli_args.num_workers
        if cli_args.num_workers is not None
        else int(train_args.get("num_workers", 2))
    )
    device = resolve_device(cli_args.device)
    dataset_spec = get_dataset_spec(dataset)
    loss_name = str(train_args.get("logic_readout_loss", "mse"))
    readout_temperature = float(train_args.get("logic_readout_temperature", 1.0))
    logit_scale = (
        int(train_args.get("logic_outputs_per_class", 64)) / readout_temperature
        if loss_name == "cross_entropy"
        else 1.0
    )

    test_loader = build_test_dataloader(
        dataset_name=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=infer_pin_memory(device),
        mnist_binarize_threshold=(
            train_args.get("mnist_binarize_threshold")
            if dataset == "mnist" and model_family == "logic_snn"
            else None
        ),
        cifar_logic_thresholds=(
            train_args.get("cifar_logic_thresholds", CIFAR10_LOGIC_THRESHOLDS)
            if dataset == "cifar10" and model_family == "logic_snn"
            else None
        ),
        dvs_frames=int(train_args.get("timesteps", 4)),
        split_seed=int(train_args.get("split_seed", 42)),
    )

    print(f"device={device}")
    print(f"inference={inference}")
    payload = {
        "model_family": model_family,
        "inference": inference,
        "loss_name": loss_name,
        "logic_readout_temperature": readout_temperature,
    }
    soft_metrics = None
    hard_metrics = None

    if inference in {"soft", "both"}:
        soft_model = build_model_from_args(train_args, legacy_default=True).to(device)
        checkpoint = load_checkpoint(
            path=soft_checkpoint_path,
            model=soft_model,
            optimizer=None,
            scheduler=None,
            map_location="cpu",
        )
        soft_model.to(device)
        soft_metrics = run_epoch(
            model=soft_model,
            loader=test_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=None,
            scaler=None,
            limit_batches=cli_args.limit_test_batches,
            desc="soft test",
            loss_name=loss_name,
            logit_scale=logit_scale,
        )
        payload.update(
            {
                "soft_checkpoint": str(soft_checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "checkpoint_best_val_acc": float(checkpoint.get("best_acc", 0.0)),
                "soft_test_loss": float(soft_metrics.loss),
                "soft_test_acc": float(soft_metrics.acc),
                "soft_test_samples": int(soft_metrics.samples),
            }
        )
        print(
            f"soft_test_loss={soft_metrics.loss:.4f} "
            f"soft_test_acc={soft_metrics.acc:.4f}"
        )

    if inference in {"hard", "both"}:
        hard_model, hard_bundle = load_hard_checkpoint(hard_checkpoint_path)
        hard_model.to(device)
        hard_metrics = run_epoch(
            model=hard_model,
            loader=test_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=None,
            scaler=None,
            limit_batches=cli_args.limit_test_batches,
            desc="hard test",
            loss_name=loss_name,
            logit_scale=logit_scale,
        )
        payload.update(
            {
                "hard_checkpoint": str(hard_checkpoint_path),
                "hard_source": hard_bundle.get("source", {}),
                "hard_test_loss": float(hard_metrics.loss),
                "hard_test_acc": float(hard_metrics.acc),
                "hard_test_samples": int(hard_metrics.samples),
            }
        )
        print(
            f"hard_test_loss={hard_metrics.loss:.4f} "
            f"hard_test_acc={hard_metrics.acc:.4f}"
        )

    if soft_metrics is not None and hard_metrics is not None:
        payload["soft_hard_test_acc_gap"] = float(
            soft_metrics.acc - hard_metrics.acc
        )
        print(f"soft_hard_test_acc_gap={soft_metrics.acc - hard_metrics.acc:.4f}")

    primary = soft_metrics if soft_metrics is not None else hard_metrics
    assert primary is not None
    payload.update(
        {
            "test_loss": float(primary.loss),
            "test_acc": float(primary.acc),
            "test_samples": int(primary.samples),
        }
    )
    if cli_args.limit_test_batches is not None:
        metrics_name = f"test_metrics_{inference}_limited.json"
    elif inference == "hard":
        metrics_name = "test_metrics_hard.json"
    else:
        metrics_name = "test_metrics.json"
    out_path = run_dir / metrics_name
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"saved test metrics: {out_path}")


if __name__ == "__main__":
    main()
