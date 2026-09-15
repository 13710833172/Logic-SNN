from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional
from torch.cuda import amp

from snn_minimal.data import (
    CIFAR10_LOGIC_THRESHOLDS,
    build_train_val_dataloaders,
    get_dataset_spec,
    infer_pin_memory,
)
from snn_minimal.engine import (
    collect_predictions,
    load_checkpoint,
    run_epoch,
    save_checkpoint,
)
from snn_minimal.models import (
    LogicSNN,
    build_model_from_args,
    default_model_family,
    save_hard_checkpoint,
)
from snn_minimal.plotting import setup_recorder


def parse_args():
    parser = argparse.ArgumentParser(description="Train Logic-SNN or legacy Conv-SNN")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument(
        "--model-family",
        choices=["logic_snn", "conv_snn"],
        default=None,
        help="Default: Logic-SNN for MNIST, Conv-SNN for CIFAR-10",
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate this many training batches before each optimizer step",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25, help="CIFAR-10 Conv-SNN only")
    parser.add_argument("--neuron", choices=["lif", "if"], default="lif")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "plateau"],
        default="cosine",
    )
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without a min-delta improvement; 0 disables",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Load model weights only and start a new optimizer/run",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--logic-current-gain", type=float, default=2.2)
    parser.add_argument("--logic-connectivity-seed", type=int, default=0)
    parser.add_argument(
        "--logic-gate-init",
        choices=["residual_a", "normal"],
        default="residual_a",
    )
    parser.add_argument(
        "--logic-gate-mode",
        choices=["soft", "straight_through"],
        default="soft",
    )
    parser.add_argument("--logic-gate-temperature", type=float, default=1.0)
    parser.add_argument("--logic-channels-per-tree", type=int, default=2)
    parser.add_argument(
        "--logic-cifar-architecture",
        choices=["original", "pyramid"],
        default="original",
    )
    parser.add_argument(
        "--logic-hidden-dim2",
        type=int,
        default=None,
        help="Optional second dense logic hidden width",
    )
    parser.add_argument(
        "--logic-growth-lr-multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for a grown MNIST fc2 logic layer",
    )
    parser.add_argument(
        "--logic-identity-middle-init",
        action="store_true",
        help="Initialize a same-width MNIST middle layer as Boolean identity A",
    )
    parser.add_argument("--logic-outputs-per-class", type=int, default=64)
    parser.add_argument(
        "--logic-readout-loss",
        choices=["mse", "cross_entropy"],
        default=None,
        help="Default: cross_entropy for Logic-SNN, mse for Conv-SNN",
    )
    parser.add_argument(
        "--logic-readout-temperature",
        type=float,
        default=None,
        help="Temperature for Logic-SNN group-sum cross-entropy",
    )
    parser.add_argument("--mnist-binarize-threshold", type=float, default=0.5)
    parser.add_argument(
        "--cifar-logic-thresholds",
        type=float,
        nargs=3,
        default=list(CIFAR10_LOGIC_THRESHOLDS),
        metavar=("LOW", "MID", "HIGH"),
        help="CIFAR Logic-SNN RGB thresholds (default: 0.25 0.5 0.75)",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="Stop after reaching this validation accuracy (for example, 0.95)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(args, model):
    parameters = model.parameters()
    if (
        args.logic_growth_lr_multiplier != 1.0
        and isinstance(model, LogicSNN)
        and model.dataset == "mnist"
        and model.hidden_dim2 is not None
    ):
        growth_parameters = list(model.fc2.parameters())
        growth_ids = {id(parameter) for parameter in growth_parameters}
        base_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in growth_ids
        ]
        parameters = [
            {"params": base_parameters, "lr": args.lr},
            {
                "params": growth_parameters,
                "lr": args.lr * args.logic_growth_lr_multiplier,
            },
        ]
    if args.optimizer == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )


def step_scheduler(scheduler, scheduler_name: str, val_acc: float):
    if scheduler_name == "plateau":
        scheduler.step(val_acc)
    else:
        scheduler.step()


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if args.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be non-negative")
    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError("scheduler_factor must be in (0, 1)")
    if args.min_lr < 0.0:
        raise ValueError("min_lr must be non-negative")
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    set_seed(args.seed)

    device = resolve_device(args.device)
    dataset_spec = get_dataset_spec(args.dataset)
    run_args = vars(args).copy()
    run_args["model_family"] = args.model_family or default_model_family(args.dataset)
    is_logic = run_args["model_family"] == "logic_snn"
    is_cifar_logic = args.dataset == "cifar10" and is_logic
    run_args["logic_readout_loss"] = args.logic_readout_loss or (
        "cross_entropy" if is_logic else "mse"
    )
    run_args["logic_readout_temperature"] = (
        args.logic_readout_temperature
        if args.logic_readout_temperature is not None
        else (5.0 if is_logic else 1.0)
    )
    if run_args["logic_readout_temperature"] <= 0:
        raise ValueError("logic_readout_temperature must be positive")
    if run_args["logic_gate_temperature"] <= 0:
        raise ValueError("logic_gate_temperature must be positive")
    if args.target_accuracy is not None and not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")
    if args.logic_growth_lr_multiplier <= 0:
        raise ValueError("logic_growth_lr_multiplier must be positive")
    if run_args["logic_readout_loss"] == "cross_entropy" and not is_logic:
        raise ValueError("Cross-entropy logic readout requires Logic-SNN")
    logit_scale = (
        args.logic_outputs_per_class / run_args["logic_readout_temperature"]
        if run_args["logic_readout_loss"] == "cross_entropy"
        else 1.0
    )

    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_family = resume_payload.get("args", {}).get("model_family", "conv_snn")
        if saved_family != run_args["model_family"]:
            raise ValueError(
                f"Cannot resume {saved_family!r} checkpoint as "
                f"{run_args['model_family']!r}; start a new run instead."
            )
        # Keep writing into the original run folder when resuming.
        run_dir = Path(args.resume).resolve().parent
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2, ensure_ascii=False)

    recorder = setup_recorder(run_dir, resume=bool(args.resume))

    train_loader, val_loader = build_train_val_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=infer_pin_memory(device),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        mnist_binarize_threshold=(
            args.mnist_binarize_threshold
            if args.dataset == "mnist" and run_args["model_family"] == "logic_snn"
            else None
        ),
        cifar_logic_thresholds=(
            args.cifar_logic_thresholds
            if args.dataset == "cifar10" and run_args["model_family"] == "logic_snn"
            else None
        ),
    )

    model = build_model_from_args(run_args).to(device)

    if args.init_checkpoint:
        init_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        init_family = init_payload.get("args", {}).get("model_family", "conv_snn")
        if init_family != run_args["model_family"]:
            raise ValueError(
                f"Cannot initialize {run_args['model_family']!r} from "
                f"{init_family!r} checkpoint"
            )
        source_args = init_payload.get("args", {})
        grows_shallow_mnist = (
            isinstance(model, LogicSNN)
            and args.dataset == "mnist"
            and args.logic_hidden_dim2 == args.hidden_dim
            and source_args.get("logic_hidden_dim2") is None
            and int(source_args.get("hidden_dim", -1)) == args.hidden_dim
            and int(source_args.get("logic_outputs_per_class", -1))
            == args.logic_outputs_per_class
        )
        if grows_shallow_mnist:
            model.initialize_from_shallow_mnist(init_payload["model"])
            print(
                "grew shallow MNIST checkpoint with an identity-initialized "
                "dense logic layer"
            )
        else:
            model.load_state_dict(init_payload["model"], strict=True)
        print(f"initialized model weights from {args.init_checkpoint}")

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = amp.GradScaler()

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    early_stopping_best = best_acc
    epochs_without_improvement = 0

    print(f"device={device}")
    print(
        f"loss={run_args['logic_readout_loss']} "
        f"temperature={run_args['logic_readout_temperature']} "
        f"logit_scale={logit_scale}"
    )
    print(model)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=optimizer,
            scaler=scaler,
            limit_batches=args.limit_train_batches,
            desc=f"train {epoch}",
            loss_name=run_args["logic_readout_loss"],
            logit_scale=logit_scale,
            grad_accum_steps=args.grad_accum_steps,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=None,
            scaler=None,
            limit_batches=args.limit_val_batches,
            desc=f"val {epoch}",
            loss_name=run_args["logic_readout_loss"],
            logit_scale=logit_scale,
        )
        step_scheduler(scheduler, args.lr_scheduler, val_metrics.acc)

        latest_path = run_dir / "checkpoint_latest.pt"
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=max(best_acc, val_metrics.acc),
            args=run_args,
        )

        if val_metrics.acc >= best_acc:
            best_acc = val_metrics.acc
            save_checkpoint(
                path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                args=run_args,
            )

        elapsed = time.time() - epoch_start
        recorder.record_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_acc=best_acc,
            elapsed=elapsed,
            lr=optimizer.param_groups[0]["lr"],
        )

        if val_metrics.acc > early_stopping_best + args.early_stopping_min_delta:
            early_stopping_best = val_metrics.acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.target_accuracy is not None and best_acc >= args.target_accuracy:
            print(
                f"target validation accuracy reached: {best_acc:.4f} "
                f">= {args.target_accuracy:.4f}"
            )
            break
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "early stopping: "
                f"no validation improvement > {args.early_stopping_min_delta:.4f} "
                f"for {epochs_without_improvement} epochs"
            )
            break

    best_ckpt = run_dir / "checkpoint_best.pt"
    best_checkpoint = None
    if best_ckpt.exists():
        best_checkpoint = load_checkpoint(
            path=best_ckpt,
            model=model,
            optimizer=None,
            scheduler=None,
            map_location="cpu",
        )
        model.to(device)

    y_true, y_pred = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        limit_batches=args.limit_val_batches,
        desc="confusion",
    )
    recorder.finalize(
        y_true=y_true,
        y_pred=y_pred,
        class_names=dataset_spec.class_names,
    )

    if isinstance(model, LogicSNN):
        hard_model = model.to_hard().to(device)
        hard_path = run_dir / "checkpoint_best_hard.pt"
        source = {
            "checkpoint": str(best_ckpt.resolve()),
            "epoch": int(best_checkpoint.get("epoch", -1)) if best_checkpoint else -1,
            "best_soft_val_acc": float(best_checkpoint.get("best_acc", 0.0))
            if best_checkpoint
            else 0.0,
        }
        save_hard_checkpoint(hard_path, hard_model, source=source)

        hard_val = run_epoch(
            model=hard_model,
            loader=val_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=None,
            scaler=None,
            limit_batches=args.limit_val_batches,
            desc="hard val",
            loss_name=run_args["logic_readout_loss"],
            logit_scale=logit_scale,
        )

        images, _ = next(iter(val_loader))
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            _, soft_rates = model.forward_with_spike_rates(images)
            functional.reset_net(model)
            _, hard_rates = hard_model.forward_with_spike_rates(images)
            functional.reset_net(hard_model)

        hard_summary = {
            "source_checkpoint": source,
            "hard_checkpoint": str(hard_path.resolve()),
            "soft_val_acc": float(best_acc),
            "hard_val_loss": float(hard_val.loss),
            "hard_val_acc": float(hard_val.acc),
            "soft_hard_val_acc_gap": float(best_acc - hard_val.acc),
            "soft_spike_rates": [float(value) for value in soft_rates],
            "hard_spike_rates": [float(value) for value in hard_rates],
        }
        with (run_dir / "hard_validation.json").open("w", encoding="utf-8") as f:
            json.dump(hard_summary, f, indent=2, ensure_ascii=False)
        print(
            f"hard_val_loss={hard_val.loss:.4f} hard_val_acc={hard_val.acc:.4f} "
            f"soft_hard_gap={best_acc - hard_val.acc:.4f}"
        )
        print(f"saved hard checkpoint: {hard_path}")


if __name__ == "__main__":
    main()
