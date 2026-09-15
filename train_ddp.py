from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from spikingjelly.activation_based import functional
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from snn_minimal.data import build_train_val_dataloaders, get_dataset_spec
from snn_minimal.engine import (
    collect_predictions,
    load_checkpoint,
    run_epoch,
    save_checkpoint,
)
from snn_minimal.models import LogicSNN, build_model_from_args, default_model_family
from snn_minimal.models import save_hard_checkpoint
from snn_minimal.plotting import setup_recorder
from train import (
    build_optimizer,
    build_scheduler,
    parse_args,
    set_seed,
    step_scheduler,
)


class StridedDistributedSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicating samples."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = max(0, len(self.dataset) - self.rank)
        return (remaining + self.world_size - 1) // self.world_size


def _distributed_loader(
    loader: DataLoader,
    rank: int,
    world_size: int,
    is_train: bool,
    seed: int,
) -> tuple[DataLoader, Sampler[int]]:
    if is_train:
        sampler = DistributedSampler(
            loader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
    else:
        sampler = StridedDistributedSampler(loader.dataset, rank, world_size)

    distributed_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
        drop_last=is_train,
    )
    return distributed_loader, sampler


def _resolve_run_args(args) -> tuple[dict, float]:
    run_args = vars(args).copy()
    run_args["model_family"] = args.model_family or default_model_family(args.dataset)
    is_logic = run_args["model_family"] == "logic_snn"
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
    if args.target_accuracy is not None and not 0.0 < args.target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")

    logit_scale = (
        args.logic_outputs_per_class / run_args["logic_readout_temperature"]
        if run_args["logic_readout_loss"] == "cross_entropy"
        else 1.0
    )
    return run_args, logit_scale


def _shared_run_dir(args, rank: int) -> Path:
    value = None
    if rank == 0:
        if args.resume:
            value = str(Path(args.resume).resolve().parent)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            value = str((Path(args.output_dir) / f"{args.dataset}_{timestamp}").resolve())
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return Path(values[0])


def _rank_zero_finalize(
    args,
    run_args: dict,
    run_dir: Path,
    model,
    val_loader: DataLoader,
    device: torch.device,
    best_acc: float,
    logit_scale: float,
    recorder,
):
    dataset_spec = get_dataset_spec(args.dataset)
    best_ckpt = run_dir / "checkpoint_best.pt"
    best_checkpoint = None
    if best_ckpt.exists():
        best_checkpoint = load_checkpoint(best_ckpt, model=model, map_location="cpu")
        model.to(device)

    y_true, y_pred = collect_predictions(
        model,
        val_loader,
        device,
        limit_batches=args.limit_val_batches,
        desc="confusion",
    )
    recorder.finalize(y_true, y_pred, dataset_spec.class_names)

    if not isinstance(model, LogicSNN):
        return

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
        hard_model,
        val_loader,
        device,
        dataset_spec.num_classes,
        desc="hard val",
        limit_batches=args.limit_val_batches,
        loss_name=run_args["logic_readout_loss"],
        logit_scale=logit_scale,
    )

    images, _ = next(iter(val_loader))
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        _, soft_rates = model.forward_with_spike_rates(images)
        functional.reset_net(model)
        _, hard_rates = hard_model.forward_with_spike_rates(images)
        functional.reset_net(hard_model)

    summary = {
        "source_checkpoint": source,
        "hard_checkpoint": str(hard_path.resolve()),
        "soft_val_acc": float(best_acc),
        "hard_val_loss": float(hard_val.loss),
        "hard_val_acc": float(hard_val.acc),
        "soft_hard_val_acc_gap": float(best_acc - hard_val.acc),
        "soft_spike_rates": [float(value) for value in soft_rates],
        "hard_spike_rates": [float(value) for value in hard_rates],
    }
    with (run_dir / "hard_validation.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(
        f"hard_val_loss={hard_val.loss:.4f} hard_val_acc={hard_val.acc:.4f} "
        f"soft_hard_gap={best_acc - hard_val.acc:.4f}"
    )


def distributed_main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL distributed training requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    try:
        set_seed(args.seed)
        run_args, logit_scale = _resolve_run_args(args)
        run_args["distributed_world_size"] = world_size
        run_args["effective_batch_size"] = (
            args.batch_size * args.grad_accum_steps * world_size
        )
        dataset_spec = get_dataset_spec(args.dataset)
        run_dir = _shared_run_dir(args, rank)

        if args.resume:
            resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            saved_family = resume_payload.get("args", {}).get("model_family", "conv_snn")
            if saved_family != run_args["model_family"]:
                raise ValueError(
                    f"Cannot resume {saved_family!r} checkpoint as "
                    f"{run_args['model_family']!r}"
                )

        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "args.json").open("w", encoding="utf-8") as file:
                json.dump(run_args, file, indent=2, ensure_ascii=False)
        dist.barrier()

        full_train_loader, full_val_loader = build_train_val_dataloaders(
            dataset_name=args.dataset,
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
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
        train_loader, train_sampler = _distributed_loader(
            full_train_loader, rank, world_size, is_train=True, seed=args.seed
        )
        val_loader, _ = _distributed_loader(
            full_val_loader, rank, world_size, is_train=False, seed=args.seed
        )

        model = build_model_from_args(run_args).to(device)
        if args.init_checkpoint:
            payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)

        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)
        scaler = amp.GradScaler() if args.amp else None

        start_epoch = 0
        best_acc = 0.0
        if args.resume:
            checkpoint = load_checkpoint(
                Path(args.resume),
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cpu",
            )
            start_epoch = int(checkpoint["epoch"]) + 1
            best_acc = float(checkpoint.get("best_acc", 0.0))

        early_stopping_best = best_acc
        epochs_without_improvement = 0

        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        set_seed(args.seed + rank)
        recorder = setup_recorder(run_dir, resume=bool(args.resume)) if rank == 0 else None

        if rank == 0:
            print(
                f"DDP world_size={world_size} per_gpu_batch={args.batch_size} "
                f"grad_accum={args.grad_accum_steps} "
                f"effective_batch={run_args['effective_batch_size']}"
            )
            print(
                f"loss={run_args['logic_readout_loss']} "
                f"temperature={run_args['logic_readout_temperature']} "
                f"logit_scale={logit_scale}"
            )

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            train_sampler.set_epoch(epoch)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                dataset_spec.num_classes,
                optimizer=optimizer,
                scaler=scaler,
                limit_batches=args.limit_train_batches,
                desc=f"train {epoch}",
                loss_name=run_args["logic_readout_loss"],
                logit_scale=logit_scale,
                grad_accum_steps=args.grad_accum_steps,
                sync_distributed_metrics=True,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                dataset_spec.num_classes,
                limit_batches=args.limit_val_batches,
                desc=f"val {epoch}",
                loss_name=run_args["logic_readout_loss"],
                logit_scale=logit_scale,
                sync_distributed_metrics=True,
            )
            step_scheduler(scheduler, args.lr_scheduler, val_metrics.acc)

            if val_metrics.acc > early_stopping_best + args.early_stopping_min_delta:
                early_stopping_best = val_metrics.acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            best_acc = max(best_acc, val_metrics.acc)
            if rank == 0:
                base_model = model.module
                save_checkpoint(
                    run_dir / "checkpoint_latest.pt",
                    base_model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_acc,
                    run_args,
                )
                if val_metrics.acc >= best_acc:
                    save_checkpoint(
                        run_dir / "checkpoint_best.pt",
                        base_model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_acc,
                        run_args,
                    )
                recorder.record_epoch(
                    epoch,
                    train_metrics,
                    val_metrics,
                    best_acc,
                    time.time() - epoch_start,
                    optimizer.param_groups[0]["lr"],
                )

            reached_target = (
                args.target_accuracy is not None and best_acc >= args.target_accuracy
            )
            stopped_early = (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            )
            stop = reached_target or stopped_early
            stop_tensor = torch.tensor(int(stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            dist.barrier()
            if stop_tensor.item():
                if rank == 0:
                    if reached_target:
                        print(f"target validation accuracy reached: {best_acc:.4f}")
                    else:
                        print(
                            "early stopping: "
                            f"no validation improvement > "
                            f"{args.early_stopping_min_delta:.4f} for "
                            f"{epochs_without_improvement} epochs"
                        )
                break

        dist.barrier()
        if rank == 0:
            _rank_zero_finalize(
                args,
                run_args,
                run_dir,
                model.module,
                full_val_loader,
                device,
                best_acc,
                logit_scale,
                recorder,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    distributed_main()
