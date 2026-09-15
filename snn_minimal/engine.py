from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    samples: int


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target, num_classes=num_classes).float()


def classification_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    loss_name: str = "mse",
    logit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the training loss and scores used for prediction."""
    if loss_name == "mse":
        return F.mse_loss(outputs, _one_hot(targets, num_classes)), outputs
    if loss_name == "cross_entropy":
        if logit_scale <= 0:
            raise ValueError("logit_scale must be positive")
        logits = outputs * float(logit_scale)
        return F.cross_entropy(logits, targets), logits
    raise ValueError(f"Unsupported loss: {loss_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[amp.GradScaler] = None,
    limit_batches: Optional[int] = None,
    desc: str = "train",
    loss_name: str = "mse",
    logit_scale: float = 1.0,
    grad_accum_steps: int = 1,
    sync_distributed_metrics: bool = False,
) -> EpochMetrics:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    is_distributed = dist.is_available() and dist.is_initialized()
    is_main_process = not is_distributed or dist.get_rank() == 0
    iterator = tqdm(loader, desc=desc, leave=False, disable=not is_main_process)
    amp_enabled = scaler is not None and device.type == "cuda" # amp_enabled 是一个布尔值，用于指示是否启用自动混合精度训练。
    context = torch.enable_grad if is_train else torch.no_grad # context 是一个上下文管理器，用于指示是否启用梯度计算。
    total_batches = len(loader)
    if limit_batches is not None:
        total_batches = min(total_batches, limit_batches)

    with context():
        for step, (images, targets) in enumerate(iterator):
            if limit_batches is not None and step >= limit_batches:
                break

            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            group_start = (step // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, total_batches - group_start)
            if is_train and step == group_start:
                optimizer.zero_grad(set_to_none=True)

            finishes_group = step + 1 == group_start + group_size
            sync_context = nullcontext()
            if is_train and hasattr(model, "no_sync") and not finishes_group:
                sync_context = model.no_sync()

            with sync_context:
                with amp.autocast(enabled=amp_enabled):
                    outputs = model(images)
                    loss, scores = classification_loss(
                        outputs,
                        targets,
                        num_classes,
                        loss_name=loss_name,
                        logit_scale=logit_scale,
                    )

                if is_train:
                    backward_loss = loss / group_size
                    if scaler is not None:
                        scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()

                if is_train and finishes_group:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

            functional.reset_net(model)

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (scores.argmax(dim=1) == targets).sum().item()

            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

    if sync_distributed_metrics:
        if not is_distributed:
            raise RuntimeError("Distributed metric sync requires an initialized process group")
        totals = torch.tensor(
            [total_loss, total_correct, total_samples],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = totals.tolist()

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        acc=total_correct / max(total_samples, 1),
        samples=total_samples,
    )


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: Optional[int] = None,
    desc: str = "confusion",
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    iterator = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for step, (images, targets) in enumerate(iterator):
            if limit_batches is not None and step >= limit_batches:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            functional.reset_net(model)
            preds = outputs.argmax(dim=1).detach().cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.detach().cpu().tolist())

    return y_true, y_pred


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_acc: float,
    args: dict,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": args,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
