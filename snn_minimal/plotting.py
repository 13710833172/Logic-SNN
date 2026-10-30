from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence, TextIO

import matplotlib
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_minimal.engine import EpochMetrics

_CM_STYLE = {
    "font.family": "Liberation Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.7,
    "svg.fonttype": "path",
}
_CM_CMAP = "Blues"
_CM_TITLES = {
    "mnist": "Confusion matrix (MNIST)",
    "cifar10": "Confusion matrix (CIFAR-10)",
    "cifar10_dvs": "Confusion matrix (CIFAR10-DVS)",
}

LOG_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[0-9]*\.?[0-9]+)\s+"
    r"train_acc=(?P<train_acc>[0-9]*\.?[0-9]+)\s+"
    r"val_loss=(?P<val_loss>[0-9]*\.?[0-9]+)\s+"
    r"val_acc=(?P<val_acc>[0-9]*\.?[0-9]+)\s+"
    r"best_val_acc=(?P<best_val_acc>[0-9]*\.?[0-9]+)\s+"
    r"time=(?P<time>[0-9]*\.?[0-9]+)s\s+"
    r"lr=(?P<lr>[0-9]*\.?[0-9]+)"
)


def empty_history() -> dict[str, list[float] | list[int]]:
    return {"epochs": [], "train_loss": [], "val_acc": []}


def _parse_history_from_log(path: Path) -> dict[str, list[float] | list[int]]:
    history = empty_history()
    if not path.exists():
        return history

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.match(line.strip())
            if match is None:
                continue
            history["epochs"].append(int(match.group("epoch")))
            history["train_loss"].append(float(match.group("train_loss")))
            history["val_acc"].append(float(match.group("val_acc")))
    return history


def _save_metrics_summary(path: Path, history: dict[str, list[float] | list[int]]) -> None:
    if not history["epochs"]:
        payload = {"final": None, "best": None}
    else:
        final_idx = len(history["epochs"]) - 1
        best_val = max(history["val_acc"])
        best_idx = max(
            idx for idx, val_acc in enumerate(history["val_acc"]) if val_acc == best_val
        )
        payload = {
            "final": {
                "epoch": int(history["epochs"][final_idx]),
                "train_loss": float(history["train_loss"][final_idx]),
                "val_acc": float(history["val_acc"][final_idx]),
            },
            "best": {
                "epoch": int(history["epochs"][best_idx]),
                "train_loss": float(history["train_loss"][best_idx]),
                "val_acc": float(history["val_acc"][best_idx]),
            },
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_training_curves(
    history: dict[str, list[float] | list[int]], run_dir: Path
) -> tuple[Path, Path]:
    loss_path = run_dir / "train_loss.svg"
    acc_path = run_dir / "val_acc.svg"
    epochs = history["epochs"]
    if not epochs:
        return loss_path, acc_path

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(loss_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["val_acc"], marker="o", linewidth=1.5, color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(acc_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return loss_path, acc_path


def confusion_matrix_title(dataset: str) -> str:
    return _CM_TITLES.get(dataset.lower(), "Confusion matrix")


def _confusion_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]],
) -> tuple[np.ndarray, list[str]]:
    labels = np.array(y_true, dtype=int)
    preds = np.array(y_pred, dtype=int)
    if class_names:
        names = list(class_names)
        num_classes = len(names)
    else:
        observed = 0
        if labels.size:
            observed = max(observed, int(labels.max()))
        if preds.size:
            observed = max(observed, int(preds.max()))
        num_classes = observed + 1
        names = [str(i) for i in range(num_classes)]

    matrix = np.zeros((num_classes, num_classes), dtype=float)
    for true_label, pred_label in zip(labels, preds, strict=True):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            matrix[true_label, pred_label] += 1
    return matrix, names


def _row_percentages(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    percentages = np.zeros_like(counts, dtype=float)
    np.divide(counts, totals, out=percentages, where=totals > 0)
    return percentages * 100.0


def _annotation_color(value: float, cmap, norm: Normalize) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.45 else "black"


def _paint_matrix(ax, percentages: np.ndarray, title: str, fontsize: float, grid_lw: float, title_pad: float, cmap, norm: Normalize) -> None:
    num_classes = percentages.shape[0]
    ax.imshow(
        percentages,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        zorder=1,
    )
    for edge in range(num_classes + 1):
        ax.axhline(edge - 0.5, color="white", linewidth=grid_lw, zorder=2)
        ax.axvline(edge - 0.5, color="white", linewidth=grid_lw, zorder=2)
    ax.tick_params(axis="both", which="both", length=3.0, width=0.7, direction="out", pad=2)
    ax.set_title(title, pad=title_pad)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)
        spine.set_zorder(4)
    for row in range(num_classes):
        for col in range(num_classes):
            value = percentages[row, col]
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=_annotation_color(value, cmap, norm),
                fontsize=fontsize,
                zorder=3,
            )


def _set_category_ticks(ax, labels: Sequence[str], rotate_x: bool, tick_size: float) -> None:
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=tick_size)
    ax.set_yticklabels(labels, fontsize=tick_size)
    if rotate_x:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


def _add_left_colorbar(fig, cax, cmap, norm: Normalize) -> None:
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[0, 100])
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label("Pct. (%)", labelpad=6)
    colorbar.ax.tick_params(length=3.0, width=0.7, labelsize=8.5, pad=2)
    colorbar.outline.set_linewidth(0.7)


def _save_digit_confusion(percentages: np.ndarray, names: Sequence[str], title: str, out_path: Path, cmap, norm: Normalize) -> None:
    fig, ax = plt.subplots(figsize=(5.15, 4.55))
    _paint_matrix(ax, percentages, title, 7.2, 1.15, 8, cmap, norm)
    _set_category_ticks(ax, names, rotate_x=False, tick_size=8.5)
    ax.set_xlabel("Predicted categories", labelpad=4)
    ax.set_ylabel("Ground truth categories")
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        location="left",
        fraction=0.046,
        pad=0.16,
        ticks=[0, 100],
    )
    colorbar.ax.yaxis.set_ticks_position("left")
    colorbar.ax.yaxis.set_label_position("left")
    colorbar.set_label("Pct. (%)", labelpad=6)
    colorbar.ax.tick_params(length=3.0, width=0.7, labelsize=8.5, pad=2)
    colorbar.outline.set_linewidth(0.7)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _save_named_confusion(percentages: np.ndarray, names: Sequence[str], title: str, out_path: Path, cmap, norm: Normalize) -> None:
    fig_w, fig_h = 5.67, 5.05
    heat, bottom, left_x = 3.55, 0.92, 1.67
    cbar_x, cbar_w = 0.52, 0.15
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left_x / fig_w, bottom / fig_h, heat / fig_w, heat / fig_h])
    cax = fig.add_axes([cbar_x / fig_w, bottom / fig_h, cbar_w / fig_w, heat / fig_h])
    _paint_matrix(ax, percentages, title, 6.7, 1.05, 7, cmap, norm)
    _set_category_ticks(ax, names, rotate_x=True, tick_size=8.0)
    ax.set_xlabel("Predicted categories", labelpad=4)
    ax.set_ylabel("Ground truth categories", labelpad=4)
    _add_left_colorbar(fig, cax, cmap, norm)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    run_dir: Path,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
) -> Path:
    out_path = run_dir / "confusion_matrix.svg"
    if not y_true:
        return out_path

    counts, names = _confusion_counts(y_true, y_pred, class_names)
    percentages = _row_percentages(counts)
    cmap = plt.get_cmap(_CM_CMAP)
    norm = Normalize(vmin=0, vmax=100)
    figure_title = title or "Confusion matrix"
    named = any(len(name) > 1 for name in names)
    with plt.rc_context(_CM_STYLE):
        if named:
            _save_named_confusion(percentages, names, figure_title, out_path, cmap, norm)
        else:
            _save_digit_confusion(percentages, names, figure_title, out_path, cmap, norm)
    return out_path


class TrainingRecorder:
    def __init__(self, run_dir: Path, resume: bool = False):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.json"
        self.log_path = run_dir / "train.log"
        if resume:
            self.history = _parse_history_from_log(self.log_path)
        else:
            self.history = empty_history()
        log_mode = "a" if resume else "w"
        self._log_file: TextIO = self.log_path.open(log_mode, encoding="utf-8")
        _save_metrics_summary(self.metrics_path, self.history)

    def record_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        best_acc: float,
        elapsed: float,
        lr: float,
    ) -> str:
        line = (
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.acc:.4f} "
            f"best_val_acc={best_acc:.4f} time={elapsed:.1f}s lr={lr:.6f}"
        )
        self._log_file.write(line + "\n")
        self._log_file.flush()

        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_metrics.loss)
        self.history["val_acc"].append(val_metrics.acc)
        _save_metrics_summary(self.metrics_path, self.history)

        print(line)
        return line

    def finalize(
        self,
        y_true: Optional[Sequence[int]] = None,
        y_pred: Optional[Sequence[int]] = None,
        class_names: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ) -> tuple[Path, Path, Path | None]:
        self._log_file.close()
        loss_path, acc_path = plot_training_curves(self.history, self.run_dir)
        cm_path = None
        if y_true is not None and y_pred is not None:
            cm_path = plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                run_dir=self.run_dir,
                class_names=class_names,
                title=title,
            )
        if self.history["epochs"]:
            saved = f"saved plots: {loss_path}, {acc_path}"
            if cm_path is not None:
                saved += f", {cm_path}"
            print(saved)
        print(f"training log: {self.log_path}")
        print(f"metrics: {self.metrics_path}")
        return loss_path, acc_path, cm_path


def setup_recorder(run_dir: Path, resume: bool = False) -> TrainingRecorder:
    return TrainingRecorder(run_dir, resume=resume)
