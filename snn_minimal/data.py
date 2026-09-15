from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_channels: int
    image_size: int
    num_classes: int = 10
    class_names: tuple[str, ...] = field(default_factory=tuple)


DATASET_SPECS = {
    "mnist": DatasetSpec(
        name="mnist",
        in_channels=1,
        image_size=28,
        class_names=tuple(str(i) for i in range(10)),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        in_channels=3,
        image_size=32,
        class_names=(
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
    ),
}

CIFAR10_LOGIC_THRESHOLDS = (0.25, 0.5, 0.75)


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_SPECS[key]


def _binarize_tensor(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return (x >= threshold).to(x.dtype)


def _threshold_encode_cifar(
    x: torch.Tensor, thresholds: tuple[float, float, float]
) -> torch.Tensor:
    """Encode RGB intensities as three thresholded binary channel groups."""
    return torch.cat([(x > threshold).to(x.dtype) for threshold in thresholds], dim=0)


def _validate_cifar_logic_thresholds(
    thresholds: tuple[float, ...] | list[float],
) -> tuple[float, float, float]:
    values = tuple(float(value) for value in thresholds)
    if len(values) != 3:
        raise ValueError("CIFAR Logic-SNN requires exactly three thresholds")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("CIFAR logic thresholds must be in (0, 1)")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("CIFAR logic thresholds must be strictly increasing")
    return values


def build_transforms(
    dataset_name: str,
    mnist_binarize_threshold: float | None = None,
    cifar_logic_thresholds: tuple[float, ...] | list[float] | None = None,
):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        if mnist_binarize_threshold is None:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            threshold = float(mnist_binarize_threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("mnist_binarize_threshold must be in [0, 1]")
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(partial(_binarize_tensor, threshold=threshold)),
                ]
            )
        train_transform = transform
        test_transform = transform
        return train_transform, test_transform

    if dataset_name == "cifar10":
        if cifar_logic_thresholds is not None:
            thresholds = _validate_cifar_logic_thresholds(cifar_logic_thresholds)
            encode = transforms.Lambda(
                partial(_threshold_encode_cifar, thresholds=thresholds)
            )
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    encode,
                ]
            )
            test_transform = transforms.Compose(
                [transforms.ToTensor(), encode]
            )
            return train_transform, test_transform

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_transform, test_transform

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_datasets(
    dataset_name: str,
    data_root: str,
    mnist_binarize_threshold: float | None = None,
    cifar_logic_thresholds: tuple[float, ...] | list[float] | None = None,
):
    train_transform, test_transform = build_transforms(
        dataset_name,
        mnist_binarize_threshold=mnist_binarize_threshold,
        cifar_logic_thresholds=cifar_logic_thresholds,
    )
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        train_set = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_set = datasets.MNIST(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        return train_set, test_set

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_set = datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        return train_set, test_set

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _split_train_val_indices(
    total_size: int, val_ratio: float, split_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError(
            f"val_ratio={val_ratio} is too large for dataset size={total_size}"
        )
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total_size, generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def build_train_val_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_ratio: float = 0.1,
    split_seed: int = 42,
    mnist_binarize_threshold: float | None = None,
    cifar_logic_thresholds: tuple[float, ...] | list[float] | None = None,
):
    dataset_name = dataset_name.lower()
    train_transform, test_transform = build_transforms(
        dataset_name,
        mnist_binarize_threshold=mnist_binarize_threshold,
        cifar_logic_thresholds=cifar_logic_thresholds,
    )

    if dataset_name == "mnist":
        train_set_aug = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        train_set_eval = datasets.MNIST(
            root=data_root,
            train=True,
            transform=test_transform,
            download=True,
        )
    elif dataset_name == "cifar10":
        train_set_aug = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        train_set_eval = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=test_transform,
            download=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_indices, val_indices = _split_train_val_indices(
        total_size=len(train_set_aug), val_ratio=val_ratio, split_seed=split_seed
    )

    train_set = Subset(train_set_aug, train_indices.tolist())
    val_set = Subset(train_set_eval, val_indices.tolist())

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def build_test_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    mnist_binarize_threshold: float | None = None,
    cifar_logic_thresholds: tuple[float, ...] | list[float] | None = None,
):
    _, test_set = build_datasets(
        dataset_name,
        data_root,
        mnist_binarize_threshold=mnist_binarize_threshold,
        cifar_logic_thresholds=cifar_logic_thresholds,
    )
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )


def infer_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
