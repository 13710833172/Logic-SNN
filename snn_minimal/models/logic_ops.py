"""Differentiable and hard implementations of all binary Boolean operators."""

from __future__ import annotations

import torch
from torch import Tensor


NUM_OPERATORS = 16
OPERATOR_NAMES = (
    "FALSE",
    "AND",
    "A AND NOT B",
    "A",
    "NOT A AND B",
    "B",
    "XOR",
    "OR",
    "NOR",
    "XNOR",
    "NOT B",
    "B=>A",
    "NOT A",
    "A=>B",
    "NAND",
    "TRUE",
)


def compute_all_16_ops(a: Tensor, b: Tensor) -> Tensor:
    """Return the 16 fuzzy Boolean operators for inputs in ``[0, 1]``."""
    ab = a * b
    a_plus_b = a + b
    xor = a_plus_b - 2.0 * ab
    or_ = a_plus_b - ab
    return torch.stack(
        (
            torch.zeros_like(a),
            ab,
            a - ab,
            a,
            b - ab,
            b,
            xor,
            or_,
            1.0 - or_,
            1.0 - xor,
            1.0 - b,
            1.0 - b + ab,
            1.0 - a,
            1.0 - a + ab,
            1.0 - ab,
            torch.ones_like(a),
        ),
        dim=-1,
    )


def weighted_sum_16_ops(a: Tensor, b: Tensor, probs: Tensor) -> Tensor:
    """Evaluate the exact convex gate mixture as one bilinear polynomial.

    Any fuzzy extension used here is multilinear in ``a`` and ``b``.  Mixing
    the four truth-table values first avoids materializing fifteen full-size
    activation terms, which is important for large convolutional logic nets.
    """
    p = probs
    f00 = p[..., (8, 9, 10, 11, 12, 13, 14, 15)].sum(dim=-1)
    f01 = p[..., (4, 5, 6, 7, 12, 13, 14, 15)].sum(dim=-1)
    f10 = p[..., (2, 3, 6, 7, 10, 11, 14, 15)].sum(dim=-1)
    f11 = p[..., (1, 3, 5, 7, 9, 11, 13, 15)].sum(dim=-1)
    value = (
        f00
        + (f10 - f00) * a
        + (f01 - f00) * b
        + (f11 - f10 - f01 + f00) * (a * b)
    )
    return value.clamp(0.0, 1.0)


def build_truth_table(device: torch.device | None = None) -> Tensor:
    """Build a ``[16, 2, 2]`` integer truth table."""
    table = torch.empty(16, 2, 2, dtype=torch.long, device=device)
    for a in range(2):
        for b in range(2):
            values = compute_all_16_ops(
                torch.tensor(float(a), device=device),
                torch.tensor(float(b), device=device),
            )
            table[:, a, b] = values.long()
    return table


def hard_logic_lookup(
    a: Tensor,
    b: Tensor,
    gate_ids: Tensor,
    truth_table: Tensor,
) -> Tensor:
    """Evaluate selected hard operators by truth-table lookup."""
    ai = (a > 0.5).long()
    bi = (b > 0.5).long()
    return truth_table[gate_ids.long().expand_as(ai), ai, bi]
