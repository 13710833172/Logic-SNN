import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from snn_minimal.engine import run_epoch
from train import build_scheduler, step_scheduler


class GradientAccumulationTests(unittest.TestCase):
    def test_accumulated_microbatches_match_one_large_batch(self):
        images = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.25]]
        )
        targets = torch.tensor([0, 1, 0, 1])

        large_batch_model = nn.Linear(2, 2, bias=False)
        microbatch_model = nn.Linear(2, 2, bias=False)
        microbatch_model.load_state_dict(large_batch_model.state_dict())

        large_optimizer = torch.optim.SGD(large_batch_model.parameters(), lr=0.1)
        micro_optimizer = torch.optim.SGD(microbatch_model.parameters(), lr=0.1)

        run_epoch(
            large_batch_model,
            DataLoader(TensorDataset(images, targets), batch_size=4),
            torch.device("cpu"),
            num_classes=2,
            optimizer=large_optimizer,
            loss_name="cross_entropy",
        )
        run_epoch(
            microbatch_model,
            DataLoader(TensorDataset(images, targets), batch_size=2),
            torch.device("cpu"),
            num_classes=2,
            optimizer=micro_optimizer,
            loss_name="cross_entropy",
            grad_accum_steps=2,
        )

        self.assertTrue(
            torch.allclose(
                large_batch_model.weight,
                microbatch_model.weight,
                atol=1e-7,
                rtol=1e-6,
            )
        )


class SchedulerTests(unittest.TestCase):
    def test_plateau_scheduler_respects_nonzero_minimum_lr(self):
        parameter = nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=0.02)
        args = SimpleNamespace(
            lr_scheduler="plateau",
            scheduler_factor=0.5,
            scheduler_patience=0,
            min_lr=0.005,
            epochs=10,
        )
        scheduler = build_scheduler(args, optimizer)

        step_scheduler(scheduler, args.lr_scheduler, 0.5)
        step_scheduler(scheduler, args.lr_scheduler, 0.4)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        step_scheduler(scheduler, args.lr_scheduler, 0.3)
        step_scheduler(scheduler, args.lr_scheduler, 0.2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.005)

    def test_cosine_scheduler_uses_configured_minimum_lr(self):
        parameter = nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.SGD([parameter], lr=0.02)
        args = SimpleNamespace(
            lr_scheduler="cosine",
            scheduler_factor=0.5,
            scheduler_patience=0,
            min_lr=0.001,
            epochs=2,
        )
        scheduler = build_scheduler(args, optimizer)

        step_scheduler(scheduler, args.lr_scheduler, 0.0)
        step_scheduler(scheduler, args.lr_scheduler, 0.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001)


if __name__ == "__main__":
    unittest.main()
