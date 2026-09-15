from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from spikingjelly.activation_based import functional

from snn_minimal.data import build_transforms
from snn_minimal.engine import classification_loss
from snn_minimal.models import (
    ConvSNN,
    HardLogicSNN,
    LogicSNN,
    build_model_from_args,
    load_hard_checkpoint,
    save_hard_checkpoint,
)
from snn_minimal.models.logic_layers import ConvLogicTreeLayer, DiffLogicLayer
from snn_minimal.models.logic_ops import (
    build_truth_table,
    compute_all_16_ops,
    weighted_sum_16_ops,
)


class LogicOperatorTests(unittest.TestCase):
    def test_all_hard_truth_tables(self):
        expected = torch.tensor(
            [[(operator_id >> shift) & 1 for shift in (3, 2, 1, 0)] for operator_id in range(16)]
        )
        actual = build_truth_table().reshape(16, 4)
        torch.testing.assert_close(actual, expected)

    def test_soft_operators_stay_in_unit_interval(self):
        a = torch.rand(128)
        b = torch.rand(128)
        outputs = compute_all_16_ops(a, b)
        self.assertGreaterEqual(float(outputs.min()), 0.0)
        self.assertLessEqual(float(outputs.max()), 1.0)

    def test_bilinear_mixture_matches_explicit_ops_and_gradients(self):
        a = torch.rand(4, 7, requires_grad=True)
        b = torch.rand(4, 7, requires_grad=True)
        logits = torch.randn(1, 7, 16, requires_grad=True)
        probs = logits.softmax(dim=-1)
        actual = weighted_sum_16_ops(a, b, probs)
        expected = (compute_all_16_ops(a, b) * probs).sum(dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        actual.sum().backward(retain_graph=True)
        actual_grads = tuple(value.grad.detach().clone() for value in (a, b, logits))
        for value in (a, b, logits):
            value.grad = None
        expected.sum().backward()
        for actual_grad, value in zip(actual_grads, (a, b, logits)):
            torch.testing.assert_close(actual_grad, value.grad, rtol=1e-5, atol=1e-6)


class ReadoutLossTests(unittest.TestCase):
    def test_group_sum_temperature_cross_entropy(self):
        outputs = torch.tensor([[0.2, 0.4], [0.7, 0.1]])
        targets = torch.tensor([1, 0])
        loss, scores = classification_loss(
            outputs,
            targets,
            num_classes=2,
            loss_name="cross_entropy",
            logit_scale=64 / 5,
        )
        expected_scores = outputs * (64 / 5)
        torch.testing.assert_close(scores, expected_scores)
        torch.testing.assert_close(loss, F.cross_entropy(expected_scores, targets))

    def test_mse_readout_is_unchanged(self):
        outputs = torch.tensor([[0.2, 0.4]])
        targets = torch.tensor([1])
        loss, scores = classification_loss(outputs, targets, num_classes=2)
        torch.testing.assert_close(scores, outputs)
        torch.testing.assert_close(
            loss,
            F.mse_loss(outputs, F.one_hot(targets, num_classes=2).float()),
        )


class LogicLayerTests(unittest.TestCase):
    def test_layer_shapes_and_connectivity(self):
        conv = ConvLogicTreeLayer(1, 4, 3, connectivity_seed=7, padding=1)
        dense = DiffLogicLayer(4 * 14 * 14, 12, connectivity_seed=8)
        x = torch.rand(2, 1, 28, 28)
        conv_output = conv(x)
        dense_output = dense(F.max_pool2d(conv_output, 2).flatten(1))

        self.assertEqual(conv_output.shape, (2, 4, 28, 28))
        self.assertEqual(dense_output.shape, (2, 12))
        self.assertGreaterEqual(float(conv_output.min()), 0.0)
        self.assertLessEqual(float(conv_output.max()), 1.0)
        self.assertGreaterEqual(float(dense_output.min()), 0.0)
        self.assertLessEqual(float(dense_output.max()), 1.0)
        self.assertEqual(conv.gate_logits.shape, (4, 7, 16))
        self.assertEqual(conv.leaf_indices.shape, (4, 8))
        self.assertTrue(torch.all(conv.leaf_indices < 9))
        for row in conv.leaf_indices:
            self.assertEqual(torch.unique(row).numel(), 8)
        self.assertEqual(dense.connectivity.shape, (12, 2))
        self.assertTrue(torch.all(dense.connectivity < 4 * 14 * 14))

    def test_connectivity_is_seed_deterministic(self):
        a = ConvLogicTreeLayer(4, 8, 3, connectivity_seed=19)
        b = ConvLogicTreeLayer(4, 8, 3, connectivity_seed=19)
        c = ConvLogicTreeLayer(4, 8, 3, connectivity_seed=20)
        self.assertTrue(torch.equal(a.leaf_indices, b.leaf_indices))
        self.assertFalse(torch.equal(a.leaf_indices, c.leaf_indices))

    def test_straight_through_is_hard_forward_with_soft_gradients(self):
        layer = DiffLogicLayer(
            16,
            12,
            connectivity_seed=4,
            gate_init="normal",
            gate_mode="straight_through",
            gate_temperature=0.5,
        )
        x = torch.randint(0, 2, (5, 16)).float()
        output = layer(x)
        self.assertTrue(torch.all((output == 0) | (output == 1)))
        output.mean().backward()
        self.assertIsNotNone(layer.gate_logits.grad)
        self.assertTrue(torch.isfinite(layer.gate_logits.grad).all())


class LogicSNNTests(unittest.TestCase):
    def _small_model(self) -> LogicSNN:
        return LogicSNN(
            timesteps=4,
            channels=4,
            hidden_dim=16,
            outputs_per_class=2,
            current_gain=2.2,
            connectivity_seed=3,
        )

    def test_mnist_binarization(self):
        train_transform, test_transform = build_transforms(
            "mnist", mnist_binarize_threshold=0.5
        )
        image = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8))
        expected = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        torch.testing.assert_close(train_transform(image), expected)
        torch.testing.assert_close(test_transform(image), expected)

    def test_cifar_binarization(self):
        _, test_transform = build_transforms(
            "cifar10", cifar_logic_thresholds=(0.25, 0.5, 0.75)
        )
        pixels = np.array([[[0, 64, 192], [255, 128, 63]]], dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        output = test_transform(image)
        expected = torch.tensor(
            [
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 0.0]],
                [[1.0, 0.0]],
            ]
        )
        torch.testing.assert_close(output, expected)
        self.assertEqual(output.shape, (9, 1, 2))
        self.assertTrue(torch.all((output == 0) | (output == 1)))

    def test_full_structure_and_gradients(self):
        model = self._small_model()
        x = torch.randint(0, 2, (2, 1, 28, 28)).float()
        output = model(x)
        loss = F.mse_loss(
            output,
            F.one_hot(torch.tensor([1, 2]), num_classes=10).float(),
        )
        loss.backward()

        self.assertEqual(output.shape, (2, 10))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)
        expected_shapes = [(4, 7, 16), (8, 7, 16), (16, 16), (20, 16)]
        actual_shapes = [tuple(layer.gate_logits.shape) for layer in model.logic_layers()]
        self.assertEqual(actual_shapes, expected_shapes)
        for layer in model.logic_layers():
            self.assertIsNotNone(layer.gate_logits.grad)
            self.assertTrue(torch.isfinite(layer.gate_logits.grad).all())
        functional.reset_net(model)

    def test_default_stage_shapes(self):
        model = LogicSNN(timesteps=1)
        x = torch.randint(0, 2, (1, 1, 28, 28)).float()
        z1 = model.pool(model.spike1(model.current_gain * model.conv1(x)))
        z2 = model.pool(model.spike2(model.current_gain * model.conv2(z1)))
        z3 = model.spike3(model.current_gain * model.fc1(z2.flatten(1)))
        z4 = model.spike4(model.current_gain * model.fc2(z3))
        self.assertEqual(z1.shape, (1, 64, 14, 14))
        self.assertEqual(z2.shape, (1, 128, 7, 7))
        self.assertEqual(z3.shape, (1, 256))
        self.assertEqual(z4.shape, (1, 640))
        functional.reset_net(model)

    def test_mnist_second_hidden_layer_and_hard_round_trip(self):
        model = LogicSNN(
            timesteps=1,
            channels=2,
            hidden_dim=12,
            hidden_dim2=8,
            outputs_per_class=2,
            connectivity_seed=13,
        )
        self.assertEqual(len(model.dense_layers()), 3)
        self.assertEqual(
            [layer.num_neurons for layer in model.dense_layers()],
            [12, 8, 20],
        )

        x = torch.randint(0, 2, (2, 1, 28, 28)).float()
        output = model(x)
        self.assertEqual(output.shape, (2, 10))
        output.mean().backward()
        for layer in model.logic_layers():
            self.assertIsNotNone(layer.gate_logits.grad)
        functional.reset_net(model)

        hard = model.to_hard()
        self.assertEqual(hard.hidden_dim2, 8)
        self.assertEqual(len(hard.dense_layers()), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mnist_deep_hard.pt"
            save_hard_checkpoint(path, hard)
            loaded, _ = load_hard_checkpoint(path)
            with torch.no_grad():
                expected = hard(x)
                functional.reset_net(hard)
                actual = loaded(x)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(loaded.hidden_dim2, 8)
            functional.reset_net(loaded)
        functional.reset_net(hard)

    def test_mnist_checkpoint_growth_preserves_forward_path(self):
        shallow = LogicSNN(
            timesteps=2,
            channels=2,
            hidden_dim=12,
            outputs_per_class=2,
            connectivity_seed=17,
            gate_init="normal",
        )
        deep = LogicSNN(
            timesteps=2,
            channels=2,
            hidden_dim=12,
            hidden_dim2=12,
            outputs_per_class=2,
            connectivity_seed=17,
            gate_init="normal",
        )
        deep.initialize_from_shallow_mnist(shallow.state_dict())

        x = torch.randint(0, 2, (3, 1, 28, 28)).float()
        with torch.no_grad():
            expected = shallow(x)
            functional.reset_net(shallow)
            actual = deep(x)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(
            torch.equal(deep.fc2.connectivity[:, 0], torch.arange(12))
        )
        functional.reset_net(deep)

    def test_mnist_identity_middle_initialization(self):
        model = LogicSNN(
            timesteps=1,
            channels=2,
            hidden_dim=12,
            hidden_dim2=12,
            outputs_per_class=2,
            identity_middle_init=True,
        )
        self.assertTrue(
            torch.equal(model.fc2.connectivity[:, 0], torch.arange(12))
        )
        self.assertTrue(torch.all(model.fc2.gate_logits[..., 3] == 5.0))
        other_logits = model.fc2.gate_logits.clone()
        other_logits[..., 3] = 0.0
        self.assertTrue(torch.all(other_logits == 0.0))

    def test_cifar_structure_and_gradients(self):
        model = LogicSNN(
            dataset="cifar10",
            timesteps=2,
            channels=4,
            hidden_dim=16,
            outputs_per_class=2,
            current_gain=2.2,
            connectivity_seed=5,
        )
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        output = model(x)
        loss = F.mse_loss(
            output,
            F.one_hot(torch.tensor([1, 2]), num_classes=10).float(),
        )
        loss.backward()

        self.assertEqual(output.shape, (2, 10))
        expected_shapes = [
            (4, 7, 16),
            (4, 7, 16),
            (8, 7, 16),
            (8, 7, 16),
            (16, 16),
            (20, 16),
        ]
        self.assertEqual(
            [tuple(layer.gate_logits.shape) for layer in model.logic_layers()],
            expected_shapes,
        )
        for layer in model.logic_layers():
            self.assertIsNotNone(layer.gate_logits.grad)
            self.assertTrue(torch.isfinite(layer.gate_logits.grad).all())
        functional.reset_net(model)

    def test_cifar_stage_shapes(self):
        model = LogicSNN(
            dataset="cifar10",
            timesteps=1,
            channels=4,
            hidden_dim=16,
            outputs_per_class=2,
        )
        x = torch.randint(0, 2, (1, 9, 32, 32)).float()
        z1 = model.spike1(model.current_gain * model.conv1(x))
        z2 = model.pool(model.spike2(model.current_gain * model.conv2(z1)))
        z3 = model.spike3(model.current_gain * model.conv3(z2))
        z4 = model.pool(model.spike4(model.current_gain * model.conv4(z3)))
        z5 = model.spike5(model.current_gain * model.fc1(z4.flatten(1)))
        z6 = model.spike6(model.current_gain * model.fc2(z5))
        self.assertEqual(z1.shape, (1, 4, 32, 32))
        self.assertEqual(z2.shape, (1, 4, 16, 16))
        self.assertEqual(z3.shape, (1, 8, 16, 16))
        self.assertEqual(z4.shape, (1, 8, 8, 8))
        self.assertEqual(z5.shape, (1, 16))
        self.assertEqual(z6.shape, (1, 20))
        functional.reset_net(model)

    def test_cifar_pyramid_structure_gradients_and_hard_round_trip(self):
        model = LogicSNN(
            dataset="cifar10",
            cifar_architecture="pyramid",
            timesteps=2,
            channels=2,
            hidden_dim=16,
            hidden_dim2=8,
            outputs_per_class=2,
            connectivity_seed=11,
        )
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        z = x
        expected_conv_shapes = [
            (2, 2, 16, 16),
            (2, 8, 8, 8),
            (2, 32, 4, 4),
            (2, 64, 2, 2),
        ]
        for index, logic_layer in enumerate(model.conv_layers(), start=1):
            spike = getattr(model, f"spike{index}")
            z = model.pool(spike(model.current_gain * logic_layer(z)))
            self.assertEqual(z.shape, expected_conv_shapes[index - 1])
        z = z.flatten(1)
        for index, expected_width in enumerate((16, 8, 20), start=1):
            spike = getattr(model, f"spike{4 + index}")
            z = spike(model.current_gain * getattr(model, f"fc{index}")(z))
            self.assertEqual(z.shape, (2, expected_width))
        functional.reset_net(model)

        output = model(x)
        self.assertEqual(output.shape, (2, 10))
        output.mean().backward()
        self.assertEqual(len(model.logic_layers()), 7)
        for layer in model.logic_layers():
            self.assertIsNotNone(layer.gate_logits.grad)
            self.assertTrue(torch.isfinite(layer.gate_logits.grad).all())
        functional.reset_net(model)

        hard = model.to_hard()
        self.assertEqual(hard.cifar_architecture, "pyramid")
        self.assertEqual(hard.hidden_dim2, 8)
        self.assertEqual(len(hard.dense_layers()), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pyramid_hard.pt"
            save_hard_checkpoint(path, hard)
            loaded, _ = load_hard_checkpoint(path)
            with torch.no_grad():
                expected = hard(x)
                functional.reset_net(hard)
                actual = loaded(x)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(loaded.cifar_architecture, "pyramid")
            self.assertEqual(len(loaded.logic_layers()), 7)
            functional.reset_net(loaded)
        functional.reset_net(hard)

    def test_cifar_soft_hard_and_checkpoint_round_trip(self):
        soft = LogicSNN(
            dataset="cifar10",
            timesteps=2,
            channels=2,
            hidden_dim=8,
            outputs_per_class=2,
            connectivity_seed=9,
        )
        with torch.no_grad():
            for layer in soft.logic_layers():
                layer.gate_logits.fill_(-80.0)
                layer.gate_logits[..., 3] = 80.0
        hard = soft.to_hard()
        x = torch.randint(0, 2, (2, 9, 32, 32)).float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cifar_hard.pt"
            save_hard_checkpoint(path, hard)
            loaded, _ = load_hard_checkpoint(path)
            with torch.no_grad():
                soft_output = soft(x)
                hard_output = hard(x)
                loaded_output = loaded(x)
            torch.testing.assert_close(soft_output, hard_output, rtol=0.0, atol=0.0)
            torch.testing.assert_close(hard_output, loaded_output, rtol=0.0, atol=0.0)
            self.assertEqual(loaded.dataset, "cifar10")
            self.assertEqual(sum(p.numel() for p in loaded.parameters()), 0)
        functional.reset_net(soft)
        functional.reset_net(hard)
        functional.reset_net(loaded)

    def test_one_hot_soft_gates_match_hard_network(self):
        soft = self._small_model()
        with torch.no_grad():
            for layer in soft.logic_layers():
                layer.gate_logits.fill_(-80.0)
                layer.gate_logits[..., 3] = 80.0
        hard = soft.to_hard()
        x = torch.randint(0, 2, (3, 1, 28, 28)).float()
        with torch.no_grad():
            soft_output = soft(x)
            hard_output = hard(x)
        torch.testing.assert_close(soft_output, hard_output, rtol=0.0, atol=0.0)
        functional.reset_net(soft)
        functional.reset_net(hard)

    def test_straight_through_network_matches_hard_forward(self):
        soft = self._small_model()
        soft.set_gate_mode("straight_through", temperature=0.5)
        hard = soft.to_hard()
        x = torch.randint(0, 2, (2, 1, 28, 28)).float()
        soft_output = soft(x)
        functional.reset_net(soft)
        with torch.no_grad():
            hard_output = hard(x)
        torch.testing.assert_close(soft_output, hard_output, rtol=0.0, atol=0.0)
        soft_output.mean().backward()
        for layer in soft.logic_layers():
            self.assertIsNotNone(layer.gate_logits.grad)
            self.assertTrue(torch.isfinite(layer.gate_logits.grad).all())
        functional.reset_net(soft)
        functional.reset_net(hard)

    def test_hard_checkpoint_round_trip(self):
        soft = self._small_model()
        hard = soft.to_hard()
        self.assertEqual(sum(parameter.numel() for parameter in hard.parameters()), 0)
        x = torch.randint(0, 2, (2, 1, 28, 28)).float()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hard.pt"
            save_hard_checkpoint(path, hard, source={"epoch": 4})
            loaded, bundle = load_hard_checkpoint(path)
            with torch.no_grad():
                expected = hard(x)
                actual = loaded(x)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            self.assertEqual(bundle["source"]["epoch"], 4)
            self.assertIsInstance(loaded, HardLogicSNN)
            functional.reset_net(hard)
            functional.reset_net(loaded)

    def test_hard_logic_layers_are_exactly_binary(self):
        hard = self._small_model().to_hard()
        x = torch.randint(0, 2, (2, 1, 28, 28)).float()
        conv_output = hard.conv1(x)
        pooled = hard.pool(hard.spike1(hard.current_gain * conv_output))
        dense_input = hard.pool(
            hard.spike2(hard.current_gain * hard.conv2(pooled))
        ).flatten(1)
        dense_output = hard.fc1(dense_input)
        for output in (conv_output, dense_output):
            self.assertTrue(torch.all((output == 0) | (output == 1)))
        functional.reset_net(hard)

    def test_factory_keeps_legacy_and_cifar_conv_snn(self):
        legacy = build_model_from_args({"dataset": "mnist"}, legacy_default=True)
        cifar = build_model_from_args({"dataset": "cifar10"})
        current = build_model_from_args({"dataset": "mnist", "model_family": "logic_snn"})
        cifar_logic = build_model_from_args(
            {"dataset": "cifar10", "model_family": "logic_snn"}
        )
        normal_init = build_model_from_args(
            {
                "dataset": "mnist",
                "model_family": "logic_snn",
                "logic_gate_init": "normal",
                "logic_channels_per_tree": 4,
            }
        )
        self.assertIsInstance(legacy, ConvSNN)
        self.assertIsInstance(cifar, ConvSNN)
        self.assertIsInstance(current, LogicSNN)
        self.assertIsInstance(cifar_logic, LogicSNN)
        self.assertEqual(cifar_logic.dataset, "cifar10")
        self.assertEqual(cifar_logic.input_shape, (9, 32, 32))
        self.assertEqual(normal_init.gate_init, "normal")
        self.assertEqual(normal_init.channels_per_tree, 4)
        self.assertEqual(normal_init.to_hard().channels_per_tree, 4)
        self.assertFalse(torch.all(normal_init.conv1.gate_logits[..., 3] == 5.0))


if __name__ == "__main__":
    unittest.main()
