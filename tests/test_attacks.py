"""Çevrimdışı doğruluk testleri: `python -m unittest discover -s tests -v`."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from mnist_robust.attacks import fgsm, pgd_linf
from mnist_robust.data import make_loaders
from mnist_robust.model import SmallCNN
from mnist_robust.utils import seed_everything


class IntermediateFailureModel(nn.Module):
    """PGD 0 -> 0.6 -> 0 gider; yalnızca 0.6 yanlış sınıflandırılır."""

    def forward(self, x):
        value = x[:, 0]
        other = 0.2 - 10 * (value - 0.5).square()
        return torch.stack((torch.zeros_like(other), other), dim=1)


class LossPriorityTrap(nn.Module):
    """x=1 yanlış, x=0 doğru; fakat cross-entropy x=0'da daha yüksektir."""

    def forward(self, x):
        value = x[:, 0]
        probs = torch.stack((0.4 + 0.05 * value, 0.3 + 0.16 * value, 0.3 - 0.21 * value), dim=1)
        return probs.log()


class ThresholdModel(nn.Module):
    def forward(self, x):
        value = 8 * (x[:, 0] - 0.5)
        return torch.stack((-value, value), dim=1)


class AttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        self.x = torch.rand(7, 4)
        self.y = torch.tensor([0, 1, 2, 0, 1, 2, 0])

    def test_bounds_and_detached_output(self):
        for attack in (
            lambda: fgsm(self.model, self.x, self.y, eps=0.3),
            lambda: pgd_linf(self.model, self.x, self.y, eps=0.3, alpha=0.2, steps=5, restarts=3),
            lambda: pgd_linf(self.model, self.x, self.y, eps=0.3, alpha=0.2, steps=5, track_best=False),
        ):
            with self.subTest(attack=attack):
                original = self.x.clone()
                adversarial = attack()
                self.assertTrue(torch.all(adversarial >= 0))
                self.assertTrue(torch.all(adversarial <= 1))
                self.assertLessEqual((adversarial - self.x).abs().max().item(), 0.3 + 1e-6)
                self.assertFalse(adversarial.requires_grad)
                self.assertIsNone(adversarial.grad_fn)
                self.assertTrue(torch.equal(original, self.x))

    def test_epsilon_zero_exact_identity(self):
        for attack in (fgsm, pgd_linf):
            with self.subTest(attack=attack.__name__):
                output = attack(self.model, self.x, self.y, eps=0)
                self.assertTrue(torch.equal(output, self.x))
                self.assertNotEqual(output.data_ptr(), self.x.data_ptr())

    def test_mode_bn_and_existing_gradients_preserved(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(), nn.Linear(8, 3))
        model.train()
        model[3].eval()
        for parameter in model.parameters():
            parameter.grad = torch.randn_like(parameter)
        expected_grads = [parameter.grad.clone() for parameter in model.parameters()]
        expected_modes = [module.training for module in model.modules()]
        expected_mean = model[1].running_mean.clone()
        expected_var = model[1].running_var.clone()
        x = self.x.clone().requires_grad_(True)
        x.grad = torch.full_like(x, 0.125)
        for attack in (fgsm, pgd_linf):
            kwargs = {"steps": 3, "restarts": 2} if attack is pgd_linf else {}
            with torch.no_grad():
                attack(model, x, self.y, eps=0.2, **kwargs)
            self.assertEqual(expected_modes, [module.training for module in model.modules()])
            self.assertTrue(torch.equal(expected_mean, model[1].running_mean))
            self.assertTrue(torch.equal(expected_var, model[1].running_var))
            self.assertTrue(torch.equal(x.grad, torch.full_like(x, 0.125)))
            for expected, parameter in zip(expected_grads, model.parameters()):
                self.assertTrue(torch.equal(expected, parameter.grad))

    def test_empty_parameter_gradients_stay_empty(self):
        pgd_linf(self.model, self.x, self.y, steps=2)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_intermediate_failure_is_retained(self):
        model = IntermediateFailureModel()
        x, y = torch.tensor([[0.0]]), torch.tensor([0])
        arguments = dict(eps=1.0, alpha=0.6, steps=2, random_start=False)
        final = pgd_linf(model, x, y, track_best=False, **arguments)
        best = pgd_linf(model, x, y, track_best=True, **arguments)
        self.assertEqual(model(final).argmax(1).item(), 0)
        self.assertEqual(model(best).argmax(1).item(), 1)
        self.assertAlmostEqual(best.item(), 0.6, places=6)

    def test_misclassification_precedes_higher_loss(self):
        model = LossPriorityTrap()
        x, y = torch.tensor([[1.0]]), torch.tensor([0])
        final = pgd_linf(model, x, y, eps=1, alpha=1, steps=1, random_start=False, track_best=False)
        best = pgd_linf(model, x, y, eps=1, alpha=1, steps=1, random_start=False)
        self.assertGreater(F.cross_entropy(model(final), y), F.cross_entropy(model(best), y))
        self.assertEqual(model(final).argmax(1).item(), 0)
        self.assertEqual(model(best).argmax(1).item(), 1)
        self.assertTrue(torch.equal(best, x))

    def test_restarts_are_selected_per_example(self):
        model = ThresholdModel()
        x = torch.tensor([[0.1], [0.9], [0.2], [0.8]])
        y = torch.tensor([0, 1, 0, 1])
        torch.manual_seed(129)
        candidates = [x]
        for _ in range(12):
            candidates.append((x + torch.empty_like(x).uniform_(-0.6, 0.6)).clamp(0, 1))
        losses = torch.stack([F.cross_entropy(model(candidate), y, reduction="none") for candidate in candidates])
        winners = losses.argmax(dim=0)
        expected = torch.stack(candidates)[winners, torch.arange(len(x))]
        self.assertGreater(len(set(winners.tolist())), 1)
        torch.manual_seed(129)
        actual = pgd_linf(model, x, y, eps=0.6, steps=0, restarts=12)
        self.assertTrue(torch.equal(actual, expected))

    def test_more_restarts_never_recover_discovered_failures(self):
        torch.manual_seed(48)
        one = pgd_linf(self.model, self.x, self.y, steps=3, restarts=1)
        torch.manual_seed(48)
        many = pgd_linf(self.model, self.x, self.y, steps=3, restarts=4)
        wrong_one = self.model(one).argmax(1).ne(self.y)
        wrong_many = self.model(many).argmax(1).ne(self.y)
        self.assertTrue(torch.all(~wrong_one | wrong_many))

    def test_invalid_parameters_fail_early(self):
        for kwargs in (
            {"eps": -0.1}, {"eps": float("nan")}, {"alpha": -0.01},
            {"steps": -1}, {"steps": 1.5}, {"restarts": 0},
            {"restarts": 2, "track_best": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pgd_linf(self.model, self.x, self.y, **kwargs)
        with self.assertRaises(ValueError):
            fgsm(self.model, self.x + 1, self.y, eps=0.3)

    def test_model_mode_restored_after_model_error(self):
        class BrokenModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("deliberate test failure")

        model = BrokenModel().train()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            pgd_linf(model, self.x, self.y)
        self.assertTrue(model.training)


class DataAndModelTests(unittest.TestCase):
    def test_model_output_shape_and_real_gradient_step(self):
        model = SmallCNN(width=4, hidden=16)
        x, y = torch.rand(3, 1, 28, 28), torch.tensor([0, 4, 9])
        adversarial = pgd_linf(model, x, y, steps=1, track_best=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.features[0].weight.detach().clone()
        logits = model(adversarial)
        self.assertEqual(tuple(logits.shape), (3, 10))
        F.cross_entropy(logits, y).backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.features[0].weight))

    def test_synthetic_data_fixed_across_training_seeds(self):
        kwargs = dict(data_dir="unused", batch_size=4, split_seed=2026, synthetic=True, train_limit=16, val_limit=8, test_limit=8)
        first = make_loaders(seed=0, **kwargs)
        second = make_loaders(seed=1, **kwargs)
        for name in first:
            for a, b in zip(first[name].dataset.tensors, second[name].dataset.tensors):
                self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(next(iter(first["train"]))[0], next(iter(second["train"]))[0]))
        self.assertTrue(torch.equal(next(iter(first["val"]))[0], next(iter(second["val"]))[0]))
        self.assertFalse(torch.equal(first["val"].dataset.tensors[0], first["test"].dataset.tensors[0]))

    def test_seed_repeats_initial_weights_and_attack(self):
        seed_everything(5)
        model_a = SmallCNN(width=2, hidden=8)
        x, y = torch.full((2, 1, 28, 28), 0.5), torch.tensor([0, 1])
        first = pgd_linf(model_a, x, y, steps=1)
        seed_everything(5)
        model_b = SmallCNN(width=2, hidden=8)
        second = pgd_linf(model_b, x, y, steps=1)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
