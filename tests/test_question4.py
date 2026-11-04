"""Tests for Question 4: enhanced MLP architecture, dropout behaviour and baseline comparison."""
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2.compare import BASELINE_COMPARISON_FIELDS, baseline_comparison
from a2.models import ENHANCED_MLP_SPEC, MODEL_SPECS, EnhancedMLP


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = EnhancedMLP()

    def test_layer_dimensions_and_order(self):
        kinds = [(name, type(m).__name__) for name, m in self.model.named_children()]
        self.assertEqual(kinds, [("l1", "Linear"), ("relu1", "ReLU"), ("drop1", "Dropout"),
                                 ("l2", "Linear"), ("relu2", "ReLU"), ("drop2", "Dropout"),
                                 ("l3", "Linear")])
        dims = [(m.in_features, m.out_features) for m in (self.model.l1, self.model.l2, self.model.l3)]
        self.assertEqual(dims, [(784, 500), (500, 200), (200, 10)])
        self.assertEqual((self.model.drop1.p, self.model.drop2.p), (0.2, 0.2))

    def test_parameter_count(self):
        expected = (784 * 500 + 500) + (500 * 200 + 200) + (200 * 10 + 10)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), expected)
        self.assertEqual(expected, 494_710)

    def test_eval_forward_matches_relu_then_linear_stack(self):
        # In eval mode dropout is the identity, so the output must equal the plain stack.
        self.model.eval()
        x = torch.rand(5, 784)
        m = self.model
        manual = F.log_softmax(m.l3(F.relu(m.l2(F.relu(m.l1(x))))), dim=1)
        torch.testing.assert_close(m(x), manual)
        self.assertEqual(m(x).shape, (5, 10))
        torch.testing.assert_close(m(x).exp().sum(dim=1), torch.ones(5))   # log-probabilities

    def test_dropout_follows_relu(self):
        # Dropout must receive post-ReLU activations (all >= 0).
        seen = {}
        hooks = [getattr(self.model, n).register_forward_pre_hook(
            lambda mod, inp, n=n: seen.__setitem__(n, inp[0].min().item())) for n in ("drop1", "drop2")]
        self.model.train()
        self.model(torch.randn(8, 784))
        for h in hooks:
            h.remove()
        self.assertGreaterEqual(seen["drop1"], 0.0)
        self.assertGreaterEqual(seen["drop2"], 0.0)


class DropoutBehaviourTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = EnhancedMLP()
        self.x = torch.rand(64, 784)

    def test_eval_is_deterministic_train_is_stochastic(self):
        self.model.eval()
        torch.testing.assert_close(self.model(self.x), self.model(self.x))
        self.model.train()
        self.assertFalse(torch.equal(self.model(self.x), self.model(self.x)))

    def test_drop_rate_and_scaling(self):
        captured = {}
        self.model.drop1.register_forward_hook(lambda mod, inp, out: captured.update(inp=inp[0], out=out))
        self.model.train()
        torch.manual_seed(123)
        self.model(torch.rand(400, 784))
        inp, out = captured["inp"], captured["out"]
        active = inp > 0
        dropped = (out[active] == 0).float().mean().item()
        self.assertAlmostEqual(dropped, 0.2, delta=0.01)          # ~20 % of units zeroed
        kept = active & (out != 0)
        torch.testing.assert_close(out[kept], inp[kept] / 0.8)     # inverted-dropout scaling

    def test_dropout_disabled_in_eval(self):
        captured = {}
        self.model.drop2.register_forward_hook(lambda mod, inp, out: captured.update(inp=inp[0], out=out))
        self.model.eval()
        self.model(self.x)
        torch.testing.assert_close(captured["out"], captured["inp"])


class SpecTests(unittest.TestCase):
    def test_training_configuration_matches_original_mlp(self):
        orig, new = MODEL_SPECS["mlp"], ENHANCED_MLP_SPEC
        for field in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input"):
            self.assertEqual(getattr(new, field), getattr(orig, field), field)
        self.assertIs(new.make_loss, nn.NLLLoss)
        self.assertIs(new.build, EnhancedMLP)
        self.assertEqual(new.prepare_input(torch.rand(3, 1, 28, 28)).shape, (3, 784))

    def test_questions_1_to_3_model_set_unchanged(self):
        self.assertEqual(list(MODEL_SPECS), ["mlp", "cnn", "mobilenetv2"])
        self.assertNotIn(ENHANCED_MLP_SPEC.name, MODEL_SPECS)


class BaselineComparisonTests(unittest.TestCase):
    @staticmethod
    def agg(acc, train, inf, sd=0.1):
        return {"accuracy_percent": {"mean": acc, "std_sample": sd},
                "train_seconds": {"mean": train, "std_sample": sd},
                "inference_seconds": {"mean": inf, "std_sample": sd}}

    def test_differences_and_ratios(self):
        rows = {r["metric"]: r for r in baseline_comparison(self.agg(98.0, 12.0, 0.30),
                                                            self.agg(97.0, 10.0, 0.25))}
        self.assertEqual(tuple(rows["accuracy_percent"]), BASELINE_COMPARISON_FIELDS)
        self.assertAlmostEqual(rows["accuracy_percent"]["difference"], 1.0)      # +1 pp
        self.assertAlmostEqual(rows["train_seconds"]["difference"], 2.0)         # 2 s slower
        self.assertAlmostEqual(rows["train_seconds"]["ratio"], 1.2)
        self.assertAlmostEqual(rows["inference_seconds"]["ratio"], 1.2)
        self.assertEqual(rows["accuracy_percent"]["baseline_mean"], 97.0)

    def test_reduction_is_negative(self):
        rows = {r["metric"]: r for r in baseline_comparison(self.agg(96.5, 8.0, 0.2), self.agg(97.0, 10.0, 0.25))}
        self.assertAlmostEqual(rows["accuracy_percent"]["difference"], -0.5)
        self.assertTrue(math.isclose(rows["train_seconds"]["ratio"], 0.8))

    def test_zero_baseline_rejected(self):
        with self.assertRaises(ValueError):
            baseline_comparison(self.agg(1, 1, 1), self.agg(0, 1, 1))


class CliTests(unittest.TestCase):
    def test_protocol_must_match_question1(self):
        import q4_enhanced_mlp as q4
        baseline = {"protocol": {"seeds": [0, 1, 2], "warmup_batches": 20, "num_workers": 0}}
        ok = SimpleNamespace(seeds=[0, 1, 2], warmup_batches=20, num_workers=0, smoke=False)
        q4.check_protocol(baseline, ok)                                  # no exit
        for change in ({"seeds": [5, 6, 7]}, {"warmup_batches": 0}, {"num_workers": 2}):
            with self.subTest(change=change), self.assertRaises(SystemExit):
                q4.check_protocol(baseline, SimpleNamespace(**{**vars(ok), **change}))

    def test_out_dir_guard_and_defaults(self):
        import q4_enhanced_mlp as q4
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            q4.parse_args(["--out", "results/question1/q4"])
        args = q4.parse_args([])
        self.assertEqual((args.seeds, args.warmup_batches, str(args.out)), ([0, 1, 2], 20, "results/question4"))


if __name__ == "__main__":
    unittest.main()
