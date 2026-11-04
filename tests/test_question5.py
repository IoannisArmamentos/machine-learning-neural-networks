"""Tests for Question 5: dropout MLP, grid, validation split, selection and leakage guards."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from a2.compare import load_summary
from a2.data import IndexedSubset, make_split_loaders, train_validation_indices
from a2.figures import BLUE_RAMP, _hex, heatmap, ramp_color, text_color_for
from a2.models import MODEL_SPECS, DropoutMLP, dropout_mlp_spec
from a2.search import DROPOUTS, HIDDEN_SIZES, grid, rank, run_search, select_best


class DropoutMLPTests(unittest.TestCase):
    def test_single_hidden_layer_with_dropout_after_relu(self):
        m = DropoutMLP(600, 0.3)
        kinds = [(n, type(x).__name__) for n, x in m.named_children()]
        self.assertEqual(kinds, [("l1", "Linear"), ("relu", "ReLU"), ("dropout", "Dropout"), ("l2", "Linear")])
        self.assertEqual((m.l1.in_features, m.l1.out_features, m.l2.in_features, m.l2.out_features),
                         (784, 600, 600, 10))
        self.assertEqual(m.dropout.p, 0.3)
        seen = {}
        m.dropout.register_forward_pre_hook(lambda mod, inp: seen.update(min=inp[0].min().item()))
        m.train()
        m(torch.randn(16, 784))
        self.assertGreaterEqual(seen["min"], 0.0)          # dropout receives post-ReLU activations

    def test_eval_equals_original_mlp_computation(self):
        torch.manual_seed(0)
        m = DropoutMLP(200, 0.5).eval()
        x = torch.rand(4, 784)
        torch.testing.assert_close(m(x), F.log_softmax(m.l2(F.relu(m.l1(x))), dim=1))
        m.train()
        self.assertFalse(torch.equal(m(x), m(x)))          # dropout active in training mode

    def test_parameter_counts(self):
        for h in HIDDEN_SIZES:
            self.assertEqual(sum(p.numel() for p in DropoutMLP(h, 0.2).parameters()), 784 * h + h + h * 10 + 10)

    def test_spec_keeps_original_training_configuration(self):
        orig, spec = MODEL_SPECS["mlp"], dropout_mlp_spec(400, 0.2)
        for field in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input"):
            self.assertEqual(getattr(spec, field), getattr(orig, field), field)
        self.assertIs(spec.make_loss, nn.NLLLoss)
        model = spec.build()
        self.assertIsInstance(model, DropoutMLP)
        self.assertEqual((model.l1.out_features, model.dropout.p), (400, 0.2))
        self.assertEqual(spec.name, "mlp_dropout_h400_p0.2")
        self.assertEqual(list(MODEL_SPECS), ["mlp", "cnn", "mobilenetv2"])


class GridTests(unittest.TestCase):
    def test_all_25_combinations(self):
        g = grid()
        self.assertEqual(len(g), 25)
        self.assertEqual(len(set(g)), 25)
        self.assertEqual({h for h, _ in g}, {200, 400, 600, 800, 1000})
        self.assertEqual({p for _, p in g}, {0.1, 0.2, 0.3, 0.4, 0.5})
        self.assertEqual(g[:2], [(200, 0.1), (200, 0.2)])


class SplitTests(unittest.TestCase):
    def test_sizes_disjoint_and_complete(self):
        tr, va = train_validation_indices(60_000, 6_000, seed=0)
        self.assertEqual((len(tr), len(va)), (54_000, 6_000))
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(set(tr) | set(va), set(range(60_000)))

    def test_reproducible_and_seed_dependent(self):
        self.assertEqual(train_validation_indices(1000, 100, 0), train_validation_indices(1000, 100, 0))
        self.assertNotEqual(train_validation_indices(1000, 100, 0)[1], train_validation_indices(1000, 100, 1)[1])
        with self.assertRaises(ValueError):
            train_validation_indices(10, 10, 0)

    def test_loaders_only_see_their_own_subset(self):
        n = 50
        ds = TensorDataset(torch.arange(n).float().unsqueeze(1), torch.zeros(n, dtype=torch.long))
        tr, va = train_validation_indices(n, 10, seed=3)
        train_loader, val_loader = make_split_loaders(ds, tr, va, batch_size=7, seed=0)
        seen_train = {int(v) for x, _ in train_loader for v in x.flatten()}
        self.assertEqual(seen_train, set(tr))
        seen_val = [(int(x.flatten()[k]), int(i)) for x, _, idx in val_loader for k, i in enumerate(idx)]
        self.assertEqual(sorted(i for _, i in seen_val), sorted(va))
        self.assertTrue(all(value == index for value, index in seen_val))   # original index preserved
        with self.assertRaises(ValueError):
            make_split_loaders(ds, [0, 1, 2], [2, 3], batch_size=2, seed=0)

    def test_indexed_subset(self):
        ds = TensorDataset(torch.arange(5).float(), torch.arange(5))
        sub = IndexedSubset(ds, [4, 1])
        self.assertEqual(len(sub), 2)
        self.assertEqual([int(sub[k][2]) for k in range(2)], [4, 1])


def _row(h, p, correct, **extra):
    return {"hidden_size": h, "dropout": p, "val_n_correct": correct,
            "val_accuracy_percent": correct / 60, **extra}


class SelectionTests(unittest.TestCase):
    def test_highest_validation_accuracy_wins(self):
        rows = [_row(h, p, 5800) for h, p in grid()]
        rows[13]["val_n_correct"] = 5900
        best = select_best(rows, expected_configs=grid())
        self.assertEqual((best["hidden_size"], best["dropout"]), grid()[13])

    def test_tie_break_fewer_neurons_then_lower_dropout(self):
        rows = [_row(800, 0.2, 5900), _row(400, 0.5, 5900), _row(400, 0.3, 5900), _row(200, 0.1, 5850)]
        best = select_best(rows)
        self.assertEqual((best["hidden_size"], best["dropout"]), (400, 0.3))
        self.assertEqual([(r["rank"], r["hidden_size"], r["dropout"]) for r in rank(rows)],
                         [(1, 400, 0.3), (2, 400, 0.5), (3, 800, 0.2), (4, 200, 0.1)])
        self.assertEqual(select_best(list(reversed(rows))), best)       # order-independent

    def test_test_fields_are_rejected(self):
        rows = [_row(200, 0.1, 5900), _row(400, 0.1, 5800, test_accuracy_percent=99.0)]
        with self.assertRaises(ValueError):
            select_best(rows)
        with self.assertRaises(ValueError):
            rank(rows)

    def test_incomplete_grid_is_rejected(self):
        rows = [_row(h, p, 5800) for h, p in grid()[:-1]]
        with self.assertRaises(ValueError):
            select_best(rows, expected_configs=grid())
        with self.assertRaises(ValueError):
            select_best([])


class SearchTests(unittest.TestCase):
    def test_search_api_has_no_test_set(self):
        params = inspect.signature(run_search).parameters
        self.assertFalse([p for p in params if "test" in p.lower()])

    def test_end_to_end_mini_search(self):
        torch.manual_seed(0)
        ds = TensorDataset(torch.rand(150, 1, 28, 28), torch.randint(0, 10, (150,)))
        tr, va = train_validation_indices(150, 30, seed=0)
        configs = [(200, 0.1), (400, 0.5)]
        rows = run_search(ds, tr, va, torch.device("cpu"), seed=0, configs=configs,
                          max_train_batches=1, progress=lambda *_: None)
        self.assertEqual([(r["hidden_size"], r["dropout"]) for r in rows], configs)
        self.assertTrue(all(r["n_val"] == 30 and r["n_train"] == 120 for r in rows))
        self.assertTrue(all(0 <= r["val_n_correct"] <= 30 for r in rows))
        self.assertFalse([k for r in rows for k in r if k.startswith("test")])
        self.assertIn(select_best(rows, configs), rows)


class HeatmapTests(unittest.TestCase):
    def test_ramp_is_light_to_dark(self):
        self.assertEqual(ramp_color(0.0), _hex(BLUE_RAMP[0]))
        self.assertEqual(ramp_color(1.0), _hex(BLUE_RAMP[-1]))
        self.assertEqual(text_color_for(_hex(BLUE_RAMP[0])), (20, 20, 20))       # dark ink on light cell
        self.assertEqual(text_color_for(_hex(BLUE_RAMP[-1])), (255, 255, 255))   # white on dark cell

    def test_renders(self):
        img = heatmap([[1, 2], [3, 4]], ["a", "b"], ["x", "y"], "t", "rows", "cols", highlight=(1, 1))
        self.assertGreater(img.size[0], 200)


class CliAndBaselineTests(unittest.TestCase):
    def test_out_guard(self):
        import q5_mlp_hyperparameter_search as q5
        for out in ("results/question1/x", "results/question4"):
            with self.subTest(out=out), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                q5.parse_args(["--out", out])
        self.assertEqual(q5.parse_args([]).seeds, [0, 1, 2])
        self.assertEqual((q5.N_TRAIN_FULL, q5.VAL_SIZE), (60_000, 6_000))

    def test_validation_matrix_and_digest(self):
        import q5_mlp_hyperparameter_search as q5
        rows = [_row(h, p, 5000 + 10 * k) for k, (h, p) in enumerate(grid())]
        m = q5.validation_matrix(rows)
        self.assertEqual((len(m), len(m[0])), (5, 5))
        self.assertAlmostEqual(m[1][2], rows[7]["val_accuracy_percent"])   # hidden 400, dropout 0.3
        self.assertEqual(q5.index_digest([3, 1, 2]), q5.index_digest([1, 2, 3]))

    def test_load_summary_validation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.json"
            path.write_text(json.dumps({"question": 4, "model": "mlp_enhanced", "device": "gpu", "smoke": False}))
            self.assertEqual(load_summary(path, 4, "mlp_enhanced")["question"], 4)
            for q, m in ((1, "mlp_enhanced"), (4, "mlp")):
                with self.subTest(q=q, m=m), self.assertRaises(ValueError):
                    load_summary(path, q, m)


if __name__ == "__main__":
    unittest.main()
