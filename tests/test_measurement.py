"""Focused tests for timing, aggregation, prediction bookkeeping and output files.

Run: python -m unittest discover -s tests -v
(CPU only, synthetic data; no MNIST download, no GPU required.)
"""
import csv
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from a2.data import IndexedDataset
from a2.experiment import evaluate, run_once
from a2.io_utils import combined_rows, rebuild_combined_summary, write_json, write_predictions
from a2.models import MODEL_SPECS, CNNet, ModelSpec, NeuralNet
from a2.stats import accuracy_percent, mean_std, summarize_runs
from a2.timing import SyncTimer, device_synchronizer


class SyncTimerTests(unittest.TestCase):
    def test_syncs_before_each_clock_read(self):
        events = []
        ticks = iter([10.0, 12.5])

        def clock():
            events.append("clock")
            return next(ticks)

        with SyncTimer(lambda: events.append("sync"), clock) as t:
            events.append("work")
        self.assertEqual(events, ["sync", "clock", "work", "sync", "clock"])
        self.assertAlmostEqual(t.seconds, 2.5)

    def test_syncs_even_if_block_raises(self):
        syncs = []
        ticks = iter([0.0, 1.0])
        with self.assertRaises(RuntimeError):
            with SyncTimer(lambda: syncs.append(1), lambda: next(ticks)):
                raise RuntimeError
        self.assertEqual(len(syncs), 2)

    def test_cpu_synchronizer_is_noop(self):
        self.assertIsNone(device_synchronizer(torch.device("cpu"))())


class StatsTests(unittest.TestCase):
    def test_mean_std_matches_statistics_module(self):
        vals = [96.91, 97.02, 97.01]
        s = mean_std(vals)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], statistics.mean(vals))
        self.assertAlmostEqual(s["std_sample"], statistics.stdev(vals))
        self.assertAlmostEqual(s["std_population"], statistics.pstdev(vals))
        self.assertEqual((s["min"], s["max"]), (96.91, 97.02))

    def test_known_values(self):
        s = mean_std([2.0, 4.0, 6.0])
        self.assertAlmostEqual(s["mean"], 4.0)
        self.assertAlmostEqual(s["std_sample"], 2.0)
        self.assertAlmostEqual(s["std_population"], math.sqrt(8 / 3))

    def test_single_value_and_empty(self):
        self.assertTrue(math.isnan(mean_std([1.0])["std_sample"]))
        self.assertEqual(mean_std([1.0])["std_population"], 0.0)
        with self.assertRaises(ValueError):
            mean_std([])

    def test_accuracy_percent(self):
        self.assertAlmostEqual(accuracy_percent(9701, 10000), 97.01)
        with self.assertRaises(ValueError):
            accuracy_percent(1, 0)

    def test_summarize_runs_fields(self):
        runs = [{"accuracy_percent": a, "train_seconds": t, "inference_seconds": i}
                for a, t, i in [(97.0, 10.0, 0.3), (98.0, 12.0, 0.5)]]
        agg = summarize_runs(runs)
        self.assertAlmostEqual(agg["accuracy_percent"]["mean"], 97.5)
        self.assertAlmostEqual(agg["train_seconds"]["mean"], 11.0)
        self.assertAlmostEqual(agg["inference_seconds"]["std_sample"], statistics.stdev([0.3, 0.5]))


class _FixedClassifier(nn.Module):
    """Predicts class = round(x[:, 0]); lets us know the exact expected predictions."""

    def forward(self, x):
        return nn.functional.one_hot(x[:, 0].round().long().clamp(0, 9), 10).float()


def _toy_spec(build=_FixedClassifier, epochs=1):
    return ModelSpec("toy", build, nn.CrossEntropyLoss, epochs=epochs,
                     transform="tensor", prepare_input=lambda t: t, batch_size=4)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        # 10 samples; feature 0 encodes the predicted class, labels chosen so 7/10 match.
        preds = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.float32)
        self.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
        self.ds = TensorDataset(preds.unsqueeze(1), self.labels)

    def test_counts_correct_and_keeps_original_indices(self):
        loader = DataLoader(IndexedDataset(self.ds), batch_size=4, shuffle=False)
        secs, n_correct, idx, true, pred = evaluate(_FixedClassifier(), _toy_spec(), loader,
                                                    torch.device("cpu"))
        self.assertEqual(n_correct, 7)
        self.assertEqual(idx, list(range(10)))
        self.assertEqual(true, self.labels.tolist())
        self.assertEqual(pred, list(range(10)))
        self.assertGreaterEqual(secs, 0.0)

    def test_indices_survive_reordering(self):
        # Even if a loader reorders samples, indices must still point at the original item.
        loader = DataLoader(IndexedDataset(self.ds), batch_size=3, shuffle=True,
                            generator=torch.Generator().manual_seed(0))
        _, _, idx, true, _ = evaluate(_FixedClassifier(), _toy_spec(), loader, torch.device("cpu"))
        self.assertNotEqual(idx, list(range(10)))
        for i, t in zip(idx, true):
            self.assertEqual(self.labels[i].item(), t)


class RunOnceTests(unittest.TestCase):
    def test_same_seed_reproduces_on_cpu(self):
        torch.manual_seed(123)
        x = torch.rand(40, 1, 28, 28)
        y = torch.randint(0, 10, (40,))
        spec = ModelSpec("mlp_toy", NeuralNet, nn.NLLLoss, epochs=2, transform="tensor",
                         prepare_input=MODEL_SPECS["mlp"].prepare_input, batch_size=8)

        def go(seed):
            train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))
            test_loader = DataLoader(IndexedDataset(TensorDataset(x, y)), batch_size=8)
            return run_once(spec, train_loader, test_loader, torch.device("cpu"), seed, log_interval=0)

        a, b, c = go(1), go(1), go(2)
        self.assertEqual(a.predicted_labels, b.predicted_labels)
        self.assertEqual(a.epoch_mean_loss, b.epoch_mean_loss)
        self.assertNotEqual(a.epoch_mean_loss, c.epoch_mean_loss)
        self.assertEqual(len(a.epoch_seconds), 2)
        self.assertAlmostEqual(a.train_seconds, sum(a.epoch_seconds), delta=0.5)
        self.assertEqual(a.n_test, 40)


class ModelFidelityTests(unittest.TestCase):
    def test_parameter_counts_match_original_architectures(self):
        count = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(count(NeuralNet()), 784 * 100 + 100 + 100 * 10 + 10)
        self.assertEqual(count(CNNet()), (32 * 9 + 32) + (64 * 32 * 9 + 64) + (1600 * 10 + 10))

    def test_output_shapes(self):
        self.assertEqual(NeuralNet()(torch.rand(2, 784)).shape, (2, 10))
        self.assertEqual(CNNet()(torch.rand(2, 1, 28, 28)).shape, (2, 10))

    def test_hyperparameters(self):
        for name, epochs in [("mlp", 5), ("cnn", 5), ("mobilenetv2", 3)]:
            s = MODEL_SPECS[name]
            self.assertEqual((s.epochs, s.batch_size, s.learning_rate), (epochs, 100, 0.001))


class IOTests(unittest.TestCase):
    def test_predictions_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pred.csv"
            write_predictions(p, [5, 9], [3, 4], [3, 7])
            rows = list(csv.DictReader(p.read_text().splitlines()))
            self.assertEqual(rows[0], {"mnist_test_index": "5", "true_label": "3",
                                       "predicted_label": "3", "correct": "1"})
            self.assertEqual(rows[1]["correct"], "0")
            with self.assertRaises(ValueError):
                write_predictions(p, [1], [1, 2], [1])

    def test_combined_summary(self):
        agg = summarize_runs([{"accuracy_percent": 97.0, "train_seconds": 1.0, "inference_seconds": 0.1},
                              {"accuracy_percent": 99.0, "train_seconds": 3.0, "inference_seconds": 0.3}])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cnn").mkdir()
            write_json(root / "cnn" / "summary.json", {"model": "cnn", "aggregate": agg})
            target = rebuild_combined_summary(root, ["mlp", "cnn"])
            rows = list(csv.DictReader(target.read_text().splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "cnn")
            self.assertAlmostEqual(float(rows[0]["accuracy_mean"]), 98.0)
            self.assertAlmostEqual(float(rows[0]["accuracy_std_sample"]), math.sqrt(2))
            self.assertIsNone(rebuild_combined_summary(Path(d) / "empty", ["mlp"]))
        self.assertEqual(combined_rows([{"model": "x", "aggregate": agg}])[0]["n_runs"], 2)


if __name__ == "__main__":
    unittest.main()
