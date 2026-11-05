"""Tests for Question 8: loading/aggregation of the saved Q1-Q7 results, ranking, McNemar test.
Uses the committed results files (no MNIST download, no GPU, no training or inference)."""
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from a2.final_comparison import (ConfigResult, ConfigSpec, load_config, mcnemar_exact, mcnemar_report,
                                 paired_table, pareto_front, rank, read_rows)
from a2.io_utils import write_predictions, write_rows_csv
from a2.misclassification import Prediction

ROOT = Path(__file__).resolve().parents[1]


def spec(key, order=1):
    return ConfigSpec(key, key, 1, "", "runs.csv", "p_run{run}_seed{seed}.csv", "s.json", order)


def result(key, correct, trainable, order):
    runs = [{"n_correct": c, "accuracy_percent": c / 100, "train_seconds": 1, "inference_seconds": 1} for c in correct]
    return ConfigResult(spec(key, order), runs, trainable, trainable)


def preds(pairs):
    return [Prediction(i, t, p) for i, (t, p) in enumerate(pairs)]


class SavedResultsTests(unittest.TestCase):
    """Loads the real saved results of all seven configurations."""

    @classmethod
    def setUpClass(cls):
        import q8_final_comparison as q8
        cls.q8 = q8
        cls.loaded = {s.key: load_config(ROOT, s) for s in q8.CONFIGS}

    def test_seven_configurations_three_runs_each(self):
        self.assertEqual(len(self.q8.CONFIGS), 7)
        self.assertEqual(len({s.key for s in self.q8.CONFIGS}), 7)
        self.assertEqual([s.order for s in self.q8.CONFIGS], list(range(1, 8)))
        for key, (runs, p) in self.loaded.items():
            self.assertEqual([int(r["seed"]) for r in runs], [0, 1, 2], key)
            self.assertEqual(sorted(p), [0, 1, 2])
            for seed, pr in p.items():
                self.assertEqual([x.index for x in pr], list(range(10_000)))     # 10,000 unique original indices

    def test_accuracies_match_predictions_and_labels_agree(self):
        reference = [x.true for x in self.loaded["q1_mlp"][1][0]]
        for key, (runs, p) in self.loaded.items():
            for r in runs:
                pr = p[int(r["seed"])]
                self.assertEqual([x.true for x in pr], reference, key)
                self.assertEqual(sum(not x.wrong for x in pr), int(r["n_correct"]), key)

    def test_parameter_counts(self):
        expected = {"q1_mlp": (79_510, 79_510), "q1_cnn": (34_826, 34_826), "q1_mobilenetv2_frozen": (12_810, 2_236_682),
                    "q4_enhanced_mlp": (494_710, 494_710), "q5_tuned_mlp": (636_010, 636_010),
                    "q6_mobilenetv2_full": (2_236_682, 2_236_682), "q7_mobilenetv2_last3": (1_218_890, 2_236_682)}
        for s in self.q8.CONFIGS:
            summary = json.loads((ROOT / s.summary_json).read_text())
            self.assertEqual(self.q8.parameter_counts(ROOT, s, summary), expected[s.key], s.key)

    def test_ranking_of_saved_results(self):
        results = [ConfigResult(s, self.loaded[s.key][0], 1, 1) for s in self.q8.CONFIGS]
        order = [r.spec.key for r in rank(results)]
        self.assertEqual(order[:3], ["q6_mobilenetv2_full", "q7_mobilenetv2_last3", "q1_cnn"])
        self.assertEqual(order[-1], "q1_mobilenetv2_frozen")
        means = [statistics.mean(float(x["accuracy_percent"]) for x in self.loaded[k][0]) for k in order]
        self.assertEqual(means, sorted(means, reverse=True))

    def test_timing_means_from_saved_runs(self):
        t = {k: statistics.mean(float(r["train_seconds"]) for r in runs) for k, (runs, _) in self.loaded.items()}
        self.assertEqual(min(t, key=t.get), "q5_tuned_mlp")
        self.assertEqual(max(t, key=t.get), "q6_mobilenetv2_full")
        self.assertGreater(t["q6_mobilenetv2_full"] / t["q7_mobilenetv2_last3"], 4)

    def test_question5_selection_used_validation_only(self):
        check = self.q8.q5_selection_check(ROOT)
        self.assertTrue(check["passed"], check)


class RankingTests(unittest.TestCase):
    def test_rank_by_total_correct_then_parameters_then_order(self):
        rs = [result("a", [9800, 9800, 9800], 100, 1), result("b", [9900, 9800, 9700], 50, 2),   # tie on total
              result("c", [9900, 9900, 9900], 999, 3), result("d", [9850, 9750, 9800], 50, 4)]   # tie with b incl. params
        self.assertEqual([r.spec.key for r in rank(rs)], ["c", "b", "d", "a"])
        self.assertEqual([r.spec.key for r in rank(list(reversed(rs)))], ["c", "b", "d", "a"])

    def test_unequal_run_counts_rejected(self):
        with self.assertRaises(ValueError):
            rank([result("a", [1, 2, 3], 1, 1), result("b", [1, 2], 1, 2)])

    def test_pareto_front(self):
        pts = [("fast_ok", 97.0, 9.0), ("fast_good", 98.9, 9.5), ("slow_best", 99.0, 600.0),
               ("dominated", 96.0, 120.0), ("mid", 99.0, 127.0)]
        # slow_best has equal accuracy to mid but is slower -> dominated
        self.assertEqual(pareto_front(pts), ["fast_ok", "fast_good", "mid"])


class McNemarTests(unittest.TestCase):
    def test_exact_p_values(self):
        self.assertAlmostEqual(mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)
        self.assertAlmostEqual(mcnemar_exact(10, 0), 2 / 1024)
        self.assertEqual(mcnemar_exact(7, 7), 1.0)
        self.assertEqual(mcnemar_exact(3, 9), mcnemar_exact(9, 3))
        # cross-check against the binomial CDF computed directly
        n, k = 171, 82
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        self.assertAlmostEqual(mcnemar_exact(89, 82), min(1, 2 * tail))
        with self.assertRaises(ValueError):
            mcnemar_exact(-1, 2)

    def test_zero_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        a = preds([(1, 1), (2, 3), (4, 4)])
        rep = mcnemar_report(a, list(a))                                  # identical predictions
        self.assertEqual(rep["discordant_pairs"], 0)
        self.assertEqual(rep["p_value_exact_two_sided"], 1.0)
        self.assertFalse(rep["significant"])
        self.assertEqual(rep["contingency"], {"both_correct": 2, "a_only_correct": 0, "b_only_correct": 0,
                                              "both_incorrect": 1})

    def test_contingency_table(self):
        a = preds([(1, 1), (2, 2), (3, 0), (4, 0), (5, 5), (6, 0)])
        b = preds([(1, 1), (2, 0), (3, 3), (4, 0), (5, 0), (6, 6)])
        rep = mcnemar_report(a, b)
        self.assertEqual(rep["contingency"], {"both_correct": 1, "a_only_correct": 2, "b_only_correct": 2,
                                              "both_incorrect": 1})
        self.assertEqual(sum(rep["contingency"].values()), 6)
        self.assertAlmostEqual(rep["accuracy_a_percent"], 50.0)
        self.assertEqual((rep["a_only_correct"], rep["b_only_correct"], rep["discordant_pairs"]), (2, 2, 4))
        self.assertEqual(rep["alpha"], 0.05)

    def test_alignment_and_label_validation(self):
        a = preds([(1, 1), (2, 2)])
        with self.assertRaises(ValueError):
            paired_table(a, [Prediction(1, 2, 2), Prediction(0, 1, 1)])     # different index order
        with self.assertRaises(ValueError):
            paired_table(a, preds([(1, 1), (3, 3)]))                        # true labels differ


class LoadConfigTests(unittest.TestCase):
    def _make(self, d, seeds, n_correct=None):
        rows = []
        for run, seed in enumerate(seeds, start=1):
            write_predictions(Path(d) / f"p_run{run}_seed{seed}.csv", [0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 0])
            rows.append({"run": run, "seed": seed, "n_correct": n_correct or 3, "accuracy_percent": 75.0})
        write_rows_csv(Path(d) / "runs.csv", rows)

    def test_valid_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            self._make(d, [0, 1, 2])
            runs, p = load_config(Path(d), spec("x"), n_test=4, true_labels=[1, 2, 3, 4])
            self.assertEqual(len(runs), 3)
            with self.assertRaises(ValueError):
                load_config(Path(d), spec("x"), n_test=4, true_labels=[1, 2, 3, 5])      # label mismatch
        with tempfile.TemporaryDirectory() as d:
            self._make(d, [0, 1, 3])
            with self.assertRaises(ValueError):
                load_config(Path(d), spec("x"), n_test=4)                                # wrong seeds
        with tempfile.TemporaryDirectory() as d:
            self._make(d, [0, 1, 2], n_correct=4)
            with self.assertRaises(ValueError):
                load_config(Path(d), spec("x"), n_test=4)                                # accuracy mismatch

    def test_out_dir_guard(self):
        import q8_final_comparison as q8
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            q8.parse_args(["--out", "results/question7"])
        self.assertEqual(str(q8.parse_args([]).out), "results/question8")


if __name__ == "__main__":
    unittest.main()
