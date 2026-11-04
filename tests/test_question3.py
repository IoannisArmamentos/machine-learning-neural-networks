"""Tests for the Question 3 misclassification analysis (synthetic data; no MNIST, no GPU)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from a2.figures import Tile, crop_digit, make_grid
from a2.io_utils import write_predictions
from a2.misclassification import (Prediction, check_aligned, common_errors, confusion_pairs,
                                  error_overlap, errors_by_digit, examples_by_digit,
                                  load_predictions, misclassified, validate)


def preds_from(pairs):
    """[(true, pred), ...] -> predictions with indices 0..n-1."""
    return [Prediction(i, t, p) for i, (t, p) in enumerate(pairs)]


class LoadAndValidateTests(unittest.TestCase):
    def _write(self, d, indices, true, pred):
        path = Path(d) / "p.csv"
        write_predictions(path, indices, true, pred)
        return path

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, [0, 1, 2], [7, 2, 1], [7, 8, 1])
            preds = load_predictions(path, expected_n=3)
        self.assertEqual(preds, [Prediction(0, 7, 7), Prediction(1, 2, 8), Prediction(2, 1, 1)])

    def test_rejects_wrong_count_and_bad_indices(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                load_predictions(self._write(d, [0, 1, 2], [1, 1, 1], [1, 1, 1]), expected_n=10_000)
            for idx in ([0, 2, 3], [1, 0, 2], [0, 0, 1]):     # gap, out of order, duplicate
                with self.subTest(idx=idx), self.assertRaises(ValueError):
                    load_predictions(self._write(d, idx, [1, 1, 1], [1, 1, 1]), expected_n=3)

    def test_rejects_inconsistent_correct_column(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "p.csv"
            path.write_text("mnist_test_index,true_label,predicted_label,correct\n0,3,5,1\n")
            with self.assertRaises(ValueError):
                load_predictions(path, expected_n=1)

    def test_rejects_labels_outside_0_9(self):
        with self.assertRaises(ValueError):
            validate([Prediction(0, 3, 10)])


class CountingTests(unittest.TestCase):
    def setUp(self):
        # digit 0: 4 samples, 2 wrong (50 %); digit 1: 10 samples, 3 wrong (30 %)
        self.preds = preds_from([(0, 0), (0, 0), (0, 6), (0, 6)] + [(1, 1)] * 7 + [(1, 7), (1, 7), (1, 4)])

    def test_misclassified(self):
        wrong = misclassified(self.preds)
        self.assertEqual([p.index for p in wrong], [2, 3, 11, 12, 13])

    def test_count_versus_rate(self):
        rows = {r["digit"]: r for r in errors_by_digit(self.preds)}
        # digit 1 has more errors, but digit 0 has the higher error *rate*
        self.assertEqual((rows[0]["n_errors"], rows[1]["n_errors"]), (2, 3))
        self.assertAlmostEqual(rows[0]["error_rate_percent"], 50.0)
        self.assertAlmostEqual(rows[1]["error_rate_percent"], 30.0)
        self.assertAlmostEqual(rows[1]["share_of_errors_percent"], 60.0)
        self.assertEqual(rows[7]["n_predicted_as_wrongly"], 2)
        self.assertEqual(rows[5]["n_test_samples"], 0)
        self.assertEqual(rows[5]["error_rate_percent"], 0.0)
        self.assertEqual(sum(r["n_errors"] for r in rows.values()), 5)

    def test_confusion_pairs_ranked_with_deterministic_ties(self):
        pairs = confusion_pairs(self.preds)
        got = [(p["true_label"], p["predicted_label"], p["count"]) for p in pairs]
        self.assertEqual(got, [(0, 6, 2), (1, 7, 2), (1, 4, 1)])    # tie (0,6)/(1,7) broken by labels
        self.assertEqual([p["rank"] for p in pairs], [1, 2, 3])
        self.assertAlmostEqual(pairs[0]["percent_of_true_class"], 50.0)
        self.assertAlmostEqual(pairs[1]["percent_of_true_class"], 20.0)
        self.assertAlmostEqual(sum(p["percent_of_model_errors"] for p in pairs), 100.0)

    def test_examples_by_digit_takes_lowest_indices(self):
        ex = examples_by_digit(self.preds, per_digit=2)
        self.assertEqual([p.index for p in ex], [2, 3, 11, 12])


class CommonErrorTests(unittest.TestCase):
    def setUp(self):
        truth = [3, 5, 7, 1, 9, 0]
        self.models = {
            "a": [Prediction(i, t, p) for i, (t, p) in enumerate(zip(truth, [5, 5, 1, 1, 4, 0]))],  # wrong 0,2,4
            "b": [Prediction(i, t, p) for i, (t, p) in enumerate(zip(truth, [5, 3, 1, 1, 9, 6]))],  # wrong 0,1,2,5
            "c": [Prediction(i, t, p) for i, (t, p) in enumerate(zip(truth, [8, 5, 1, 7, 9, 0]))],  # wrong 0,2,3
        }

    def test_intersection_and_predictions(self):
        rows = common_errors(self.models)
        self.assertEqual([r["mnist_test_index"] for r in rows], [0, 2])
        self.assertEqual(rows[0], {"mnist_test_index": 0, "true_label": 3, "a_predicted": 5,
                                   "b_predicted": 5, "c_predicted": 8, "same_wrong_prediction": 0})
        self.assertEqual(rows[1]["same_wrong_prediction"], 1)       # all predicted 1 for a 7

    def test_pairwise_overlap_and_jaccard(self):
        rows = {r["models"]: r for r in error_overlap(self.models)}
        self.assertEqual(rows["a+b"]["common_errors"], 2)            # {0,2}
        self.assertEqual(rows["a+b"]["union_errors"], 5)             # {0,1,2,4,5}
        self.assertAlmostEqual(rows["a+b"]["jaccard"], 2 / 5)
        self.assertEqual(rows["a+b+c"]["common_errors"], 2)
        self.assertEqual(rows["a+b+c"]["errors_each"], "3;4;3")

    def test_misaligned_models_are_rejected(self):
        shifted = dict(self.models)
        shifted["c"] = [Prediction(i, (p.true + 1) % 10, p.pred) for i, p in enumerate(self.models["c"])]
        with self.assertRaises(ValueError):
            check_aligned(shifted)
        with self.assertRaises(ValueError):
            common_errors(shifted)


class FigureTests(unittest.TestCase):
    @staticmethod
    def digit(value):
        img = np.zeros((28, 28), dtype=np.uint8)
        img[value % 28, :] = 255          # a distinct horizontal stripe per image
        img[:, 0] = value
        return img

    def test_each_tile_holds_its_own_image(self):
        images = {i: self.digit(i) for i in range(1, 8)}
        rows = [[Tile(images[1], ["#1", "a"]), None, Tile(images[2], ["#2", "b"])],
                [Tile(images[i], [f"#{i}", "c"]) for i in range(3, 8)]]
        png, placed = make_grid(rows, "t", row_labels=["r0", "r1"])
        expected = {(0, 0): 1, (0, 2): 2, **{(1, c): c + 3 for c in range(5)}}
        self.assertEqual({(p.row, p.col) for p in placed}, set(expected))
        for p in placed:
            np.testing.assert_array_equal(crop_digit(png, p.box), images[expected[(p.row, p.col)]])

    def test_rejects_wrong_image_shape(self):
        with self.assertRaises(ValueError):
            make_grid([[Tile(np.zeros((32, 32), dtype=np.uint8), ["x"])]], "t")


class CliTests(unittest.TestCase):
    def test_out_may_not_be_inside_question1(self):
        import q3_misclassification_analysis as q3
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            q3.parse_args(["--out", "results/question1/q3"])
        self.assertEqual(q3.parse_args([]).run, 1)

    def test_selects_requested_run(self):
        import q3_misclassification_analysis as q3
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "mlp").mkdir()
            (Path(d) / "mlp" / "runs.csv").write_text("model,run,seed,n_correct\nmlp,1,0,9698\nmlp,2,1,9691\n")
            self.assertEqual(q3.q1_run(Path(d), "mlp", 2)["seed"], "1")
            with self.assertRaises(SystemExit):
                q3.q1_run(Path(d), "mlp", 3)


if __name__ == "__main__":
    unittest.main()
