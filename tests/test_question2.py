"""Tests for the Question 2 CPU-vs-GPU comparison logic (CPU only, no MNIST, no GPU needed)."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from a2.compare import COMPARISON_FIELDS, comparison_row, load_gpu_baseline, speedup
from a2.environment import environment
from a2.io_utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def _gpu_summary(model="mlp", **overrides):
    runs = [{"seed": s, "accuracy_percent": a, "train_seconds": t, "inference_seconds": i}
            for s, a, t, i in [(0, 97.0, 10.0, 0.2), (1, 98.0, 12.0, 0.3), (2, 99.0, 14.0, 0.4)]]
    summary = {
        "question": 1, "model": model, "device": "gpu", "smoke": False,
        "protocol": {"seeds": [0, 1, 2], "warmup_batches": 20, "num_workers": 0},
        "aggregate": {
            "accuracy_percent": {"mean": 98.0, "std_sample": 1.0},
            "train_seconds": {"mean": 12.0},
            "inference_seconds": {"mean": 0.3},
        },
        "runs": runs,
    }
    summary.update(overrides)
    return summary


class SpeedupTests(unittest.TestCase):
    def test_ratio_direction(self):
        self.assertAlmostEqual(speedup(120.0, 10.0), 12.0)   # GPU 12x faster
        self.assertAlmostEqual(speedup(5.0, 10.0), 0.5)      # GPU slower

    def test_rejects_non_positive(self):
        for cpu, gpu in [(0, 1), (1, 0), (-1, 1)]:
            with self.assertRaises(ValueError):
                speedup(cpu, gpu)


class ComparisonRowTests(unittest.TestCase):
    def test_mean_and_same_seed(self):
        cpu = {"model": "mlp", "seed": 1, "accuracy_percent": 97.5,
               "train_seconds": 60.0, "inference_seconds": 1.5}
        row = comparison_row(cpu, _gpu_summary())
        self.assertEqual(tuple(row), COMPARISON_FIELDS)
        self.assertAlmostEqual(row["train_speedup"], 60.0 / 12.0)
        self.assertAlmostEqual(row["inference_speedup"], 1.5 / 0.3)
        self.assertEqual(row["gpu_train_seconds_same_seed"], 12.0)
        self.assertAlmostEqual(row["train_speedup_same_seed"], 5.0)
        self.assertAlmostEqual(row["inference_speedup_same_seed"], 5.0)
        self.assertEqual(row["gpu_accuracy_same_seed"], 98.0)
        self.assertEqual((row["gpu_accuracy_mean"], row["gpu_accuracy_std_sample"]), (98.0, 1.0))

    def test_seed_without_gpu_counterpart(self):
        cpu = {"model": "mlp", "seed": 42, "accuracy_percent": 97.0,
               "train_seconds": 30.0, "inference_seconds": 0.6}
        row = comparison_row(cpu, _gpu_summary())
        self.assertIsNone(row["train_speedup_same_seed"])
        self.assertIsNone(row["gpu_accuracy_same_seed"])
        self.assertAlmostEqual(row["train_speedup"], 2.5)


class BaselineLoadingTests(unittest.TestCase):
    def _write(self, root, summary, model="mlp"):
        (root / model).mkdir(parents=True, exist_ok=True)
        write_json(root / model / "summary.json", summary)

    def test_accepts_valid_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(Path(d), _gpu_summary())
            self.assertEqual(load_gpu_baseline(Path(d), "mlp")["device"], "gpu")

    def test_rejects_invalid_baselines(self):
        bad = [{"smoke": True}, {"device": "cpu"}, {"question": 2}, {"model": "cnn"}]
        for override in bad:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as d:
                self._write(Path(d), _gpu_summary(**override))
                with self.assertRaises(ValueError):
                    load_gpu_baseline(Path(d), "mlp")

    def test_missing_baseline(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            load_gpu_baseline(Path(d), "mlp")


class CpuOnlyTests(unittest.TestCase):
    def test_importing_q2_hides_gpu_from_torch(self):
        # Fresh interpreter: the env vars must be set before torch initializes HIP/CUDA.
        code = ("import os, q2_cpu_comparison, torch; "
                "print(torch.cuda.is_available(), repr(os.environ['HIP_VISIBLE_DEVICES']))")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, timeout=120, check=True).stdout.strip().splitlines()[-1]
        self.assertEqual(out, "False ''")

    def test_assert_cpu_only_refuses_visible_gpu(self):
        import q2_cpu_comparison as q2
        with mock.patch("torch.cuda.is_available", return_value=True), self.assertRaises(SystemExit):
            q2.assert_cpu_only()
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(q2.assert_cpu_only().type, "cpu")

    def test_out_dir_may_not_be_inside_question1(self):
        import q2_cpu_comparison as q2
        for out in ["results/question1", "results/question1/cpu"]:
            with self.subTest(out=out), self.assertRaises(SystemExit), \
                    mock.patch("sys.stderr"):
                q2.parse_args(["--out", out])
        self.assertEqual(q2.parse_args([]).out, Path("results/question2"))
        self.assertEqual(q2.parse_args(["--model", "all"]).model, ["mlp", "cnn", "mobilenetv2"])

    def test_environment_without_gpu(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            env = environment()
        self.assertIsNone(env["gpu"])
        self.assertFalse(env["gpu_visible_to_pytorch"])
        self.assertGreaterEqual(env["torch_num_threads"], 1)


if __name__ == "__main__":
    unittest.main()
