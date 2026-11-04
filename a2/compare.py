"""CPU vs GPU comparison against the Question 1 GPU baseline (read-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

COMPARISON_FIELDS = (
    "model", "seed",
    "cpu_accuracy_percent", "gpu_accuracy_mean", "gpu_accuracy_std_sample", "gpu_accuracy_same_seed",
    "cpu_train_seconds", "gpu_train_seconds_mean", "train_speedup",
    "gpu_train_seconds_same_seed", "train_speedup_same_seed",
    "cpu_inference_seconds", "gpu_inference_seconds_mean", "inference_speedup",
    "gpu_inference_seconds_same_seed", "inference_speedup_same_seed",
)


def speedup(cpu_seconds: float, gpu_seconds: float) -> float:
    """How many times faster the GPU is: cpu_time / gpu_time (>1 means the GPU is faster)."""
    if cpu_seconds <= 0 or gpu_seconds <= 0:
        raise ValueError("times must be positive")
    return cpu_seconds / gpu_seconds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gpu_baseline(q1_root: Path, model: str) -> dict:
    """Load and sanity-check results/question1/<model>/summary.json."""
    path = q1_root / model / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if summary.get("question") != 1:
        problems.append("not a Question 1 summary")
    if summary.get("model") != model:
        problems.append(f"model is {summary.get('model')!r}")
    if summary.get("device") != "gpu":
        problems.append("device is not gpu")
    if summary.get("smoke"):
        problems.append("smoke-test results")
    if problems:
        raise ValueError(f"{path}: invalid GPU baseline ({', '.join(problems)})")
    return summary


def comparison_row(cpu_run: dict, gpu_summary: dict) -> dict:
    """Primary speedups use the GPU mean over all Q1 runs; *_same_seed uses the GPU run
    that had the same seed as the CPU run (None if there was none)."""
    agg = gpu_summary["aggregate"]
    seed = cpu_run["seed"]
    same = next((r for r in gpu_summary["runs"] if r["seed"] == seed), None)
    gpu_train, gpu_inf = agg["train_seconds"]["mean"], agg["inference_seconds"]["mean"]
    return {
        "model": cpu_run["model"],
        "seed": seed,
        "cpu_accuracy_percent": cpu_run["accuracy_percent"],
        "gpu_accuracy_mean": agg["accuracy_percent"]["mean"],
        "gpu_accuracy_std_sample": agg["accuracy_percent"]["std_sample"],
        "gpu_accuracy_same_seed": same["accuracy_percent"] if same else None,
        "cpu_train_seconds": cpu_run["train_seconds"],
        "gpu_train_seconds_mean": gpu_train,
        "train_speedup": speedup(cpu_run["train_seconds"], gpu_train),
        "gpu_train_seconds_same_seed": same["train_seconds"] if same else None,
        "train_speedup_same_seed":
            speedup(cpu_run["train_seconds"], same["train_seconds"]) if same else None,
        "cpu_inference_seconds": cpu_run["inference_seconds"],
        "gpu_inference_seconds_mean": gpu_inf,
        "inference_speedup": speedup(cpu_run["inference_seconds"], gpu_inf),
        "gpu_inference_seconds_same_seed": same["inference_seconds"] if same else None,
        "inference_speedup_same_seed":
            speedup(cpu_run["inference_seconds"], same["inference_seconds"]) if same else None,
    }


BASELINE_COMPARISON_FIELDS = (
    "metric", "baseline_mean", "baseline_std_sample", "new_mean", "new_std_sample",
    "difference", "ratio",
)


def baseline_comparison(new_aggregate: dict, baseline_aggregate: dict,
                        metrics=("accuracy_percent", "train_seconds", "inference_seconds")) -> list[dict]:
    """Per metric: difference = new - baseline (accuracy: percentage points; times: seconds,
    positive = slower) and ratio = new / baseline. Inputs are `summarize_runs` aggregates."""
    rows = []
    for m in metrics:
        new, base = new_aggregate[m], baseline_aggregate[m]
        if base["mean"] == 0:
            raise ValueError(f"baseline mean of {m} is zero")
        rows.append({
            "metric": m,
            "baseline_mean": base["mean"],
            "baseline_std_sample": base["std_sample"],
            "new_mean": new["mean"],
            "new_std_sample": new["std_sample"],
            "difference": new["mean"] - base["mean"],
            "ratio": new["mean"] / base["mean"],
        })
    return rows
