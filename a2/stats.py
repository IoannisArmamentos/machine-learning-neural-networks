"""Aggregation of repeated runs."""
from __future__ import annotations

import math
from typing import Sequence

AGGREGATED_FIELDS = ("accuracy_percent", "train_seconds", "inference_seconds")


def mean_std(values: Sequence[float]) -> dict[str, float]:
    """Mean plus sample (ddof=1) and population (ddof=0) standard deviation."""
    n = len(values)
    if n == 0:
        raise ValueError("need at least one value")
    mean = math.fsum(values) / n
    ss = math.fsum((v - mean) ** 2 for v in values)
    return {
        "n": n,
        "mean": mean,
        "std_sample": math.sqrt(ss / (n - 1)) if n > 1 else float("nan"),
        "std_population": math.sqrt(ss / n),
        "min": min(values),
        "max": max(values),
    }


def accuracy_percent(n_correct: int, n_total: int) -> float:
    if n_total <= 0:
        raise ValueError("n_total must be positive")
    return 100.0 * n_correct / n_total


def summarize_runs(runs: Sequence[dict]) -> dict[str, dict[str, float]]:
    return {field: mean_std([float(r[field]) for r in runs]) for field in AGGREGATED_FIELDS}
