"""Question 8: consolidation of the saved Question 1-7 results and an exact McNemar test.

Everything here reads saved CSV files; nothing trains or runs a model.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from a2.misclassification import Prediction, load_predictions


@dataclass(frozen=True)
class ConfigSpec:
    key: str
    label: str
    question: int
    description: str
    runs_csv: str
    prediction_pattern: str           # formatted with run= and seed=
    summary_json: str
    order: int                        # fixed configuration order (last tie-breaker)


@dataclass
class ConfigResult:
    spec: ConfigSpec
    runs: list[dict]
    trainable_parameters: int
    total_parameters: int
    measured: str = ""
    predictions: dict[int, list[Prediction]] = field(default_factory=dict)   # seed -> predictions

    @property
    def total_correct(self) -> int:
        return sum(int(r["n_correct"]) for r in self.runs)

    def _values(self, key):
        return [float(r[key]) for r in self.runs]

    def mean(self, key):
        return statistics.mean(self._values(key))

    def sd(self, key):
        return statistics.stdev(self._values(key))


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def load_config(root: Path, spec: ConfigSpec, expected_seeds=(0, 1, 2), n_test: int = 10_000,
                true_labels: Sequence[int] | None = None) -> tuple[list[dict], dict[int, list[Prediction]]]:
    """Read the runs and every run's prediction file; verify counts, indices, labels and accuracy."""
    runs = read_rows(root / spec.runs_csv)
    seeds = [int(r["seed"]) for r in runs]
    if seeds != list(expected_seeds):
        raise ValueError(f"{spec.key}: seeds {seeds}, expected {list(expected_seeds)}")
    preds = {}
    for r in runs:
        path = root / spec.prediction_pattern.format(run=int(r["run"]), seed=int(r["seed"]))
        p = load_predictions(path, expected_n=n_test)            # exactly 0..n-1, labels 0-9
        if true_labels is not None and [x.true for x in p] != list(true_labels):
            raise ValueError(f"{path}: true labels differ from the reference labels")
        correct = sum(not x.wrong for x in p)
        if correct != int(r["n_correct"]) or not math.isclose(float(r["accuracy_percent"]), 100 * correct / n_test):
            raise ValueError(f"{path}: {correct} correct, runs file says {r['n_correct']} / {r['accuracy_percent']} %")
        preds[int(r["seed"])] = p
    return runs, preds


def rank(results: Sequence[ConfigResult]) -> list[ConfigResult]:
    """By mean accuracy (= total correct over the same number of runs, exact integers), ties broken by
    fewer trainable parameters, then by the fixed configuration order."""
    n = {len(r.runs) for r in results}
    if len(n) != 1:
        raise ValueError("all configurations must have the same number of runs")
    return sorted(results, key=lambda r: (-r.total_correct, r.trainable_parameters, r.spec.order))


def pareto_front(points: Sequence[tuple[str, float, float]]) -> list[str]:
    """(key, accuracy, time): keys not dominated by another point with accuracy >= and time <=,
    at least one strictly better. Ordered by time."""
    front = []
    for k, acc, t in points:
        dominated = any(a2 >= acc and t2 <= t and (a2 > acc or t2 < t) for k2, a2, t2 in points if k2 != k)
        if not dominated:
            front.append((t, k))
    return [k for _, k in sorted(front)]


def paired_table(a: Sequence[Prediction], b: Sequence[Prediction]) -> dict[str, int]:
    """2x2 table over the same test images, aligned by original MNIST index."""
    if [p.index for p in a] != [p.index for p in b]:
        raise ValueError("predictions are not aligned by test index")
    if [p.true for p in a] != [p.true for p in b]:
        raise ValueError("true labels differ between the two prediction sets")
    table = {"both_correct": 0, "a_only_correct": 0, "b_only_correct": 0, "both_incorrect": 0}
    for pa, pb in zip(a, b):
        ca, cb = not pa.wrong, not pb.wrong
        key = ("both_correct" if cb else "a_only_correct") if ca else ("b_only_correct" if cb else "both_incorrect")
        table[key] += 1
    return table


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: under H0 the discordant pairs split Binomial(b + c, 0.5).
    p = min(1, 2 * P(X <= min(b, c))); p = 1 when there are no discordant pairs."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar_report(a: Sequence[Prediction], b: Sequence[Prediction], alpha: float = 0.05) -> dict:
    t = paired_table(a, b)
    n01, n10 = t["a_only_correct"], t["b_only_correct"]
    p = mcnemar_exact(n01, n10)
    n = len(a)
    return {
        "n_images": n, "contingency": t, "discordant_pairs": n01 + n10,
        "a_only_correct": n01, "b_only_correct": n10,
        "accuracy_a_percent": 100 * (t["both_correct"] + n01) / n,
        "accuracy_b_percent": 100 * (t["both_correct"] + n10) / n,
        "p_value_exact_two_sided": p, "alpha": alpha, "significant": p < alpha,
    }
