"""Misclassification analysis on saved per-image predictions (pure Python, no torch)."""
from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import NamedTuple, Sequence

DIGITS = tuple(range(10))


class Prediction(NamedTuple):
    index: int   # original MNIST test-set index
    true: int
    pred: int

    @property
    def wrong(self) -> bool:
        return self.true != self.pred


def load_predictions(path: Path, expected_n: int | None = 10_000) -> list[Prediction]:
    """Read a predictions CSV (mnist_test_index,true_label,predicted_label,correct) and validate it."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    preds = [Prediction(int(r["mnist_test_index"]), int(r["true_label"]), int(r["predicted_label"]))
             for r in rows]
    for r, p in zip(rows, preds):
        if int(r["correct"]) != int(not p.wrong):
            raise ValueError(f"{path}: 'correct' column inconsistent at index {p.index}")
    validate(preds, expected_n, str(path))
    return preds


def validate(preds: Sequence[Prediction], expected_n: int | None = None, name: str = "predictions") -> None:
    n = len(preds)
    if expected_n is not None and n != expected_n:
        raise ValueError(f"{name}: {n} predictions, expected {expected_n}")
    if [p.index for p in preds] != list(range(n)):
        raise ValueError(f"{name}: indices are not exactly 0..{n - 1} in order")
    bad = [p.index for p in preds if p.true not in DIGITS or p.pred not in DIGITS]
    if bad:
        raise ValueError(f"{name}: labels outside 0-9 at indices {bad[:5]}")


def misclassified(preds: Sequence[Prediction]) -> list[Prediction]:
    return [p for p in preds if p.wrong]


def errors_by_digit(preds: Sequence[Prediction]) -> list[dict]:
    """Per true digit: support, error count, error rate (count / support) and share of all errors.
    `n_predicted_as_wrongly` counts how often the digit was the (wrong) predicted label."""
    support = Counter(p.true for p in preds)
    errors = misclassified(preds)
    wrong_true = Counter(p.true for p in errors)
    wrong_pred = Counter(p.pred for p in errors)
    total = len(errors)
    return [{
        "digit": d,
        "n_test_samples": support[d],
        "n_errors": wrong_true[d],
        "error_rate_percent": 100.0 * wrong_true[d] / support[d] if support[d] else 0.0,
        "share_of_errors_percent": 100.0 * wrong_true[d] / total if total else 0.0,
        "n_predicted_as_wrongly": wrong_pred[d],
    } for d in DIGITS]


def confusion_pairs(preds: Sequence[Prediction]) -> list[dict]:
    """All off-diagonal (true, predicted) pairs with count > 0, most frequent first.
    Ties are broken by (true, predicted) so the order is deterministic."""
    support = Counter(p.true for p in preds)
    errors = misclassified(preds)
    counts = Counter((p.true, p.pred) for p in errors)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{
        "rank": rank,
        "true_label": t,
        "predicted_label": pr,
        "count": c,
        "percent_of_true_class": 100.0 * c / support[t],
        "percent_of_model_errors": 100.0 * c / len(errors),
    } for rank, ((t, pr), c) in enumerate(ranked, start=1)]


def check_aligned(by_model: dict[str, Sequence[Prediction]]) -> None:
    """All models must cover the same test indices with the same true labels."""
    names = list(by_model)
    ref = [(p.index, p.true) for p in by_model[names[0]]]
    for name in names[1:]:
        if [(p.index, p.true) for p in by_model[name]] != ref:
            raise ValueError(f"{name} is not aligned with {names[0]} (indices or true labels differ)")


def error_index_sets(by_model: dict[str, Sequence[Prediction]]) -> dict[str, set[int]]:
    return {m: {p.index for p in preds if p.wrong} for m, preds in by_model.items()}


def common_errors(by_model: dict[str, Sequence[Prediction]]) -> list[dict]:
    """Test samples misclassified by every model, with each model's prediction."""
    check_aligned(by_model)
    sets = error_index_sets(by_model)
    shared = sorted(set.intersection(*sets.values()))
    first = next(iter(by_model.values()))
    rows = []
    for i in shared:
        preds = {m: by_model[m][i].pred for m in by_model}
        rows.append({
            "mnist_test_index": i,
            "true_label": first[i].true,
            **{f"{m}_predicted": v for m, v in preds.items()},
            "same_wrong_prediction": int(len(set(preds.values())) == 1),
        })
    return rows


def error_overlap(by_model: dict[str, Sequence[Prediction]]) -> list[dict]:
    """Shared errors for every pair of models and for all models together; Jaccard = |A∩B| / |A∪B|."""
    check_aligned(by_model)
    sets = error_index_sets(by_model)
    groups = [list(c) for c in combinations(sets, 2)] + [list(sets)]
    rows = []
    for g in groups:
        inter = set.intersection(*(sets[m] for m in g))
        union = set.union(*(sets[m] for m in g))
        rows.append({
            "models": "+".join(g),
            "errors_each": ";".join(str(len(sets[m])) for m in g),
            "common_errors": len(inter),
            "union_errors": len(union),
            "jaccard": len(inter) / len(union) if union else 0.0,
        })
    return rows


def examples_by_digit(preds: Sequence[Prediction], per_digit: int) -> list[Prediction]:
    """Deterministic, unbiased selection: the first `per_digit` errors (lowest test index) of each true digit."""
    errors = misclassified(preds)
    out: list[Prediction] = []
    for d in DIGITS:
        out.extend([p for p in errors if p.true == d][:per_digit])
    return out
