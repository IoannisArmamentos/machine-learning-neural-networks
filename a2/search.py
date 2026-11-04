"""Question 5: grid search over (hidden size, dropout) on a fixed train/validation split.

The search only ever receives the training dataset and the split indices; the test set is not
an argument of anything in this module, so it cannot influence the selection.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

import torch

from a2.data import make_split_loaders
from a2.experiment import run_once
from a2.models import dropout_mlp_spec

HIDDEN_SIZES = (200, 400, 600, 800, 1000)
DROPOUTS = (0.1, 0.2, 0.3, 0.4, 0.5)
TIE_BREAK_RULE = ("highest validation accuracy (number of correct validation images); "
                  "ties broken by fewer hidden neurons, then by lower dropout probability")


def grid(hidden_sizes: Sequence[int] = HIDDEN_SIZES,
         dropouts: Sequence[float] = DROPOUTS) -> list[tuple[int, float]]:
    """All (hidden_size, dropout) pairs, hidden-size-major order."""
    return list(product(hidden_sizes, dropouts))


def run_search(train_ds, train_idx: Sequence[int], val_idx: Sequence[int], device: torch.device,
               seed: int, configs: Sequence[tuple[int, float]], log_interval: int = 0,
               num_workers: int = 0, max_train_batches: int | None = None,
               max_val_batches: int | None = None, progress=print) -> list[dict]:
    """Train each configuration on the training subset and evaluate it on the validation subset.
    Every configuration uses the same split, seed (initialisation and shuffle order) and protocol."""
    val_set = set(val_idx)
    rows = []
    for k, (hidden, p) in enumerate(configs, start=1):
        spec = dropout_mlp_spec(hidden, p)
        train_loader, val_loader = make_split_loaders(train_ds, train_idx, val_idx, spec.batch_size,
                                                      seed, num_workers)
        r = run_once(spec, train_loader, val_loader, device, seed, log_interval,
                     max_train_batches, max_val_batches)
        if not set(r.indices) <= val_set:
            raise RuntimeError("evaluation used images outside the validation subset")
        if max_val_batches is None and sorted(r.indices) != sorted(val_idx):
            raise RuntimeError("evaluation did not cover the whole validation subset")
        rows.append({
            "hidden_size": hidden, "dropout": p, "seed": seed,
            "n_train": len(train_idx), "n_val": r.n_test, "val_n_correct": r.n_correct,
            "val_accuracy_percent": r.accuracy_percent,
            "train_seconds": r.train_seconds, "val_inference_seconds": r.inference_seconds,
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        })
        progress(f"  [{k:2d}/{len(configs)}] hidden={hidden:4d} dropout={p:.1f}: "
                 f"val acc={r.accuracy_percent:.2f}% ({r.n_correct}/{r.n_test})  "
                 f"train={r.train_seconds:.2f}s  val inference={r.inference_seconds:.3f}s")
    return rows


def _check_no_test_fields(rows: Sequence[dict]) -> None:
    leaked = sorted({k for r in rows for k in r if k.lower().startswith("test")})
    if leaked:
        raise ValueError(f"selection must use validation results only; found test fields {leaked}")


def select_best(rows: Sequence[dict], expected_configs: Sequence[tuple[int, float]] | None = None) -> dict:
    """Winner by validation correct count only, with the documented deterministic tie-break."""
    if not rows:
        raise ValueError("no search results")
    _check_no_test_fields(rows)
    if expected_configs is not None:
        got = sorted((r["hidden_size"], r["dropout"]) for r in rows)
        if got != sorted(expected_configs):
            raise ValueError("search results do not cover exactly the expected configurations")
    return min(rows, key=lambda r: (-r["val_n_correct"], r["hidden_size"], r["dropout"]))


def rank(rows: Sequence[dict]) -> list[dict]:
    """Rows sorted by the selection order, with a 1-based `rank` column."""
    _check_no_test_fields(rows)
    ordered = sorted(rows, key=lambda r: (-r["val_n_correct"], r["hidden_size"], r["dropout"]))
    return [{"rank": i, **r} for i, r in enumerate(ordered, start=1)]
