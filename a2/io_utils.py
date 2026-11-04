"""CSV/JSON writers for experiment results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

PREDICTION_FIELDS = ("mnist_test_index", "true_label", "predicted_label", "correct")
COMBINED_FIELDS = ("model", "n_runs", "accuracy_mean", "accuracy_std_sample",
                   "train_seconds_mean", "train_seconds_std_sample",
                   "inference_seconds_mean", "inference_seconds_std_sample")


def write_predictions(path: Path, indices: Sequence[int], true: Sequence[int],
                      pred: Sequence[int]) -> None:
    if not (len(indices) == len(true) == len(pred)):
        raise ValueError("indices/true/pred lengths differ")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PREDICTION_FIELDS)
        for i, t, p in zip(indices, true, pred):
            w.writerow((i, t, p, int(t == p)))


def write_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def combined_rows(summaries: Iterable[dict]) -> list[dict]:
    rows = []
    for s in summaries:
        a = s["aggregate"]
        rows.append({
            "model": s["model"],
            "n_runs": a["accuracy_percent"]["n"],
            "accuracy_mean": a["accuracy_percent"]["mean"],
            "accuracy_std_sample": a["accuracy_percent"]["std_sample"],
            "train_seconds_mean": a["train_seconds"]["mean"],
            "train_seconds_std_sample": a["train_seconds"]["std_sample"],
            "inference_seconds_mean": a["inference_seconds"]["mean"],
            "inference_seconds_std_sample": a["inference_seconds"]["std_sample"],
        })
    return rows


def rebuild_combined_summary(out_root: Path, model_order: Sequence[str]) -> Path | None:
    """Collect <out_root>/<model>/summary.json for every model run so far."""
    summaries = []
    for name in model_order:
        p = out_root / name / "summary.json"
        if p.exists():
            summaries.append(json.loads(p.read_text(encoding="utf-8")))
    if not summaries:
        return None
    target = out_root / "summary_all_models.csv"
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMBINED_FIELDS)
        w.writeheader()
        w.writerows(combined_rows(summaries))
    return target
