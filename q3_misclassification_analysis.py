"""Assignment 2 - Question 3: misclassification analysis of one Question 1 GPU run per model.

Uses the saved per-image predictions (no retraining, no inference). For every model it
lists the misclassified MNIST test images (original indices), counts errors per true digit
(count and rate), ranks the confused (true -> predicted) pairs, and draws example images.
Across models it finds the images that all three networks misclassify.

Example (from the project root, with the venv active):
    python q3_misclassification_analysis.py            # run 1 (seed 0) of each model
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from torchvision import datasets

from a2.compare import sha256
from a2.figures import Tile, make_grid
from a2.io_utils import write_rows_csv
from a2.misclassification import (DIGITS, check_aligned, common_errors, confusion_pairs,
                                  error_overlap, errors_by_digit, examples_by_digit,
                                  load_predictions, misclassified)

MODELS = ("mlp", "cnn", "mobilenetv2")
LABEL = {"mlp": "MLP", "cnn": "CNN", "mobilenetv2": "MobileNetV2"}
SHORT = {"mlp": "MLP", "cnn": "CNN", "mobilenetv2": "MNV2"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=int, default=1, help="Question 1 run to analyse (default 1 = seed 0)")
    p.add_argument("--q1", type=Path, default=Path("results/question1"), help="Question 1 results (read-only)")
    p.add_argument("--out", type=Path, default=Path("results/question3"))
    p.add_argument("--per-digit", type=int, default=6, help="examples per true digit in the per-digit figure")
    p.add_argument("--top-pairs", type=int, default=5, help="confusion pairs shown in the confusion figure")
    p.add_argument("--overwrite", action="store_true", help="replace existing Question 3 results")
    args = p.parse_args(argv)
    if args.out.resolve() == args.q1.resolve() or args.q1.resolve() in args.out.resolve().parents:
        p.error("--out must not be inside the Question 1 results directory")
    return args


def q1_run(q1: Path, model: str, run: int) -> dict:
    """The row of results/question1/<model>/runs.csv for the chosen run."""
    rows = list(csv.DictReader((q1 / model / "runs.csv").read_text(encoding="utf-8").splitlines()))
    match = [r for r in rows if int(r["run"]) == run]
    if len(match) != 1:
        sys.exit(f"{model}: run {run} not found in {q1 / model / 'runs.csv'}")
    return match[0]


def main(argv=None) -> None:
    args = parse_args(argv)
    if (args.out / "summary.csv").exists() and not args.overwrite:
        sys.exit(f"{args.out}/summary.csv exists; pass --overwrite to replace it.")
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- load and validate predictions of the selected run
    by_model, summary = {}, []
    for m in MODELS:
        run = q1_run(args.q1, m, args.run)
        path = args.q1 / m / f"predictions_run{args.run}_seed{run['seed']}.csv"
        preds = load_predictions(path)                       # exactly 10,000, indices 0..9999
        n_err = len(misclassified(preds))
        if len(preds) - n_err != int(run["n_correct"]):
            sys.exit(f"{m}: {len(preds) - n_err} correct in {path}, runs.csv says {run['n_correct']}")
        by_model[m] = preds
        summary.append({
            "model": m, "run": args.run, "seed": int(run["seed"]), "predictions_file": str(path),
            "predictions_sha256": sha256(path), "n_test": len(preds), "n_correct": len(preds) - n_err,
            "n_errors": n_err, "accuracy_percent": 100.0 * (len(preds) - n_err) / len(preds),
        })
    check_aligned(by_model)

    # --- original MNIST test images; every label must match the saved true label at that index
    test = datasets.MNIST("./data", train=False, download=True)
    images, targets = test.data.numpy(), test.targets.tolist()
    if len(targets) != 10_000:
        sys.exit(f"MNIST test set has {len(targets)} images, expected 10000")
    for m, preds in by_model.items():
        mismatch = [p.index for p in preds if targets[p.index] != p.true]
        if mismatch:
            sys.exit(f"{m}: saved true labels differ from MNIST at indices {mismatch[:5]}")

    # --- tables
    write_rows_csv(args.out / "summary.csv", summary)
    write_rows_csv(args.out / "errors_by_digit.csv",
                   [{"model": m, **row} for m in MODELS for row in errors_by_digit(by_model[m])])
    write_rows_csv(args.out / "confusion_pairs.csv",
                   [{"model": m, **row} for m in MODELS for row in confusion_pairs(by_model[m])])
    common = common_errors(by_model)
    write_rows_csv(args.out / "common_errors.csv", common)
    write_rows_csv(args.out / "error_overlap.csv", error_overlap(by_model))
    write_rows_csv(args.out / "misclassified.csv",
                   [{"model": m, "mnist_test_index": p.index, "true_label": p.true, "predicted_label": p.pred}
                    for m in MODELS for p in misclassified(by_model[m])])

    # --- figures (+ manifest of every image shown, for traceability)
    manifest = []

    def save(name: str, rows, title, row_labels=None, caption_lines=2, tile_width=None, meta=None):
        img, placed = make_grid(rows, title, row_labels, caption_lines, tile_width)
        img.save(fig_dir / name, optimize=True)
        for pl in placed:
            manifest.append({"figure": name, "row": pl.row, "col": pl.col,
                             "box": ";".join(map(str, pl.box)), **meta[(pl.row, pl.col)]})

    def tile(p, extra=""):
        return Tile(images[p.index], [f"#{p.index}{extra}", f"true {p.true} -> pred {p.pred}"])

    for m in MODELS:
        preds = by_model[m]
        # 1) per true digit: the first N errors (lowest test index) of each digit
        chosen = examples_by_digit(preds, args.per_digit)
        rows, meta, labels = [], {}, []
        stats = {r["digit"]: r for r in errors_by_digit(preds)}
        for r, d in enumerate(DIGITS):
            ex = [p for p in chosen if p.true == d]
            rows.append([tile(p) for p in ex])
            labels.append(f"true {d}\n{stats[d]['n_errors']} err")
            for c, p in enumerate(ex):
                meta[(r, c)] = {"model": m, "mnist_test_index": p.index, "true_label": p.true,
                                "predicted_label": str(p.pred)}
        save(f"{m}_errors_by_digit.png", rows,
             f"{LABEL[m]} (Q1 run {args.run}): first {args.per_digit} misclassified test images per true digit",
             labels, meta=meta)

        # 2) most frequent confusion pairs
        pairs = confusion_pairs(preds)[:args.top_pairs]
        errors = misclassified(preds)
        rows, meta, labels = [], {}, []
        for r, pair in enumerate(pairs):
            ex = [p for p in errors if (p.true, p.pred) == (pair["true_label"], pair["predicted_label"])]
            ex = ex[:args.per_digit]
            rows.append([tile(p) for p in ex])
            labels.append(f"{pair['true_label']} -> {pair['predicted_label']}\n{pair['count']} times")
            for c, p in enumerate(ex):
                meta[(r, c)] = {"model": m, "mnist_test_index": p.index, "true_label": p.true,
                                "predicted_label": str(p.pred)}
        save(f"{m}_top_confusions.png", rows,
             f"{LABEL[m]} (Q1 run {args.run}): {len(pairs)} most frequent confusions (true -> predicted)",
             labels, meta=meta)

    # 3) images misclassified by all three models
    cols = 8
    rows, meta = [], {}
    for k, row in enumerate(common):
        r, c = divmod(k, cols)
        if c == 0:
            rows.append([])
        i = row["mnist_test_index"]
        rows[-1].append(Tile(images[i], [
            f"#{i}  true {row['true_label']}",
            " ".join(f"{SHORT[m]} {row[f'{m}_predicted']}" for m in MODELS[:2]),
            f"{SHORT['mobilenetv2']} {row['mobilenetv2_predicted']}",
        ]))
        meta[(r, c)] = {"model": "+".join(MODELS), "mnist_test_index": i, "true_label": row["true_label"],
                        "predicted_label": ";".join(str(row[f"{m}_predicted"]) for m in MODELS)}
    save("common_errors.png", rows,
         f"Misclassified by all three models (Q1 run {args.run}): {len(common)} test images",
         caption_lines=3, tile_width=104, meta=meta)

    write_rows_csv(fig_dir / "figures_manifest.csv", manifest)

    # --- console summary
    for s in summary:
        print(f"{LABEL[s['model']]:12s} errors {s['n_errors']:4d}  accuracy {s['accuracy_percent']:.2f}%")
    print(f"Misclassified by all three models: {len(common)}")
    print(f"Wrote {args.out}/ ({len(list(fig_dir.glob('*.png')))} figures)")


if __name__ == "__main__":
    main()
