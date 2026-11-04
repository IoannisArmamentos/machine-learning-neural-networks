"""Assignment 2 - Question 5: original MLP + dropout after the hidden ReLU. Grid search over
hidden size {200..1000} x dropout {0.1..0.5} on a fixed 54,000 / 6,000 train/validation split,
then the winner is retrained from scratch on all 60,000 training images (3 GPU runs, seeds
0 1 2) and evaluated on the 10,000-image test set.

Phase 1 (search) never loads the test set. The selection uses validation accuracy only and is
written to best_config.json before the test set is loaded in phase 2.

Examples (from the project root, with the venv active):
    python q5_mlp_hyperparameter_search.py             # ~5 min on the GPU
    python q5_mlp_hyperparameter_search.py --smoke     # truncated pipeline check -> results/smoke/question5
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset
from torchvision import datasets

from a2.compare import (BASELINE_COMPARISON_FIELDS, baseline_comparison, load_gpu_baseline,
                        load_summary, sha256)
from a2.data import DATA_ROOT, load_mnist, make_loaders, make_transform, train_validation_indices
from a2.environment import TIMING_NOTES, environment
from a2.experiment import run_once, warmup
from a2.figures import heatmap
from a2.io_utils import write_json, write_predictions, write_rows_csv
from a2.models import DropoutMLP, dropout_mlp_spec
from a2.search import DROPOUTS, HIDDEN_SIZES, TIE_BREAK_RULE, grid, rank, run_search, select_best
from a2.stats import summarize_runs

DEFAULT_OUT = Path("results/question5")
SMOKE_OUT = Path("results/smoke/question5")
N_TRAIN_FULL, VAL_SIZE = 60_000, 6_000            # 10 % of the training set for validation
SPLIT_SEED = 0                                    # fixed split, the same for all 25 configurations
SEARCH_SEED = 0                                   # same initialisation/shuffle seed for every configuration
SMOKE_TRAIN_BATCHES, SMOKE_EVAL_BATCHES = 5, 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="final-run seeds (default: 0 1 2, as in Question 1)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--q1", type=Path, default=Path("results/question1"), help="Question 1 results (read-only)")
    p.add_argument("--q4", type=Path, default=Path("results/question4"), help="Question 4 results (read-only)")
    p.add_argument("--warmup-batches", type=int, default=20, help="untimed warm-up batches (as in Q1)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (as in Q1: 0)")
    p.add_argument("--log-interval", type=int, default=0, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing Question 5 results")
    p.add_argument("--smoke", action="store_true",
                   help=f"pipeline check: {SMOKE_TRAIN_BATCHES} train batches/epoch, {SMOKE_EVAL_BATCHES} "
                        "evaluation batches. Results are NOT valid measurements.")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    for protected in (args.q1, args.q4):
        if args.out.resolve() == protected.resolve() or protected.resolve() in args.out.resolve().parents:
            p.error(f"--out must not be inside {protected}")
    return args


def index_digest(indices) -> str:
    return hashlib.sha256(",".join(map(str, sorted(indices))).encode()).hexdigest()


def validation_matrix(rows: list[dict]) -> list[list[float]]:
    by = {(r["hidden_size"], r["dropout"]): r["val_accuracy_percent"] for r in rows}
    return [[by[(h, p)] for p in DROPOUTS] for h in HIDDEN_SIZES]


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch; Question 5 is run on the GPU, like Question 1.")
    if (args.out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{args.out}/summary.json exists; pass --overwrite to replace it.")
    q1 = load_gpu_baseline(args.q1, "mlp")                       # fail fast, before training
    q4 = load_summary(args.q4 / "summary.json", 4, "mlp_enhanced")
    device = torch.device("cuda:0")
    args.out.mkdir(parents=True, exist_ok=True)
    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_eval = SMOKE_EVAL_BATCHES if args.smoke else None
    print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | {torch.cuda.get_device_name(0)}")
    if args.smoke:
        print("SMOKE MODE: truncated training/evaluation; numbers are not valid results.")

    # ---------------- Phase 1: search on the training set only ----------------
    search_started = dt.datetime.now().isoformat(timespec="seconds")
    transform = dropout_mlp_spec(HIDDEN_SIZES[0], DROPOUTS[0]).transform
    train_full = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=make_transform(transform))
    if len(train_full) != N_TRAIN_FULL:
        sys.exit(f"training set has {len(train_full)} images, expected {N_TRAIN_FULL}")
    train_idx, val_idx = train_validation_indices(len(train_full), VAL_SIZE, SPLIT_SEED)
    assert len(train_idx) == 54_000 and len(val_idx) == 6_000 and not set(train_idx) & set(val_idx)
    configs = grid()
    print(f"=== Phase 1: {len(configs)} configurations, train {len(train_idx)} / validation {len(val_idx)} "
          f"(split seed {SPLIT_SEED}), seed {SEARCH_SEED}", flush=True)
    warmup(dropout_mlp_spec(HIDDEN_SIZES[0], DROPOUTS[0]), Subset(train_full, train_idx), device,
           args.warmup_batches)
    rows = run_search(train_full, train_idx, val_idx, device, SEARCH_SEED, configs, args.log_interval,
                      args.num_workers, max_train, max_eval)
    best = select_best(rows, expected_configs=configs)
    ranked = rank(rows)
    write_rows_csv(args.out / "validation_grid.csv", ranked)
    best_cfg = {
        "question": 5, "smoke": args.smoke,
        "hidden_size": best["hidden_size"], "dropout": best["dropout"],
        "val_accuracy_percent": best["val_accuracy_percent"], "val_n_correct": best["val_n_correct"],
        "n_val": best["n_val"],
        "selection_criterion": "validation accuracy only (test set not loaded during the search)",
        "tie_break_rule": TIE_BREAK_RULE,
        "n_configurations": len(rows),
        "tied_with_best": [[r["hidden_size"], r["dropout"]] for r in rows
                           if r["val_n_correct"] == best["val_n_correct"] and r is not best],
        "runner_up": {k: ranked[1][k] for k in ("hidden_size", "dropout", "val_accuracy_percent")},
        "split": {"method": "torch.utils.data.random_split", "seed": SPLIT_SEED,
                  "n_train": len(train_idx), "n_val": len(val_idx),
                  "val_indices_sha256": index_digest(val_idx), "train_indices_sha256": index_digest(train_idx)},
        "search_seed": SEARCH_SEED,
        "selected_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(args.out / "best_config.json", best_cfg)
    b_row, b_col = HIDDEN_SIZES.index(best["hidden_size"]), DROPOUTS.index(best["dropout"])
    heatmap(validation_matrix(rows), [str(h) for h in HIDDEN_SIZES], [f"{p:.1f}" for p in DROPOUTS],
            f"Question 5: validation accuracy (%), {len(val_idx):,} validation images",
            "Hidden neurons", "Dropout probability", highlight=(b_row, b_col),
            scale_label="validation accuracy (%)").save(args.out / "validation_grid.png", optimize=True)
    print(f"  => best: hidden={best['hidden_size']}, dropout={best['dropout']}, "
          f"val acc={best['val_accuracy_percent']:.2f}%  (written to best_config.json)", flush=True)

    # ---------------- Phase 2: retrain the winner on all 60,000, evaluate on test ----------------
    spec = dropout_mlp_spec(best["hidden_size"], best["dropout"])
    train_ds, test_ds = load_mnist(spec.transform)              # test set is loaded only now
    print(f"=== Phase 2: {spec.name}, {len(args.seeds)} run(s) from scratch on {len(train_ds)} images", flush=True)
    warmup(spec, train_ds, device, args.warmup_batches)
    final = []
    for run_no, seed in enumerate(args.seeds, start=1):
        train_loader, test_loader = make_loaders(train_ds, test_ds, spec.batch_size, seed, args.num_workers)
        r = run_once(spec, train_loader, test_loader, device, seed, args.log_interval, max_train, max_eval)
        if not args.smoke and r.n_test != len(test_ds):
            raise RuntimeError(f"evaluated {r.n_test} images, expected {len(test_ds)}")
        write_predictions(args.out / f"predictions_final_run{run_no}_seed{seed}.csv",
                          r.indices, r.true_labels, r.predicted_labels)
        final.append({
            "model": spec.name, "hidden_size": best["hidden_size"], "dropout": best["dropout"],
            "run": run_no, "seed": seed, "n_train": len(train_ds),
            "n_correct": r.n_correct, "n_test": r.n_test, "accuracy_percent": r.accuracy_percent,
            "train_seconds": r.train_seconds, "inference_seconds": r.inference_seconds,
            "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        })
        print(f"  run {run_no} (seed {seed}): test acc={r.accuracy_percent:.2f}%  "
              f"train={r.train_seconds:.2f}s  inference={r.inference_seconds:.3f}s", flush=True)
    agg = summarize_runs(final)
    write_rows_csv(args.out / "final_runs.csv", final)
    comparison = [{"baseline": name, **{k: row[k] for k in BASELINE_COMPARISON_FIELDS}}
                  for name, base in (("q1_original_mlp", q1), ("q4_enhanced_mlp", q4))
                  for row in baseline_comparison(agg, base["aggregate"])]
    write_rows_csv(args.out / "comparison.csv", comparison)
    model = DropoutMLP(best["hidden_size"], best["dropout"])
    write_json(args.out / "summary.json", {
        "question": 5, "model": spec.name, "smoke": args.smoke, "device": "gpu",
        "search_started": search_started, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "architecture": {"layers": f"Linear(784,{best['hidden_size']}) -> ReLU -> Dropout({best['dropout']}) "
                                   f"-> Linear({best['hidden_size']},10) -> log_softmax",
                         "parameters": sum(p.numel() for p in model.parameters())},
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam",
                            "loss": spec.make_loss().__class__.__name__},
        "search": {"hidden_sizes": list(HIDDEN_SIZES), "dropouts": list(DROPOUTS),
                   "n_configurations": len(rows), "best_config_file": str(args.out / "best_config.json")},
        "protocol": {"seeds": args.seeds, "warmup_batches": args.warmup_batches,
                     "num_workers": args.num_workers, "final_training_images": len(train_ds),
                     "max_train_batches_per_epoch": max_train, "max_eval_batches": max_eval,
                     "std_note": "std_sample uses ddof=1 (n-1)"},
        "timing_definitions": {**TIMING_NOTES, "val_inference_seconds":
                               "Full pass over the 6,000 validation images (same boundaries as inference)."},
        "environment": environment(),
        "baselines": {name: {"path": str(path), "sha256": sha256(path)} for name, path in
                      (("q1_original_mlp", args.q1 / "mlp" / "summary.json"),
                       ("q4_enhanced_mlp", args.q4 / "summary.json"))},
        "aggregate": agg,
        "comparison": comparison,
        "runs": final,
    })
    a = agg["accuracy_percent"]
    print(f"  => test accuracy {a['mean']:.2f}% ± {a['std_sample']:.2f} (sample SD, n={a['n']}); "
          f"train {agg['train_seconds']['mean']:.2f}s; inference {agg['inference_seconds']['mean']:.3f}s")
    for c in comparison:
        if c["metric"] == "accuracy_percent":
            print(f"  vs {c['baseline']}: {c['difference']:+.2f} pp")


if __name__ == "__main__":
    main()
