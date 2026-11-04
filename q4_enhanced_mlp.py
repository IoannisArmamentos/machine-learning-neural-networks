"""Assignment 2 - Question 4: enhanced MLP (784-500-200-10, ReLU + Dropout 0.2 after each
hidden layer), 3 GPU runs with the Question 1 protocol, compared with the Question 1 MLP.

The Question 1 MLP results are only read (never rerun or written).

Examples (from the project root, with the venv active):
    python q4_enhanced_mlp.py                  # seeds 0 1 2, ~40 s
    python q4_enhanced_mlp.py --smoke          # truncated pipeline check -> results/smoke/question4
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch

from a2.compare import (BASELINE_COMPARISON_FIELDS, baseline_comparison, load_gpu_baseline,
                        sha256)
from a2.data import load_mnist, make_loaders
from a2.environment import TIMING_NOTES, environment
from a2.experiment import run_once, warmup
from a2.io_utils import write_json, write_predictions, write_rows_csv
from a2.models import ENHANCED_MLP_SPEC, EnhancedMLP
from a2.stats import summarize_runs

DEFAULT_OUT = Path("results/question4")
SMOKE_OUT = Path("results/smoke/question4")
Q1_ROOT = Path("results/question1")
SMOKE_TRAIN_BATCHES, SMOKE_TEST_BATCHES = 5, 3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="one independent run per seed (default: 0 1 2, as in Question 1)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--q1", type=Path, default=Q1_ROOT, help="Question 1 results root (read-only)")
    p.add_argument("--warmup-batches", type=int, default=20, help="untimed warm-up batches (as in Q1)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (as in Q1: 0)")
    p.add_argument("--log-interval", type=int, default=100, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing Question 4 results")
    p.add_argument("--smoke", action="store_true",
                   help=f"pipeline check: {SMOKE_TRAIN_BATCHES} train batches/epoch, "
                        f"{SMOKE_TEST_BATCHES} test batches. Results are NOT valid measurements.")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    if args.out.resolve() == args.q1.resolve() or args.q1.resolve() in args.out.resolve().parents:
        p.error("--out must not be inside the Question 1 results directory")
    return args


def architecture() -> dict:
    m = EnhancedMLP()
    return {
        "layers": "Linear(784,500) -> ReLU -> Dropout(0.2) -> Linear(500,200) -> ReLU -> "
                  "Dropout(0.2) -> Linear(200,10) -> log_softmax",
        "parameters": sum(p.numel() for p in m.parameters()),
    }


def check_protocol(baseline: dict, args) -> None:
    """The comparison is only meaningful with the Question 1 protocol."""
    proto = baseline["protocol"]
    diffs = [k for k, mine in (("seeds", args.seeds), ("warmup_batches", args.warmup_batches),
                               ("num_workers", args.num_workers)) if proto[k] != mine]
    if diffs and not args.smoke:
        sys.exit(f"protocol differs from the Question 1 baseline in {diffs}; "
                 f"baseline uses {[proto[k] for k in diffs]}")


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch; Question 4 is run on the GPU, like Question 1.")
    if (args.out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{args.out}/summary.json exists; pass --overwrite to replace it.")
    baseline = load_gpu_baseline(args.q1, "mlp")             # fail fast, before training
    check_protocol(baseline, args)
    baseline_file = args.q1 / "mlp" / "summary.json"

    spec, device = ENHANCED_MLP_SPEC, torch.device("cuda:0")
    args.out.mkdir(parents=True, exist_ok=True)
    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_test = SMOKE_TEST_BATCHES if args.smoke else None
    print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | {torch.cuda.get_device_name(0)}")
    if args.smoke:
        print("SMOKE MODE: truncated training/testing; numbers are not valid results.")
    print(f"=== {spec.name}: {len(args.seeds)} run(s), epochs={spec.epochs}, batch={spec.batch_size}, "
          f"lr={spec.learning_rate}, parameters={architecture()['parameters']}", flush=True)

    train_ds, test_ds = load_mnist(spec.transform)
    warmup(spec, train_ds, device, args.warmup_batches)
    runs = []
    for run_no, seed in enumerate(args.seeds, start=1):
        train_loader, test_loader = make_loaders(train_ds, test_ds, spec.batch_size, seed, args.num_workers)
        r = run_once(spec, train_loader, test_loader, device, seed, args.log_interval, max_train, max_test)
        if not args.smoke and r.n_test != len(test_ds):
            raise RuntimeError(f"evaluated {r.n_test} images, expected {len(test_ds)}")
        write_predictions(args.out / f"predictions_run{run_no}_seed{seed}.csv",
                          r.indices, r.true_labels, r.predicted_labels)
        runs.append({
            "model": spec.name, "run": run_no, "seed": seed,
            "n_correct": r.n_correct, "n_test": r.n_test,
            "accuracy_percent": r.accuracy_percent,
            "train_seconds": r.train_seconds,
            "inference_seconds": r.inference_seconds,
            "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        })
        print(f"  run {run_no} (seed {seed}): acc={r.accuracy_percent:.2f}%  "
              f"train={r.train_seconds:.2f}s  inference={r.inference_seconds:.3f}s", flush=True)

    agg = summarize_runs(runs)
    comparison = baseline_comparison(agg, baseline["aggregate"])
    write_rows_csv(args.out / "runs.csv", runs)
    write_rows_csv(args.out / "comparison_vs_q1_mlp.csv",
                   [{k: row[k] for k in BASELINE_COMPARISON_FIELDS} for row in comparison])
    write_json(args.out / "summary.json", {
        "question": 4,
        "model": spec.name,
        "smoke": args.smoke,
        "device": "gpu",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "architecture": architecture(),
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam",
                            "loss": spec.make_loss().__class__.__name__, "dropout": 0.2},
        "protocol": {"seeds": args.seeds, "warmup_batches": args.warmup_batches,
                     "num_workers": args.num_workers,
                     "max_train_batches_per_epoch": max_train, "max_test_batches": max_test,
                     "std_note": "std_sample uses ddof=1 (n-1)"},
        "timing_definitions": TIMING_NOTES,
        "environment": environment(),
        "baseline": {"path": str(baseline_file), "sha256": sha256(baseline_file),
                     "model": "mlp (Question 1, original architecture)",
                     "comparison_note": "difference = Q4 - Q1 (accuracy in percentage points, "
                                        "times in seconds); ratio = Q4 / Q1"},
        "aggregate": agg,
        "comparison_vs_q1_mlp": comparison,
        "runs": runs,
    })

    a, d = agg["accuracy_percent"], {c["metric"]: c for c in comparison}
    print(f"  => accuracy {a['mean']:.2f}% ± {a['std_sample']:.2f} (sample SD, n={a['n']}); "
          f"train {agg['train_seconds']['mean']:.2f}s; inference {agg['inference_seconds']['mean']:.3f}s")
    print(f"  vs Q1 MLP: accuracy {d['accuracy_percent']['difference']:+.2f} pp, "
          f"train x{d['train_seconds']['ratio']:.2f}, inference x{d['inference_seconds']['ratio']:.2f}")


if __name__ == "__main__":
    main()
