"""Assignment 2 - Question 1: run each original model 3x on the GPU.

Records per run: test accuracy, training time, inference (test-set) time, and
per-image predictions keyed by the original MNIST test index (for Question 3).

Examples (from the project root, with the venv active):
    python q1_gpu_baseline.py --model mlp
    python q1_gpu_baseline.py --model mlp cnn mobilenetv2
    python q1_gpu_baseline.py --model all --smoke      # seconds-long pipeline check
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch

from a2.data import load_mnist, make_loaders
from a2.environment import TIMING_NOTES, environment
from a2.experiment import run_once, warmup
from a2.io_utils import rebuild_combined_summary, write_json, write_predictions, write_rows_csv
from a2.models import MODEL_SPECS
from a2.stats import summarize_runs

DEFAULT_OUT = Path("results/question1")
SMOKE_OUT = Path("results/smoke/question1")
SMOKE_TRAIN_BATCHES, SMOKE_TEST_BATCHES = 5, 3

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["all"], choices=[*MODEL_SPECS, "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="one independent run per seed (default: 0 1 2)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--warmup-batches", type=int, default=20,
                   help="untimed GPU warm-up batches before the runs (0 disables)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (original scripts: 0). Affects timing.")
    p.add_argument("--log-interval", type=int, default=100, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing results for a model")
    p.add_argument("--smoke", action="store_true",
                   help=f"pipeline check: {SMOKE_TRAIN_BATCHES} train batches/epoch, "
                        f"{SMOKE_TEST_BATCHES} test batches. Results are NOT valid measurements.")
    args = p.parse_args(argv)
    if "all" in args.model:
        args.model = list(MODEL_SPECS)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    return args


def run_model(name: str, args, device: torch.device) -> None:
    spec = MODEL_SPECS[name]
    out_dir = args.out / name
    if (out_dir / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{out_dir}/summary.json exists; pass --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_test = SMOKE_TEST_BATCHES if args.smoke else None

    print(f"\n=== {name}: {len(args.seeds)} run(s), epochs={spec.epochs}, "
          f"batch={spec.batch_size}, lr={spec.learning_rate}", flush=True)
    train_ds, test_ds = load_mnist(spec.transform)
    warmup(spec, train_ds, device, args.warmup_batches)

    runs = []
    for run_no, seed in enumerate(args.seeds, start=1):
        train_loader, test_loader = make_loaders(train_ds, test_ds, spec.batch_size, seed,
                                                 args.num_workers)
        r = run_once(spec, train_loader, test_loader, device, seed, args.log_interval,
                     max_train, max_test)
        if not args.smoke and r.n_test != len(test_ds):
            raise RuntimeError(f"evaluated {r.n_test} images, expected {len(test_ds)}")
        write_predictions(out_dir / f"predictions_run{run_no}_seed{seed}.csv",
                          r.indices, r.true_labels, r.predicted_labels)
        row = {
            "model": name, "run": run_no, "seed": seed,
            "n_correct": r.n_correct, "n_test": r.n_test,
            "accuracy_percent": r.accuracy_percent,
            "train_seconds": r.train_seconds,
            "inference_seconds": r.inference_seconds,
            "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        }
        runs.append(row)
        print(f"  run {run_no} (seed {seed}): acc={r.accuracy_percent:.2f}%  "
              f"train={r.train_seconds:.2f}s  inference={r.inference_seconds:.3f}s", flush=True)

    write_rows_csv(out_dir / "runs.csv", runs)
    agg = summarize_runs(runs)
    write_json(out_dir / "summary.json", {
        "question": 1,
        "model": name,
        "smoke": args.smoke,
        "device": "gpu",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam",
                            "loss": spec.make_loss().__class__.__name__},
        "protocol": {"seeds": args.seeds, "warmup_batches": args.warmup_batches,
                     "num_workers": args.num_workers,
                     "max_train_batches_per_epoch": max_train, "max_test_batches": max_test,
                     "std_note": "std_sample uses ddof=1 (n-1)"},
        "timing_definitions": TIMING_NOTES,
        "environment": environment(),
        "aggregate": agg,
        "runs": runs,
    })
    a = agg["accuracy_percent"]
    print(f"  => accuracy {a['mean']:.2f}% ± {a['std_sample']:.2f} (sample SD, n={a['n']}); "
          f"train {agg['train_seconds']['mean']:.2f}s; "
          f"inference {agg['inference_seconds']['mean']:.3f}s", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch; Question 1 must run on the GPU.")
    device = torch.device("cuda:0")
    print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | {torch.cuda.get_device_name(0)}")
    if args.smoke:
        print("SMOKE MODE: truncated training/testing; numbers are not valid results.")
    for name in args.model:
        run_model(name, args, device)
    combined = rebuild_combined_summary(args.out, list(MODEL_SPECS))
    if combined:
        print(f"\nCombined table: {combined}")


if __name__ == "__main__":
    main()
