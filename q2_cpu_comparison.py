"""Assignment 2 - Question 2: run each original model once on the CPU and compare
training and inference times with the Question 1 GPU baseline.

The GPU is hidden from PyTorch before torch is imported, so nothing can run on it.
Question 1 results are only read, never written.

Examples (from the project root, with the venv active):
    python q2_cpu_comparison.py --model mlp cnn
    python q2_cpu_comparison.py --model mobilenetv2       # long-running
    python q2_cpu_comparison.py --model all --smoke       # quick pipeline check
"""
from __future__ import annotations

import os

# Must happen before torch is imported (directly or via a2). An empty list hides
# every device from HIP/ROCm (and CUDA), so torch.cuda.is_available() is False.
for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ[_var] = ""

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from a2.compare import COMPARISON_FIELDS, comparison_row, load_gpu_baseline, sha256  # noqa: E402
from a2.data import load_mnist, make_loaders  # noqa: E402
from a2.environment import TIMING_NOTES, environment  # noqa: E402
from a2.experiment import run_once, warmup  # noqa: E402
from a2.io_utils import write_json, write_predictions, write_rows_csv  # noqa: E402
from a2.models import MODEL_SPECS  # noqa: E402

DEFAULT_OUT = Path("results/question2")
SMOKE_OUT = Path("results/smoke/question2")
GPU_BASELINE = Path("results/question1")
SMOKE_TRAIN_BATCHES, SMOKE_TEST_BATCHES = 5, 3

WARMUP_NOTE = (
    "Both CPU (Q2) and GPU (Q1) runs are preceded by the same untimed warm-up: N training "
    "batches plus N test-mode forward passes on a throw-away model instance, before seeding. "
    "On the GPU it absorbs one-time kernel compilation/autotuning and device context set-up; "
    "on the CPU there is no kernel compilation, so it only absorbs first-call overheads "
    "(memory allocator growth, oneDNN primitive creation, thread-pool start-up). Timed "
    "regions are identical; device synchronization is a no-op on the CPU because CPU "
    "operators execute synchronously."
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["all"], choices=[*MODEL_SPECS, "all"])
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the single CPU run (default 0 = seed of Q1 GPU run 1)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--gpu-baseline", type=Path, default=GPU_BASELINE,
                   help="Question 1 results root (read-only)")
    p.add_argument("--warmup-batches", type=int, default=20,
                   help="untimed warm-up batches, same default as Q1 (0 disables)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (original scripts and Q1: 0)")
    p.add_argument("--log-interval", type=int, default=100, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing CPU results for a model")
    p.add_argument("--smoke", action="store_true",
                   help=f"pipeline check: {SMOKE_TRAIN_BATCHES} train batches/epoch, "
                        f"{SMOKE_TEST_BATCHES} test batches. Results are NOT valid measurements.")
    args = p.parse_args(argv)
    if "all" in args.model:
        args.model = list(MODEL_SPECS)
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    if args.out.resolve() == args.gpu_baseline.resolve() or \
            args.gpu_baseline.resolve() in args.out.resolve().parents:
        p.error("--out must not be inside the Question 1 results directory")
    return args


def assert_cpu_only() -> torch.device:
    if torch.cuda.is_available():
        sys.exit("A GPU is still visible to PyTorch; refusing to run the CPU experiment.")
    return torch.device("cpu")


def run_model(name: str, args, device: torch.device) -> None:
    spec = MODEL_SPECS[name]
    out_dir = args.out / name
    record_path = out_dir / "cpu_run.json"
    if record_path.exists() and not args.overwrite:
        sys.exit(f"{record_path} exists; pass --overwrite to replace it.")
    baseline = load_gpu_baseline(args.gpu_baseline, name)  # fail fast, before a long run
    out_dir.mkdir(parents=True, exist_ok=True)

    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_test = SMOKE_TEST_BATCHES if args.smoke else None
    print(f"\n=== {name} on CPU: 1 run (seed {args.seed}), epochs={spec.epochs}, "
          f"batch={spec.batch_size}, lr={spec.learning_rate}, threads={torch.get_num_threads()}",
          flush=True)
    train_ds, test_ds = load_mnist(spec.transform)
    warmup(spec, train_ds, device, args.warmup_batches)
    train_loader, test_loader = make_loaders(train_ds, test_ds, spec.batch_size, args.seed,
                                             args.num_workers)
    started = dt.datetime.now().isoformat(timespec="seconds")
    r = run_once(spec, train_loader, test_loader, device, args.seed, args.log_interval,
                 max_train, max_test)
    if not args.smoke and r.n_test != len(test_ds):
        raise RuntimeError(f"evaluated {r.n_test} images, expected {len(test_ds)}")

    write_predictions(out_dir / f"predictions_cpu_seed{args.seed}.csv",
                      r.indices, r.true_labels, r.predicted_labels)
    run = {
        "model": name, "device": "cpu", "seed": args.seed,
        "n_correct": r.n_correct, "n_test": r.n_test,
        "accuracy_percent": r.accuracy_percent,
        "train_seconds": r.train_seconds,
        "inference_seconds": r.inference_seconds,
        "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
        "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
    }
    baseline_file = args.gpu_baseline / name / "summary.json"
    write_json(record_path, {
        "question": 2,
        "model": name,
        "smoke": args.smoke,
        "device": "cpu",
        "started": started,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam",
                            "loss": spec.make_loss().__class__.__name__},
        "protocol": {"runs": 1, "seed": args.seed, "warmup_batches": args.warmup_batches,
                     "num_workers": args.num_workers,
                     "max_train_batches_per_epoch": max_train, "max_test_batches": max_test},
        "timing_definitions": TIMING_NOTES,
        "warmup_methodology": WARMUP_NOTE,
        "environment": environment(),
        "gpu_baseline": {"path": str(baseline_file), "sha256": sha256(baseline_file),
                         "warmup_batches": baseline["protocol"]["warmup_batches"],
                         "num_workers": baseline["protocol"]["num_workers"],
                         "seeds": baseline["protocol"]["seeds"]},
        "run": run,
    })
    print(f"  CPU: acc={r.accuracy_percent:.2f}%  train={r.train_seconds:.2f}s  "
          f"inference={r.inference_seconds:.3f}s", flush=True)


def rebuild_tables(args) -> None:
    """Regenerate the top-level tables from every <model>/cpu_run.json present."""
    records = [json.loads(p.read_text(encoding="utf-8"))
               for name in MODEL_SPECS if (p := args.out / name / "cpu_run.json").exists()]
    if not records:
        return
    runs = [{**rec["run"], "torch_num_threads": rec["environment"]["torch_num_threads"],
             "warmup_batches": rec["protocol"]["warmup_batches"]} for rec in records]
    write_rows_csv(args.out / "cpu_runs.csv", runs)
    rows = [comparison_row(rec["run"], load_gpu_baseline(args.gpu_baseline, rec["model"]))
            for rec in records]
    write_rows_csv(args.out / "cpu_gpu_comparison.csv",
                   [{k: row[k] for k in COMPARISON_FIELDS} for row in rows])
    write_json(args.out / "config.json", {
        "question": 2,
        "smoke": any(rec["smoke"] for rec in records),
        "speedup_definition": "cpu_seconds / gpu_seconds (>1: GPU faster). Primary: GPU mean of "
                              "the 3 Q1 runs. *_same_seed: the Q1 GPU run with the CPU run's seed.",
        "timing_definitions": TIMING_NOTES,
        "warmup_methodology": WARMUP_NOTE,
        "models": {rec["model"]: {k: rec[k] for k in
                                  ("started", "finished", "hyperparameters", "protocol",
                                   "environment", "gpu_baseline")} for rec in records},
    })
    print(f"\nTables: {args.out / 'cpu_runs.csv'}, {args.out / 'cpu_gpu_comparison.csv'}")
    for row in rows:
        print(f"  {row['model']:12s} train x{row['train_speedup']:.2f}  "
              f"inference x{row['inference_speedup']:.2f} (GPU speedup vs Q1 mean)")


def main(argv=None) -> None:
    args = parse_args(argv)
    device = assert_cpu_only()
    print(f"PyTorch {torch.__version__} | CPU only (GPU visible: {torch.cuda.is_available()}) | "
          f"{environment()['cpu']} | intra-op threads {torch.get_num_threads()}")
    if args.smoke:
        print("SMOKE MODE: truncated training/testing; numbers are not valid results.")
    for name in args.model:
        run_model(name, args, device)
    rebuild_tables(args)


if __name__ == "__main__":
    main()
