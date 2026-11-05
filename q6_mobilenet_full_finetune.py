"""Assignment 2 - Question 6: MobileNetV2 with the whole pretrained network fine-tuned.

Identical to the Question 1 MobileNetV2 (same ImageNet weights, preprocessing, classifier
replacement, Adam lr 0.001, batch 100, 3 epochs, CrossEntropyLoss, seeds 0 1 2, warm-up and
timing), except that no layer is frozen. The Question 1 results are only read.

Examples (from the project root, with the venv active):
    python q6_mobilenet_full_finetune.py            # 3 GPU runs, ~30 min
    python q6_mobilenet_full_finetune.py --smoke    # truncated pipeline check -> results/smoke/question6
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch

from a2.compare import (BASELINE_COMPARISON_FIELDS, baseline_comparison, load_gpu_baseline,
                        protocol_differences, sha256)
from a2.data import load_mnist, make_loaders
from a2.environment import TIMING_NOTES, environment
from a2.experiment import run_once, warmup
from a2.io_utils import write_json, write_predictions, write_rows_csv
from a2.models import MODEL_SPECS, MOBILENET_FULL_FT_SPEC, build_mobilenet_v2_full_finetune
from a2.stats import summarize_runs

DEFAULT_OUT = Path("results/question6")
SMOKE_OUT = Path("results/smoke/question6")
SMOKE_TRAIN_BATCHES, SMOKE_TEST_BATCHES = 5, 3

PREPROCESSING = ("Resize((224, 224)) -> Grayscale(num_output_channels=3) -> ToTensor() -> "
                 "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="one run per seed (as in Q1)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--q1", type=Path, default=Path("results/question1"), help="Question 1 results (read-only)")
    p.add_argument("--warmup-batches", type=int, default=20, help="untimed warm-up batches (as in Q1)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (as in Q1: 0)")
    p.add_argument("--log-interval", type=int, default=100, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing Question 6 results")
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


def parameter_counts(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    features = sum(p.numel() for p in model.features.parameters())
    features_trainable = sum(p.numel() for p in model.features.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "features_total": features, "features_trainable": features_trainable,
            "classifier_trainable": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)}


def q1_parameter_counts() -> dict:
    """Counts of the Question 1 model (frozen features), built without downloading weights."""
    from unittest import mock
    from torchvision import models
    real = models.mobilenet_v2
    with mock.patch.object(models, "mobilenet_v2", lambda weights=None: real(weights=None)):
        return parameter_counts(MODEL_SPECS["mobilenetv2"].build())


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch; Question 6 is run on the GPU, like Question 1.")
    if (args.out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{args.out}/summary.json exists; pass --overwrite to replace it.")
    baseline = load_gpu_baseline(args.q1, "mobilenetv2")          # fail fast, before training
    diffs = protocol_differences(baseline["protocol"], seeds=args.seeds,
                                 warmup_batches=args.warmup_batches, num_workers=args.num_workers)
    if diffs and not args.smoke:
        sys.exit(f"protocol differs from the Question 1 MobileNetV2 baseline: {diffs}")
    spec, device = MOBILENET_FULL_FT_SPEC, torch.device("cuda:0")
    q1_spec = MODEL_SPECS["mobilenetv2"]
    same = {f: getattr(spec, f) == getattr(q1_spec, f)
            for f in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input", "make_loss")}
    if not all(same.values()):
        sys.exit(f"training configuration differs from Question 1: {same}")

    args.out.mkdir(parents=True, exist_ok=True)
    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_test = SMOKE_TEST_BATCHES if args.smoke else None
    counts = parameter_counts(build_mobilenet_v2_full_finetune(weights=None))
    q1_counts = q1_parameter_counts()
    props = torch.cuda.get_device_properties(device)
    print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | cuda:0 = {props.name} ({props.gcnArchName}, "
          f"{props.total_memory / 2**30:.1f} GiB)")
    if args.smoke:
        print("SMOKE MODE: truncated training/testing; numbers are not valid results.")
    print(f"=== {spec.name}: {len(args.seeds)} run(s), epochs={spec.epochs}, batch={spec.batch_size}, "
          f"lr={spec.learning_rate}, trainable {counts['trainable']:,} / {counts['total']:,} parameters "
          f"(Q1: {q1_counts['trainable']:,})", flush=True)

    train_ds, test_ds = load_mnist(spec.transform)
    warmup(spec, train_ds, device, args.warmup_batches)            # absorbs one-time kernel tuning
    runs, run_meta = [], []
    for run_no, seed in enumerate(args.seeds, start=1):
        train_loader, test_loader = make_loaders(train_ds, test_ds, spec.batch_size, seed, args.num_workers)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        r = run_once(spec, train_loader, test_loader, device, seed, args.log_interval, max_train, max_test)
        peak_alloc = torch.cuda.max_memory_allocated(device) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
        if not args.smoke and r.n_test != len(test_ds):
            raise RuntimeError(f"evaluated {r.n_test} images, expected {len(test_ds)}")
        write_predictions(args.out / f"predictions_run{run_no}_seed{seed}.csv",
                          r.indices, r.true_labels, r.predicted_labels)
        runs.append({
            "model": spec.name, "run": run_no, "seed": seed,
            "n_correct": r.n_correct, "n_test": r.n_test, "accuracy_percent": r.accuracy_percent,
            "train_seconds": r.train_seconds, "inference_seconds": r.inference_seconds,
            "trainable_parameters": counts["trainable"], "total_parameters": counts["total"],
            "peak_gpu_memory_allocated_gib": round(peak_alloc, 3),
            "peak_gpu_memory_reserved_gib": round(peak_reserved, 3),
            "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        })
        print(f"  run {run_no} (seed {seed}): acc={r.accuracy_percent:.2f}%  train={r.train_seconds:.2f}s  "
              f"inference={r.inference_seconds:.3f}s  peak GPU memory {peak_alloc:.2f} GiB allocated / "
              f"{peak_reserved:.2f} GiB reserved", flush=True)

    agg = summarize_runs(runs)
    comparison = [{k: row[k] for k in BASELINE_COMPARISON_FIELDS}
                  for row in baseline_comparison(agg, baseline["aggregate"])]
    comparison.append({"metric": "trainable_parameters", "baseline_mean": q1_counts["trainable"],
                       "baseline_std_sample": "", "new_mean": counts["trainable"], "new_std_sample": "",
                       "difference": counts["trainable"] - q1_counts["trainable"],
                       "ratio": counts["trainable"] / q1_counts["trainable"]})
    write_rows_csv(args.out / "runs.csv", runs)
    write_rows_csv(args.out / "comparison_vs_q1_mobilenet.csv", comparison)
    baseline_file = args.q1 / "mobilenetv2" / "summary.json"
    config = {
        "question": 6, "model": spec.name, "smoke": args.smoke, "device": "gpu",
        "change_vs_question1": "no layer frozen: requires_grad=True for all parameters "
                               "(Q1 froze model.features); everything else identical",
        "pretrained_weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V1 (same as Question 1)",
        "classifier": "classifier[1] replaced by Linear(1280, 10) after seeding (same as Question 1)",
        "preprocessing": PREPROCESSING,
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam (all model parameters)",
                            "loss": spec.make_loss().__class__.__name__},
        "parameters": {"question6": counts, "question1": q1_counts},
        "protocol": {"seeds": args.seeds, "warmup_batches": args.warmup_batches, "num_workers": args.num_workers,
                     "max_train_batches_per_epoch": max_train, "max_test_batches": max_test,
                     "mixed_precision": False, "gradient_accumulation": False,
                     "std_note": "std_sample uses ddof=1 (n-1)"},
        "timing_definitions": TIMING_NOTES,
        "environment": {**environment(), "gpu_arch": props.gcnArchName,
                        "gpu_total_memory_gib": round(props.total_memory / 2**30, 2)},
        "baseline": {"path": str(baseline_file), "sha256": sha256(baseline_file)},
    }
    write_json(args.out / "config.json", config)
    write_json(args.out / "summary.json", {
        "question": 6, "model": spec.name, "smoke": args.smoke, "device": "gpu",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": config["hyperparameters"], "protocol": config["protocol"],
        "parameters": counts, "aggregate": agg, "comparison_vs_q1_mobilenet": comparison, "runs": runs,
    })
    a, d = agg["accuracy_percent"], {c["metric"]: c for c in comparison}
    print(f"  => accuracy {a['mean']:.2f}% ± {a['std_sample']:.2f} (sample SD, n={a['n']}); "
          f"train {agg['train_seconds']['mean']:.2f}s; inference {agg['inference_seconds']['mean']:.3f}s")
    print(f"  vs Q1 MobileNetV2 (frozen): accuracy {d['accuracy_percent']['difference']:+.2f} pp, "
          f"train x{d['train_seconds']['ratio']:.2f}, inference x{d['inference_seconds']['ratio']:.2f}")


if __name__ == "__main__":
    main()
