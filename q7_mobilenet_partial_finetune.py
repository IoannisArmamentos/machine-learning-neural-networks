"""Assignment 2 - Question 7: MobileNetV2 where only the last 3 layers of the pretrained network
(plus the new classifier) are fine-tuned.

"Last 3 layers" = the last three top-level modules of `model.features`: features[16] and
features[17] (InvertedResidual blocks) and features[18] (final 1x1 Conv-BN-ReLU6). Everything
else is identical to the Question 1 MobileNetV2 (weights, preprocessing, classifier, Adam lr 0.001,
batch 100, 3 epochs, CrossEntropyLoss, seeds 0 1 2, warm-up and timing). Question 1 and 6 results
are only read.

Examples (from the project root, with the venv active):
    python q7_mobilenet_partial_finetune.py            # 3 GPU runs, ~7 min
    python q7_mobilenet_partial_finetune.py --smoke    # truncated pipeline check -> results/smoke/question7
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import torch

from a2.compare import (BASELINE_COMPARISON_FIELDS, baseline_comparison, load_gpu_baseline,
                        load_summary, protocol_differences, sha256)
from a2.data import load_mnist, make_loaders
from a2.environment import TIMING_NOTES, environment
from a2.experiment import run_once, warmup
from a2.io_utils import write_json, write_predictions, write_rows_csv
from a2.models import (MODEL_SPECS, MOBILENET_PARTIAL_FT_SPEC, PARTIAL_FT_FEATURE_INDICES,
                       build_mobilenet_v2_full_finetune, build_mobilenet_v2_partial_finetune)
from a2.stats import summarize_runs
from q6_mobilenet_full_finetune import PREPROCESSING, parameter_counts, q1_parameter_counts

DEFAULT_OUT = Path("results/question7")
SMOKE_OUT = Path("results/smoke/question7")
SMOKE_TRAIN_BATCHES, SMOKE_TEST_BATCHES = 5, 3
INTERPRETATION = (
    "'The last 3 layers of the pretrained network' = the last three top-level modules of model.features "
    "(features[16], features[17], features[18]). Each top-level entry of torchvision's model.features is one "
    "layer unit of the network (the first conv, 17 inverted-residual bottleneck blocks, the final 1x1 conv); "
    "this reading includes the true final layer and keeps each bottleneck block whole. The new 10-class "
    "classifier is trained as well, as in every MobileNetV2 variant of the assignment.")
ALTERNATIVES = {
    "last_3_inverted_residual_blocks": {"modules": "features[15], features[16], features[17]",
                                        "pretrained_parameters": 1_113_920,
                                        "rejected_because": "leaves the actual last layer features[18] frozen"},
    "last_3_parameterised_leaf_ops": {"modules": "features[17].conv[3] (BN), features[18][0] (Conv), features[18][1] (BN)",
                                      "pretrained_parameters": 412_800,
                                      "rejected_because": "splits an inverted-residual block and counts BatchNorm as a layer"},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="one run per seed (as in Q1)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output root (default: {DEFAULT_OUT}, or {SMOKE_OUT} with --smoke)")
    p.add_argument("--q1", type=Path, default=Path("results/question1"), help="Question 1 results (read-only)")
    p.add_argument("--q6", type=Path, default=Path("results/question6"), help="Question 6 results (read-only)")
    p.add_argument("--warmup-batches", type=int, default=20, help="untimed warm-up batches (as in Q1)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (as in Q1: 0)")
    p.add_argument("--log-interval", type=int, default=100, help="print loss every N steps (0: off)")
    p.add_argument("--overwrite", action="store_true", help="replace existing Question 7 results")
    p.add_argument("--smoke", action="store_true",
                   help=f"pipeline check: {SMOKE_TRAIN_BATCHES} train batches/epoch, "
                        f"{SMOKE_TEST_BATCHES} test batches. Results are NOT valid measurements.")
    args = p.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be distinct")
    if args.out is None:
        args.out = SMOKE_OUT if args.smoke else DEFAULT_OUT
    for protected in (args.q1, args.q6):
        if args.out.resolve() == protected.resolve() or protected.resolve() in args.out.resolve().parents:
            p.error(f"--out must not be inside {protected}")
    return args


def unfrozen_layers(model: torch.nn.Module) -> list[dict]:
    """Every top-level module of model.features / classifier with trainable parameters."""
    rows = []
    for name, parent in (("features", model.features), ("classifier", model.classifier)):
        for i, mod in enumerate(parent):
            params = list(mod.parameters())
            if params and any(p.requires_grad for p in params):
                rows.append({"module": f"{name}[{i}]", "type": type(mod).__name__,
                             "parameters": sum(p.numel() for p in params),
                             "all_trainable": all(p.requires_grad for p in params)})
    return rows


def main(argv=None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("No GPU visible to PyTorch; Question 7 is run on the GPU, like Question 1.")
    if (args.out / "summary.json").exists() and not args.overwrite:
        sys.exit(f"{args.out}/summary.json exists; pass --overwrite to replace it.")
    q1 = load_gpu_baseline(args.q1, "mobilenetv2")                   # fail fast, before training
    q6 = load_summary(args.q6 / "summary.json", 6, "mobilenetv2_full_finetune")
    for name, base in (("Question 1", q1), ("Question 6", q6)):
        diffs = protocol_differences(base["protocol"], seeds=args.seeds,
                                     warmup_batches=args.warmup_batches, num_workers=args.num_workers)
        if diffs and not args.smoke:
            sys.exit(f"protocol differs from the {name} baseline: {diffs}")
    spec, device = MOBILENET_PARTIAL_FT_SPEC, torch.device("cuda:0")
    same = {f: getattr(spec, f) == getattr(MODEL_SPECS["mobilenetv2"], f)
            for f in ("epochs", "batch_size", "learning_rate", "transform", "prepare_input", "make_loss")}
    if not all(same.values()):
        sys.exit(f"training configuration differs from Question 1: {same}")

    args.out.mkdir(parents=True, exist_ok=True)
    max_train = SMOKE_TRAIN_BATCHES if args.smoke else None
    max_test = SMOKE_TEST_BATCHES if args.smoke else None
    probe = build_mobilenet_v2_partial_finetune(weights=None)
    counts, layers = parameter_counts(probe), unfrozen_layers(probe)
    counts_q1 = q1_parameter_counts()
    counts_q6 = parameter_counts(build_mobilenet_v2_full_finetune(weights=None))
    props = torch.cuda.get_device_properties(device)
    print(f"PyTorch {torch.__version__} | HIP {torch.version.hip} | cuda:0 = {props.name} ({props.gcnArchName}, "
          f"{props.total_memory / 2**30:.1f} GiB)")
    if args.smoke:
        print("SMOKE MODE: truncated training/testing; numbers are not valid results.")
    print(f"=== {spec.name}: {len(args.seeds)} run(s), epochs={spec.epochs}, batch={spec.batch_size}, "
          f"lr={spec.learning_rate}; trainable {counts['trainable']:,} / {counts['total']:,} parameters; "
          f"unfrozen: {', '.join(l['module'] for l in layers)}", flush=True)

    train_ds, test_ds = load_mnist(spec.transform)
    warmup(spec, train_ds, device, args.warmup_batches)
    runs = []
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
            "unfrozen_layers": ";".join(l["module"] for l in layers),
            "peak_gpu_memory_allocated_gib": round(peak_alloc, 3),
            "peak_gpu_memory_reserved_gib": round(peak_reserved, 3),
            "epoch_seconds": ";".join(f"{s:.4f}" for s in r.epoch_seconds),
            "epoch_mean_loss": ";".join(f"{x:.6f}" for x in r.epoch_mean_loss),
        })
        print(f"  run {run_no} (seed {seed}): acc={r.accuracy_percent:.2f}%  train={r.train_seconds:.2f}s  "
              f"inference={r.inference_seconds:.3f}s  peak GPU memory {peak_alloc:.2f} GiB allocated / "
              f"{peak_reserved:.2f} GiB reserved", flush=True)

    agg = summarize_runs(runs)
    comparison = []
    for name, base, base_counts in (("q1_frozen", q1, counts_q1), ("q6_full_finetune", q6, counts_q6)):
        comparison += [{"baseline": name, **{k: row[k] for k in BASELINE_COMPARISON_FIELDS}}
                       for row in baseline_comparison(agg, base["aggregate"])]
        comparison.append({"baseline": name, "metric": "trainable_parameters",
                           "baseline_mean": base_counts["trainable"], "baseline_std_sample": "",
                           "new_mean": counts["trainable"], "new_std_sample": "",
                           "difference": counts["trainable"] - base_counts["trainable"],
                           "ratio": counts["trainable"] / base_counts["trainable"]})
    write_rows_csv(args.out / "runs.csv", runs)
    write_rows_csv(args.out / "comparison.csv", comparison)
    config = {
        "question": 7, "model": spec.name, "smoke": args.smoke, "device": "gpu",
        "interpretation": INTERPRETATION,
        "unfrozen_feature_indices": list(PARTIAL_FT_FEATURE_INDICES),
        "unfrozen_layers": layers,
        "rejected_interpretations": ALTERNATIVES,
        "change_vs_question1": "after freezing model.features (as in Q1), requires_grad=True for features[16], "
                               "features[17], features[18]; everything else identical",
        "batchnorm": "model.train() during training, as in the original script: running mean/variance of every "
                     "BatchNorm layer (frozen or not) are updated; frozen layers keep their weights and biases",
        "pretrained_weights": "torchvision MobileNet_V2_Weights.IMAGENET1K_V1 (same as Question 1)",
        "classifier": "classifier[1] replaced by Linear(1280, 10) after seeding (same as Question 1)",
        "preprocessing": PREPROCESSING,
        "hyperparameters": {"epochs": spec.epochs, "batch_size": spec.batch_size,
                            "learning_rate": spec.learning_rate, "optimizer": "Adam (all model parameters; "
                            "frozen ones receive no gradient and are not updated)",
                            "loss": spec.make_loss().__class__.__name__},
        "parameters": {"question7": counts, "question1": counts_q1, "question6": counts_q6},
        "protocol": {"seeds": args.seeds, "warmup_batches": args.warmup_batches, "num_workers": args.num_workers,
                     "max_train_batches_per_epoch": max_train, "max_test_batches": max_test,
                     "mixed_precision": False, "gradient_accumulation": False,
                     "std_note": "std_sample uses ddof=1 (n-1)"},
        "timing_definitions": TIMING_NOTES,
        "environment": {**environment(), "gpu_arch": props.gcnArchName,
                        "gpu_total_memory_gib": round(props.total_memory / 2**30, 2)},
        "baselines": {name: {"path": str(path), "sha256": sha256(path)} for name, path in
                      (("q1_frozen", args.q1 / "mobilenetv2" / "summary.json"),
                       ("q6_full_finetune", args.q6 / "summary.json"))},
    }
    write_json(args.out / "config.json", config)
    write_json(args.out / "summary.json", {
        "question": 7, "model": spec.name, "smoke": args.smoke, "device": "gpu",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "hyperparameters": config["hyperparameters"], "protocol": config["protocol"],
        "unfrozen_layers": layers, "parameters": counts, "aggregate": agg, "comparison": comparison, "runs": runs,
    })
    a = agg["accuracy_percent"]
    print(f"  => accuracy {a['mean']:.2f}% ± {a['std_sample']:.2f} (sample SD, n={a['n']}); "
          f"train {agg['train_seconds']['mean']:.2f}s; inference {agg['inference_seconds']['mean']:.3f}s")
    for c in comparison:
        if c["metric"] in ("accuracy_percent", "train_seconds"):
            unit = "pp" if c["metric"] == "accuracy_percent" else "s"
            print(f"  vs {c['baseline']}: {c['metric']} {c['difference']:+.2f} {unit} (ratio {c['ratio']:.3f})")


if __name__ == "__main__":
    main()
