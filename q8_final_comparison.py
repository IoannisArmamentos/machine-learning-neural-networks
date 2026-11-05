"""Assignment 2 - Question 8: consolidated comparison of all experiments (Questions 1-7) and an
exact McNemar test between the two most accurate configurations.

Reads saved result files only: no model is trained and no inference is run.

Example (from the project root, with the venv active):
    python q8_final_comparison.py            # a few seconds
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from torchvision import datasets

from a2.compare import sha256
from a2.figures import bar_panels, dot_plot
from a2.final_comparison import (ConfigResult, ConfigSpec, load_config, mcnemar_report, pareto_front, rank,
                                 read_rows)
from a2.io_utils import write_json, write_rows_csv
from a2.models import CNNet, NeuralNet

ALPHA = 0.05
PRIMARY_SEED = 0
SELECTION_RULE = ("The two configurations with the highest mean test accuracy over their 3 GPU runs (ties: fewer "
                  "trainable parameters, then fixed configuration order) are compared with an exact two-sided "
                  "McNemar test on their seed-0 runs (run 1 of every configuration), paired by original MNIST "
                  "test index. Seeds 1 and 2 are reported separately as a sensitivity check; runs are never "
                  "pooled. Significance level alpha = 0.05. The rule was fixed before computing the test.")

CONFIGS = (
    ConfigSpec("q1_mlp", "Q1 MLP (original)", 1, "MLP 784-100-10, ReLU; original script",
               "results/question1/mlp/runs.csv", "results/question1/mlp/predictions_run{run}_seed{seed}.csv",
               "results/question1/mlp/summary.json", 1),
    ConfigSpec("q1_cnn", "Q1 CNN (original)", 1, "CNN: Conv 32, Conv 64 (3x3), max-pool, Linear 1600-10; original script",
               "results/question1/cnn/runs.csv", "results/question1/cnn/predictions_run{run}_seed{seed}.csv",
               "results/question1/cnn/summary.json", 2),
    ConfigSpec("q1_mobilenetv2_frozen", "Q1 MobileNetV2, frozen", 1,
               "ImageNet MobileNetV2, frozen features, new 10-class classifier trained; original script",
               "results/question1/mobilenetv2/runs.csv",
               "results/question1/mobilenetv2/predictions_run{run}_seed{seed}.csv",
               "results/question1/mobilenetv2/summary.json", 3),
    ConfigSpec("q4_enhanced_mlp", "Q4 MLP 500-200, dropout", 4,
               "MLP 784-500-200-10, ReLU + Dropout(0.2) after each hidden layer",
               "results/question4/runs.csv", "results/question4/predictions_run{run}_seed{seed}.csv",
               "results/question4/summary.json", 4),
    ConfigSpec("q5_tuned_mlp", "Q5 MLP 800, dropout 0.2", 5,
               "MLP 784-800-10, ReLU + Dropout(0.2); hidden size and dropout chosen on a validation set",
               "results/question5/final_runs.csv", "results/question5/predictions_final_run{run}_seed{seed}.csv",
               "results/question5/summary.json", 5),
    ConfigSpec("q6_mobilenetv2_full", "Q6 MobileNetV2, full FT", 6,
               "ImageNet MobileNetV2, all layers fine-tuned",
               "results/question6/runs.csv", "results/question6/predictions_run{run}_seed{seed}.csv",
               "results/question6/summary.json", 6),
    ConfigSpec("q7_mobilenetv2_last3", "Q7 MobileNetV2, last 3 FT", 7,
               "ImageNet MobileNetV2, features[16..18] + classifier fine-tuned",
               "results/question7/runs.csv", "results/question7/predictions_run{run}_seed{seed}.csv",
               "results/question7/summary.json", 7),
)


def parameter_counts(root: Path, spec: ConfigSpec, summary: dict) -> tuple[int, int]:
    """(trainable, total). Saved counts where Q4-Q7 recorded them; Q1 counts from the model definitions."""
    count = lambda m, trainable: sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable)
    if spec.key == "q1_mlp":
        m = NeuralNet()
        return count(m, True), count(m, False)
    if spec.key == "q1_cnn":
        m = CNNet()
        return count(m, True), count(m, False)
    if spec.key == "q1_mobilenetv2_frozen":
        from q6_mobilenet_full_finetune import q1_parameter_counts     # builds without downloading weights
        c = q1_parameter_counts()
        return c["trainable"], c["total"]
    if spec.key in ("q4_enhanced_mlp", "q5_tuned_mlp"):
        n = summary["architecture"]["parameters"]
        return n, n
    return summary["parameters"]["trainable"], summary["parameters"]["total"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("."), help="project root containing results/")
    p.add_argument("--out", type=Path, default=Path("results/question8"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.out.resolve() != (args.root / "results/question8").resolve() and "question8" not in args.out.parts:
        p.error("--out must be a question8 directory (previous results are read-only)")
    return args


def q5_selection_check(root: Path) -> dict:
    """Question 5 chose its configuration on validation data only (re-checked from saved files)."""
    grid = read_rows(root / "results/question5/validation_grid.csv")
    best = json.loads((root / "results/question5/best_config.json").read_text())
    log = (root / "results/question5/logs/q5_run.log").read_text()
    winner = max(grid, key=lambda r: (int(r["val_n_correct"]), -int(r["hidden_size"]), -float(r["dropout"])))
    ok = {
        "grid_has_25_configs": len(grid) == 25,
        "grid_has_no_test_columns": not any(k.lower().startswith("test") for k in grid[0]),
        "validation_size_6000_train_54000": all(r["n_val"] == "6000" and r["n_train"] == "54000" for r in grid),
        "winner_is_validation_argmax": (int(winner["hidden_size"]), float(winner["dropout"])) ==
                                       (best["hidden_size"], best["dropout"]),
        "selection_logged_before_test_phase": log.index("=> best:") < log.index("=== Phase 2") < log.index("test acc="),
    }
    return {"passed": all(ok.values()), **ok}


def main(argv=None) -> None:
    args = parse_args(argv)
    root = args.root
    if (args.out / "model_comparison.csv").exists() and not args.overwrite:
        sys.exit(f"{args.out}/model_comparison.csv exists; pass --overwrite to replace it.")
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    labels = datasets.MNIST(str(root / "data"), train=False, download=True).targets.tolist()   # reference labels

    inputs, results = {}, []
    for spec in CONFIGS:
        summary = json.loads((root / spec.summary_json).read_text())
        if summary.get("smoke") or summary.get("device") != "gpu":
            sys.exit(f"{spec.summary_json}: not a full GPU result")
        runs, preds = load_config(root, spec, true_labels=labels)
        trainable, total = parameter_counts(root, spec, summary)
        results.append(ConfigResult(spec, runs, trainable, total, summary.get("created", ""), preds))
        for rel in [spec.runs_csv, spec.summary_json] + [
                spec.prediction_pattern.format(run=int(r["run"]), seed=int(r["seed"])) for r in runs]:
            inputs[rel] = sha256(root / rel)

    ranked = rank(results)
    front = pareto_front([(r.spec.key, r.mean("accuracy_percent"), r.mean("train_seconds")) for r in results])
    rows = []
    for pos, r in enumerate(ranked, start=1):
        rows.append({
            "rank": pos, "config": r.spec.key, "label": r.spec.label, "question": r.spec.question,
            "description": r.spec.description, "n_runs": len(r.runs),
            "accuracy_mean": r.mean("accuracy_percent"), "accuracy_std_sample": r.sd("accuracy_percent"),
            "accuracy_min": min(float(x["accuracy_percent"]) for x in r.runs),
            "accuracy_max": max(float(x["accuracy_percent"]) for x in r.runs),
            "errors_mean": 10_000 - r.total_correct / len(r.runs),
            "gap_to_best_pp": r.mean("accuracy_percent") - ranked[0].mean("accuracy_percent"),
            "train_seconds_mean": r.mean("train_seconds"), "train_seconds_std_sample": r.sd("train_seconds"),
            "inference_seconds_mean": r.mean("inference_seconds"),
            "inference_seconds_std_sample": r.sd("inference_seconds"),
            "trainable_parameters": r.trainable_parameters, "total_parameters": r.total_parameters,
            "accuracy_time_pareto_optimal": int(r.spec.key in front), "measured": r.measured,
        })
    write_rows_csv(args.out / "model_comparison.csv", rows)

    # ---- statistical test (rule fixed above, applied to the two highest-ranked configurations)
    a, b = ranked[0], ranked[1]
    primary = mcnemar_report(a.predictions[PRIMARY_SEED], b.predictions[PRIMARY_SEED], ALPHA)
    sensitivity = {f"seed_{s}": mcnemar_report(a.predictions[s], b.predictions[s], ALPHA) for s in (1, 2)}
    stat = {
        "test": "exact two-sided McNemar test (binomial test on discordant pairs, p = 0.5)",
        "selection_rule": SELECTION_RULE,
        "model_a": {"config": a.spec.key, "label": a.spec.label, "mean_accuracy_percent": a.mean("accuracy_percent")},
        "model_b": {"config": b.spec.key, "label": b.spec.label, "mean_accuracy_percent": b.mean("accuracy_percent")},
        "primary_seed": PRIMARY_SEED,
        "primary": primary,
        "sensitivity_other_seeds_not_pooled": sensitivity,
        "contingency_definition": {"both_correct": "A and B correct", "a_only_correct": "A correct, B wrong",
                                   "b_only_correct": "A wrong, B correct", "both_incorrect": "A and B wrong"},
        "interpretation_note": ("McNemar tests whether, on these 10,000 test images, the two trained models' error "
                                "rates differ (H0: discordant pairs split 50/50). A non-significant result means no "
                                "difference was detected; it does not establish equivalence. The test concerns these "
                                "two trained models (one seed each), not all possible training runs."),
    }
    write_json(args.out / "statistical_test.json", stat)

    # ---- GPU vs CPU (Question 2, read-only)
    q2 = read_rows(root / "results/question2/cpu_gpu_comparison.csv")
    inputs["results/question2/cpu_gpu_comparison.csv"] = sha256(root / "results/question2/cpu_gpu_comparison.csv")
    gpu_cpu = [{k: r[k] for k in ("model", "cpu_train_seconds", "gpu_train_seconds_mean", "train_speedup",
                                  "cpu_inference_seconds", "gpu_inference_seconds_mean", "inference_speedup")}
               for r in q2]

    by_key = {r.spec.key: r for r in results}
    fastest_train = min(results, key=lambda r: (r.mean("train_seconds"), r.spec.order))
    fastest_inf = min(results, key=lambda r: (r.mean("inference_seconds"), r.spec.order))
    q5_check = q5_selection_check(root)
    for rel in ("results/question5/validation_grid.csv", "results/question5/best_config.json",
                "results/question5/logs/q5_run.log"):
        inputs[rel] = sha256(root / rel)
    write_json(args.out / "summary.json", {
        "question": 8, "created": dt.datetime.now().isoformat(timespec="seconds"),
        "ranking": [r["config"] for r in rows],
        "highest_mean_accuracy": ranked[0].spec.key, "second_highest_mean_accuracy": ranked[1].spec.key,
        "fastest_training": fastest_train.spec.key, "fastest_inference": fastest_inf.spec.key,
        "accuracy_training_time_pareto_front": front,
        "mcnemar_primary_p_value": primary["p_value_exact_two_sided"],
        "gpu_vs_cpu_question2": gpu_cpu,
        "question5_validation_only_selection": q5_check,
    })
    write_json(args.out / "config.json", {
        "question": 8, "alpha": ALPHA, "primary_seed": PRIMARY_SEED, "selection_rule": SELECTION_RULE,
        "ranking_rule": "mean test accuracy over 3 GPU runs (exact total correct), then fewer trainable "
                        "parameters, then fixed configuration order (Q1 MLP, Q1 CNN, Q1 MobileNetV2, Q4, Q5, Q6, Q7)",
        "pareto_rule": "not dominated by any configuration with higher-or-equal mean accuracy and lower-or-equal "
                       "mean training time (at least one strictly better)",
        "no_training_or_inference": True,
        "reference_labels": "torchvision MNIST test targets (data/MNIST)",
        "configurations": [{"key": s.key, "label": s.label, "question": s.question, "description": s.description,
                            "runs_csv": s.runs_csv} for s in CONFIGS],
        "input_sha256": inputs,
    })

    # ---- figures
    emph = {a.spec.key, b.spec.key}
    dot_plot([(r.spec.label, r.mean("accuracy_percent"), r.sd("accuracy_percent"), r.spec.key in emph) for r in ranked],
             "Test accuracy of all configurations (3 GPU runs each)",
             "Dot = mean, whiskers = ±1 sample SD. Blue = the two most accurate, compared with McNemar.",
             "Test accuracy (%)", 95.5, 99.5).save(args.out / "figures/accuracy_comparison.png", optimize=True)
    bar_panels([r.spec.label for r in ranked],
               [("Mean GPU training time (s)", [r.mean("train_seconds") for r in ranked], "{:.1f}"),
                ("Mean GPU test inference time (s)", [r.mean("inference_seconds") for r in ranked], "{:.2f}")],
               "GPU training and inference time (mean of 3 runs)",
               "Same order as the accuracy ranking. Each panel has its own axis starting at zero.",
               ).save(args.out / "figures/timing_comparison.png", optimize=True)

    for r in rows:
        print(f"{r['rank']}. {r['label']:28s} {r['accuracy_mean']:.2f} ± {r['accuracy_std_sample']:.2f} %  "
              f"train {r['train_seconds_mean']:7.2f}s  inference {r['inference_seconds_mean']:.3f}s  "
              f"trainable {r['trainable_parameters']:>9,}{'  [pareto]' if r['accuracy_time_pareto_optimal'] else ''}")
    t = primary["contingency"]
    print(f"McNemar seed {PRIMARY_SEED}: {a.spec.key} vs {b.spec.key}: both correct {t['both_correct']}, "
          f"A only {t['a_only_correct']}, B only {t['b_only_correct']}, both wrong {t['both_incorrect']}; "
          f"p = {primary['p_value_exact_two_sided']:.4f} ({'significant' if primary['significant'] else 'not significant'} at alpha {ALPHA})")
    for s, rep in sensitivity.items():
        print(f"  sensitivity {s}: A only {rep['a_only_correct']}, B only {rep['b_only_correct']}, p = {rep['p_value_exact_two_sided']:.4f}")
    print(f"Q5 validation-only selection re-check: {'passed' if q5_check['passed'] else 'FAILED'}")
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
