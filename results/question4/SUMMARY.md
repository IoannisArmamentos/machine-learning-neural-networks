# Question 4: Enhanced MLP

A new MLP with two hidden layers was trained 3 times on the GPU and compared with the original MLP
of Question 1. The Question 1 results were read from `results/question1/mlp/summary.json`; they were
not rerun or modified.

## Architecture and training configuration

| | Original MLP (Question 1) | Enhanced MLP (Question 4) |
|---|---|---|
| Layers | 784 → 100 → ReLU → 10 | 784 → 500 → ReLU → Dropout(0.2) → 200 → ReLU → Dropout(0.2) → 10 |
| Output | log-softmax | log-softmax |
| Parameters | 79,510 | 494,710 (6.2×) |

Training configuration (identical for both, as specified for all networks in the assignment):

- Adam, learning rate 0.001, mini-batch size 100, 5 epochs.
- Loss: cross-entropy as negative log-likelihood (`NLLLoss` on log-softmax outputs), as in the
  original MLP.
- Input: `ToTensor()`, flattened to 784 values, no other preprocessing (as in `mnist_MLP.py`).
- Dropout is active only during training (`model.train()`). It is switched off for evaluation
  (`model.eval()`), and PyTorch rescales the kept activations by 1/(1−p) during training.

## Protocol

Same as Question 1:
- 3 runs with seeds 0, 1, 2 (seed = weight initialisation and shuffle order).
- 20 untimed warm-up batches on a throw-away model, `num_workers=0`.
- The same timing boundaries, with GPU synchronization before every clock read.
- Training time = all 5 epochs, including data loading. Inference time = one pass over the
  10,000 test images.

The runner checks that the seeds, warm-up and worker settings equal those recorded in the
Question 1 baseline, and refuses to run otherwise.

## Results

| Run | Seed | Accuracy | Training (s) | Inference (s) |
|---|---|---|---|---|
| 1 | 0 | 98.17 % | 10.59 | 0.295 |
| 2 | 1 | 97.96 % | 10.80 | 0.271 |
| 3 | 2 | 97.98 % | 10.05 | 0.260 |
| **Mean ± sample SD** | | **98.04 ± 0.12 %** | **10.48 ± 0.39** | **0.275 ± 0.018** |

## Comparison with the original MLP (Question 1)

| Metric | Original MLP | Enhanced MLP | Difference | Ratio |
|---|---|---|---|---|
| Accuracy | 96.97 ± 0.06 % | 98.04 ± 0.12 % | **+1.07 percentage points** | 1.011 |
| Training time | 9.66 ± 0.48 s | 10.48 ± 0.39 s | +0.82 s | 1.085 (8.5 % slower) |
| Inference time | 0.274 ± 0.002 s | 0.275 ± 0.018 s | +0.001 s | 1.004 (no practical difference) |

Difference = enhanced − original (accuracy in percentage points, times in seconds); ratio =
enhanced / original.

- **Accuracy improves by about 1.1 percentage points.** Errors drop from about 303 to about 196
  per 10,000 test images, a 35 % reduction. Every enhanced run (97.96–98.17 %) is above every
  original run (96.91–97.02 %). The gap (1.07 pp) is about nine times the larger of the two standard
  deviations. A formal significance test is left for the summary comparison in Question 8.
- **Likely reasons:** the wider and deeper network has 6.2 times more parameters and can represent
  more complex decision boundaries. Dropout regularises it, which counteracts the extra risk of
  overfitting. This experiment changes depth, width and dropout together, so it does not separate
  their individual effects (Question 5 varies dropout and width for the one-hidden-layer MLP).
- **Time barely changes.** The network has 6.2 times more parameters, yet training takes only 8.5 %
  longer and inference is unchanged. In Question 1, data loading on the CPU (`num_workers=0`, as in
  the original scripts) accounted for about 1.3 s of each ~1.9 s epoch. The GPU computation for a
  network of this size is a small part of the measured time.
- The mean training loss of the final epoch is also lower (0.051–0.054 vs 0.089–0.095). This is
  only indicative, because the enhanced MLP's training loss is measured with dropout active.

## Limitations

- **3 runs per model**, so the standard deviations are themselves rough estimates.
- **Different sessions:** the Question 1 baseline was measured in an earlier session on the same
  day and machine. Timing differences of a few percent are within session-to-session variation.
  The Q1 training-time SD alone is 0.48 s.
- **The time comparison reflects this pipeline.** Data loading dominates. With a faster input
  pipeline the relative cost of the larger network would be higher.
- Only the test set is used, with no validation set, following the assignment and Question 1.
  Hyperparameters were not tuned.

## Generated files

| Path | Content |
|---|---|
| `results/question4/runs.csv` | per run: seed, correct, accuracy, training/inference time, per-epoch time and mean loss |
| `results/question4/comparison_vs_q1_mlp.csv` | per metric: Q1 and Q4 mean and sample SD, difference, ratio |
| `results/question4/summary.json` | architecture, parameter count, hyperparameters, protocol, environment, timing definitions, aggregates, comparison, and SHA-256 of the Q1 baseline file |
| `results/question4/predictions_run{1,2,3}_seed{0,1,2}.csv` | per-image predictions with MNIST test index, for verification and later paired comparisons (Question 8) |
| `results/question4/logs/q4_gpu_run.log` | console log of the run |

## Reproduce

```bash
source .venv/bin/activate
python q4_enhanced_mlp.py --overwrite      # ~40 s on the GPU
python -m unittest discover -s tests -t . -v
```
