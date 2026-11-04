# Question 1: GPU baseline (3 runs per model)

Each of the three supplied networks was trained and tested three times on the GPU, using the
original architectures and hyperparameters (batch 100, lr 0.001, Adam; 5 epochs for MLP/CNN,
3 for MobileNetV2).

**Environment:** AMD Radeon RX 9070 XT (gfx1201), ROCm/HIP 7.14.60850, PyTorch 2.15.0+rocm7.14,
torchvision 0.29.2, Python 3.14.4, AMD Ryzen 9 7900X, Kubuntu 26.04. Run date: 2026-11-04.

## Results

| Model | Run 1 (seed 0) | Run 2 (seed 1) | Run 3 (seed 2) | **Accuracy mean ± SD** | Training time (s) | Inference time, 10 000 images (s) |
|---|---|---|---|---|---|---|
| MLP | 96.98 % | 96.91 % | 97.02 % | **96.97 ± 0.06 %** | 9.66 ± 0.48 | 0.274 ± 0.002 |
| CNN | 99.00 % | 98.84 % | 98.84 % | **98.89 ± 0.09 %** | 9.54 ± 0.18 | 0.257 ± 0.009 |
| MobileNetV2 (frozen features) | 96.06 % | 96.12 % | 95.78 % | **95.99 ± 0.18 %** | 120.36 ± 1.75 | 6.388 ± 0.079 |

SD is the sample standard deviation (n − 1, n = 3). Times are means over the 3 runs.

Per-run times (training / inference, s):

| Model | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| MLP | 9.65 / 0.275 | 10.15 / 0.275 | 9.18 / 0.272 |
| CNN | 9.44 / 0.266 | 9.43 / 0.256 | 9.75 / 0.248 |
| MobileNetV2 | 118.36 / 6.439 | 121.06 / 6.298 | 121.65 / 6.428 |

## Observations

- **Accuracy:** CNN > MLP > MobileNetV2. The run-to-run spread is small (SD ≤ 0.18 pp).
  With only the classifier trained on frozen ImageNet features, MobileNetV2 does slightly worse
  than the small MLP. Questions 6–7 (fine-tuning) address this.
- **Training time:** MLP and CNN take almost the same time (~9.5 s for 5 epochs), even though the
  CNN does far more computation. Iterating the training DataLoader alone, with no model, takes
  ~1.3 s per epoch, about 70 % of the ~1.9 s MLP/CNN epoch. The bottleneck is CPU-side data
  loading (PIL → tensor, `num_workers=0`, as in the original scripts), not the GPU.
- **MobileNetV2** is ~12× slower to train and ~25× slower at inference. Most of that is again the
  CPU: the 224×224 resize + grayscale→RGB + normalize pipeline takes ~32 s of each ~40 s epoch when
  run without any model. This matters for interpreting the GPU vs CPU comparison in Question 2.

## Methodology

- **Independent runs:** seeds 0, 1, 2. Each seed sets the weight initialization (`torch.manual_seed`
  before the model is built) and the training-set shuffle order (its own DataLoader generator).
  Runs are not guaranteed to be bit-identical when repeated, because GPU kernels can be
  non-deterministic.
- **Training time:** wall-clock time over all epochs. It includes DataLoader iteration, host→device
  copies, forward/backward/optimizer steps and the loss print every 100 steps. It excludes dataset
  loading and model construction.
- **Inference time:** one full pass over the 10 000 test images (loading, copy, forward, argmax).
  Predictions stay on the GPU until the timer stops; accuracy is computed afterwards.
- **GPU synchronization:** `torch.cuda.synchronize()` runs immediately before each clock read, at the
  start and end of every timed region, so asynchronous kernels are fully counted.
- **Warm-up:** 20 untimed batches on a throw-away model before the timed runs. Without it, the first
  run also pays for one-time kernel compilation and tuning (in a pipeline check, CNN run 1 took
  6.3 s vs 0.1 s for the same work in run 2).
- **Code changes vs the originals:** none to architectures, hyperparameters, losses, optimizer,
  transforms or loader settings. Only the harness differs: timing, seeding and repetition were
  added, the test loader also returns each image's dataset index, and MobileNetV2's epoch loss is
  accumulated on the GPU instead of calling `.item()` every step. See `README.md`.
- **Spec vs code:** the PDF lists the CNN as conv→pool→ReLU, but the code does conv→ReLU→pool.
  These are mathematically equivalent, and the code is kept unchanged.

## Generated files

| Path | Content |
|---|---|
| `results/question1/summary_all_models.csv` | the results table above (one row per model) |
| `results/question1/<model>/runs.csv` | per run: seed, correct, accuracy, train/inference time, per-epoch time and mean loss |
| `results/question1/<model>/summary.json` | hyperparameters, protocol, environment, timing definitions, aggregates |
| `results/question1/<model>/predictions_run{1,2,3}_seed{0,1,2}.csv` | `mnist_test_index, true_label, predicted_label, correct` for all 10 000 test images (input for Question 3) |
| `results/question1/logs/q1_gpu_run.log` | full console log of the run |

`<model>` is `mlp`, `cnn` or `mobilenetv2`.

## Reproduce

```bash
source .venv/bin/activate
python q1_gpu_baseline.py --model all --overwrite     # ~7 min on the setup above
python -m unittest discover -s tests -t . -v
```
