# Machine Learning – Assignment 2: Neural Networks (MNIST)

## Layout

```
mnist_MLP.py, mnist_CNN.py, mnist_MobileNetV2.py   professor's originals (unchanged)
a2/
  models.py      original architectures + per-model hyperparameters (ModelSpec)
  data.py        original transforms/loaders; IndexedDataset keeps MNIST test indices
  timing.py      SyncTimer: device-synchronized wall-clock timing
  experiment.py  train / evaluate / warm-up, following the original loops
  stats.py       mean, sample & population std
  io_utils.py    CSV/JSON writers, combined summary table
q1_gpu_baseline.py   Question 1 CLI
tests/               unit tests (CPU, synthetic data)
results/             outputs (results/smoke/ = pipeline checks only, not tracked)
docs/                assignment description (PDF)
```

## Setup

Python 3.14. Install torch/torchvision from the PyTorch index for your hardware first,
then the remaining pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
# AMD GPU (ROCm 7.14), as used for the reported results:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/rocm7.14
# or CPU only:
# pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

MNIST is downloaded to `./data` on first run (not tracked). The first MobileNetV2 run downloads
the ImageNet weights to `~/.cache/torch/hub`.

## Question 1: GPU baseline (3 runs per model)

Results and discussion: [`results/question1/SUMMARY.md`](results/question1/SUMMARY.md).

```bash
python q1_gpu_baseline.py --model mlp                # seeds 0 1 2
python q1_gpu_baseline.py --model cnn
python q1_gpu_baseline.py --model mobilenetv2         # slowest (224x224 resize on CPU)
python q1_gpu_baseline.py --model all                 # all three in sequence
python q1_gpu_baseline.py --model all --smoke         # ~30 s pipeline check -> results/smoke/
```

Options: `--seeds 0 1 2`, `--out DIR`, `--warmup-batches 20` (0 disables), `--num-workers 0`
(the original scripts use 0, and changing it changes the timings), `--log-interval 100`, `--overwrite`.
The CLI refuses to replace an existing `summary.json` unless you pass `--overwrite`.

### Outputs (`results/question1/<model>/`)

| file | content |
|---|---|
| `runs.csv` | per run: seed, n_correct, accuracy, train/inference seconds, per-epoch seconds and mean loss |
| `summary.json` | hyperparameters, protocol, environment, timing definitions, aggregates, runs |
| `predictions_run{k}_seed{s}.csv` | `mnist_test_index, true_label, predicted_label, correct` for all 10 000 test images (for Q3) |
| `../summary_all_models.csv` | one row per model: accuracy mean ± sample SD, mean times |

### Measurement protocol

- **Independent runs**: each seed re-initializes the weights (`torch.manual_seed` before building the model)
  and gives the training DataLoader its own shuffle generator. Runs are not bit-identical across
  invocations, because GPU kernels can be non-deterministic.
- **Training time**: all epochs. It includes DataLoader iteration (CPU transforms), host-to-device copies,
  forward/backward/step and the loss print every 100 steps (as in the originals). It excludes dataset
  loading and model construction.
- **Inference time**: one full pass over the 10 000 test images (loading, copy, forward, argmax).
  Predictions stay on the GPU until the timer stops.
- **Synchronization**: `torch.cuda.synchronize()` runs right before the clock is read at the start and at the
  end of each timed region. Without it, asynchronous kernel launches make GPU timings meaningless.
- **Warm-up**: 20 untimed batches on a throw-away model before the runs. Without it, run 1 also pays
  for one-time kernel compilation and tuning (smoke test: CNN run 1 6.3 s vs run 2 0.1 s).
  Use `--warmup-batches 0` to measure it.
- **Std**: `std_sample` (n−1) is the reported SD; `std_population` is also stored.

### Differences from the original scripts (for the report)

Architectures, hyperparameters, losses, optimizer, transforms and loader settings are unchanged.
Only the experiment harness differs:
- Timing, seeding, the repetition loop, warm-up and result files were added.
- The test loader also returns the dataset index of each image.
- MobileNetV2 epoch loss is accumulated on the GPU instead of calling `loss.item()` every step.
  This does not affect the model; it only avoids a per-step sync inside the timed region.
- Spec vs. code: the PDF lists the CNN as conv→pool→ReLU, but the code does conv→ReLU→pool. These are
  mathematically equivalent, and the code is kept. The PDF's MLP description omits the ReLU that the code has.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The 16 unit tests run on CPU with synthetic data (no MNIST download, no GPU). GitHub Actions
(`.github/workflows/python-tests.yml`) runs them with CPU-only PyTorch wheels.
