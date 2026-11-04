# Question 2: CPU vs GPU

Each of the three original networks was trained and tested **once on the CPU**, with the original
hyperparameters (batch 100, lr 0.001, Adam; 5 epochs for MLP/CNN, 3 for MobileNetV2). The times
are compared with the Question 1 GPU results, which were not rerun.

**Environment:** AMD Ryzen 9 7900X (12 cores / 24 threads; PyTorch intra-op threads = 12, the
default), PyTorch 2.15.0+rocm7.14, Python 3.14.4, Kubuntu 26.04. The GPU (Radeon RX 9070 XT) was
hidden from PyTorch for this run. CPU run: 2026-11-04, 21:06–21:56.

## Results

**Speedup** = CPU time / GPU time (> 1: GPU faster), computed separately for training and
inference. GPU times are the mean of the 3 Question 1 runs.

| Model | CPU training (s) | GPU training (s) | **Training speedup** | CPU inference (s) | GPU inference (s) | **Inference speedup** |
|---|---|---|---|---|---|---|
| MLP | 8.58 | 9.66 | **0.89×** (CPU faster) | 0.243 | 0.274 | **0.89×** (CPU faster) |
| CNN | 32.59 | 9.54 | **3.42×** | 0.424 | 0.257 | **1.65×** |
| MobileNetV2 | 2745.10 (45 min 45 s) | 120.36 | **22.81×** | 143.71 | 6.388 | **22.50×** |

Inference time is for the whole 10 000-image test set.

Speedups against the single GPU run with the same seed (seed 0, Q1 run 1): MLP 0.89× / 0.88×,
CNN 3.45× / 1.59×, MobileNetV2 23.19× / 22.32× (training / inference). These are within a few
percent of the mean-based values.

### Accuracy (sanity check)

| Model | CPU (seed 0) | GPU, same seed | GPU mean ± SD (3 runs) | Identical predictions, CPU vs GPU seed 0 |
|---|---|---|---|---|
| MLP | 96.98 % | 96.98 % | 96.97 ± 0.06 % | 10 000 / 10 000 |
| CNN | 98.98 % | 99.00 % | 98.89 ± 0.09 % | 9 990 / 10 000 |
| MobileNetV2 | 95.94 % | 96.06 % | 95.99 ± 0.18 % | 9 914 / 10 000 |

The device changes speed, not what the model learns. With the same seed, CPU and GPU produce the
same or almost the same predictions. The small differences come from floating-point differences
between CPU and GPU kernels, and are within the GPU run-to-run variation.

## Discussion

- **MLP: the GPU brings no benefit (0.89×).** The network (~80 k parameters, batches of 100) is
  far too small to keep a GPU busy. Most of the time goes to CPU-side data loading, which takes
  ~1.3 s of each ~1.7–1.9 s epoch (measured in Question 1). The GPU then adds per-batch
  host→device copies and kernel-launch overhead that the CPU does not pay.
- **CNN: the GPU helps moderately (3.4× training, 1.65× inference).** The convolutions are enough
  work for the GPU to win, but the shared data-loading cost caps the speedup. Inference gains less
  because it has no backward pass, so loading takes a larger share of the time.
- **MobileNetV2: the GPU is essential (~23× for both).** Each 224×224 RGB image needs ~0.3 G multiply-adds
  per forward pass through a 3.5 M-parameter network. On the CPU one epoch takes ~15 min, compared
  with ~40 s on the GPU.
- **The measured speedups are lower bounds on the GPU's compute advantage.** Both devices use
  identical CPU-side data loading (`num_workers=0`, as in the original scripts), and it is inside
  the timed region. For MobileNetV2, the loader alone takes ~32 s of each ~40 s GPU epoch
  (Question 1 loader-only measurement), so the GPU's pure compute time is much shorter than the
  totals show. This is a rough estimate from a separate measurement, not a timed result.

## Methodology

- **Device:** `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set to
  empty before `torch` is imported, and the script exits if `torch.cuda.is_available()` is still
  true. The model and all tensors use `device=cpu`. Each `cpu_run.json` records
  `gpu_visible_to_pytorch: false`.
- **Same pipeline as Question 1:** the same `a2` code (`warmup` → `run_once` → `train`/`evaluate`)
  with identical timing boundaries. Training time = all epochs, including DataLoader iteration and
  the loss print every 100 steps. Inference time = one full pass over the test set. Both exclude
  dataset loading and model construction.
- **Synchronization:** on the GPU, the device is synchronized before each clock read. On the CPU
  that call does nothing, because CPU operators finish before returning.
- **Warm-up (the same on both devices: 20 untimed batches on a throw-away model before seeding):**
  - On the GPU, it absorbs one-time kernel compilation and autotuning and device set-up, which
    would otherwise inflate run 1 (CNN 6.3 s vs 0.1 s in a smoke test).
  - On the CPU there is no kernel compilation, so the warm-up only absorbs small first-call costs
    (memory allocation, oneDNN set-up, thread-pool start-up).
  - Using the same warm-up on both devices keeps the comparison symmetric. It is not an extra
    measured run, so each model is still run once on the CPU, as the assignment asks.
- **Seed:** 0, the same seed as GPU run 1, so the same-seed comparison above is possible.
- **Baseline integrity:** the GPU numbers are read from `results/question1/<model>/summary.json`.
  Their SHA-256 hashes are recorded in `config.json` and were unchanged after the CPU runs.

## Limitations

- **Single CPU run per model** (as the assignment specifies), so there is no CPU variance estimate.
  Q1 run-to-run SDs of the GPU training times were 0.2–1.8 s.
- **Not a dedicated benchmark machine.** The desktop session (browser, IDE) kept running during
  the CPU runs. CPU frequency boost and temperature were not controlled, and the GPU baseline was
  measured in an earlier session the same day.
- **CPU parallelism:** PyTorch used its default 12 intra-op threads, one per physical core. With
  `num_workers=0`, data loading runs in the main process and alternates with compute instead of
  overlapping it. Other thread or worker settings would change the absolute times.
- **Results are specific to this hardware.** A 12-core desktop CPU is much faster than the CPUs of
  free Colab/Kaggle instances, which is why MobileNetV2 took ~46 min here rather than the "several
  hours" mentioned in the assignment.

## Generated files

| Path | Content |
|---|---|
| `results/question2/cpu_runs.csv` | one row per model: accuracy, train/inference seconds, per-epoch seconds and loss, threads, warm-up |
| `results/question2/cpu_gpu_comparison.csv` | CPU vs GPU accuracy and times; training and inference speedups (vs GPU mean and same-seed run) |
| `results/question2/config.json` | speedup definition, timing and warm-up methodology, per-model protocol, environment, GPU baseline paths and SHA-256 |
| `results/question2/<model>/cpu_run.json` | full per-model record |
| `results/question2/<model>/predictions_cpu_seed0.csv` | `mnist_test_index, true_label, predicted_label, correct` for all 10 000 test images |
| `results/question2/logs/q2_cpu_run.log` | console log of the run |

## Reproduce

```bash
source .venv/bin/activate
python q2_cpu_comparison.py --model all --overwrite   # ~50 min on the CPU above
python -m unittest discover -s tests -t . -v
```
