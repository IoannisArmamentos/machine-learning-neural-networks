# Question 6: MobileNetV2 with full fine-tuning

The supplied MobileNetV2 code keeps the pretrained ImageNet weights frozen and trains only the
new classifier. For Question 6 the code was changed so that **the whole pretrained network
continues training** (fine-tuning) on the MNIST training set. It was run 3 times on the GPU and
compared with the frozen Question 1 baseline, which was not rerun.

## What changed compared with Question 1

| | Question 1 (`mnist_MobileNetV2.py`) | Question 6 |
|---|---|---|
| `model.features` | `requires_grad = False` (frozen) | `requires_grad = True` (fine-tuned) |
| Classifier | `classifier[1] = Linear(1280, 10)` | same |
| Trainable parameters | 12,810 (classifier only) | **2,236,682 (all; 175×)** |
| Total parameters | 2,236,682 | 2,236,682 |

Everything else is identical, as checked against the original script and the saved Question 1
configuration:

- **Weights:** torchvision `MobileNet_V2_Weights.IMAGENET1K_V1`.
- **Preprocessing:** `Resize((224, 224))` → `Grayscale(num_output_channels=3)` → `ToTensor()` →
  `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **Training:** Adam over `model.parameters()`, learning rate 0.001, batch size 100, 3 epochs,
  `CrossEntropyLoss`, all 60,000 training images. Evaluation on the 10,000 test images.
- **Protocol:** seeds 0, 1, 2; 20 untimed warm-up batches on a throw-away model; `num_workers=0`;
  GPU-synchronized timing with the same boundaries as Question 1.
- **Same starting point:** each run builds a fresh pretrained model and replaces the classifier
  after seeding. For the same seed, the initial weights (pretrained features and classifier
  initialisation) are identical to the Question 1 model; this was checked for all three seeds.
- **No other changes:** no mixed precision, gradient accumulation, other image size or caching.
  Device: AMD Radeon RX 9070 XT (`cuda:0`, gfx1201, 15.9 GiB).

## Results

| Run | Seed | Accuracy | Training (s) | Inference (s) | Peak GPU memory |
|---|---|---|---|---|---|
| 1 | 0 | 98.81 % | 601.39 | 6.307 | 7.39 GiB allocated / 8.22 GiB reserved |
| 2 | 1 | 99.23 % | 603.96 | 6.492 | 7.39 / 8.22 GiB |
| 3 | 2 | 99.05 % | 604.83 | 6.474 | 7.39 / 8.22 GiB |
| **Mean ± sample SD** | | **99.03 ± 0.21 %** | **603.39 ± 1.79** | **6.424 ± 0.102** | |

## Comparison with the frozen Question 1 MobileNetV2

| Metric | Q1 frozen | Q6 full fine-tuning | Difference | Ratio |
|---|---|---|---|---|
| Accuracy | 95.99 ± 0.18 % | 99.03 ± 0.21 % | **+3.04 percentage points** | 1.032 |
| Training time | 120.36 s | 603.39 s | +483.03 s | **5.01×** |
| Inference time | 6.388 s | 6.424 s | +0.036 s | 1.006 (unchanged) |
| Trainable parameters | 12,810 | 2,236,682 | +2,223,872 | 174.6× |

- **Accuracy improves substantially.** Test errors drop from about 401 to about 97 per 10,000
  images (−76 %). Every fine-tuned run (98.81–99.23 %) is above every frozen run (95.78–96.12 %).
  The mean training loss of the final epoch falls from about 0.18 to about 0.03.
- **Why:** ImageNet features were learned on colour photographs of objects, not on centred
  handwritten strokes. With frozen features, only a linear classifier has to separate the digits
  using those fixed representations. Fine-tuning lets every convolutional layer adapt to stroke
  shapes. This turns MobileNetV2 from the weakest Question 1 model (below the small MLP) into the
  most accurate so far, slightly above the Question 1 CNN (98.89 ± 0.09 %). Whether that last
  difference is statistically significant is left for Question 8.
- **Training is 5× slower.** Each epoch takes about 200 s instead of about 40 s, because the
  backward pass and the Adam updates now run through all 2.2 million parameters, not only the
  classifier. In Question 1, CPU-side image resizing dominated MobileNetV2's epoch (about 32 of
  40 s). Here GPU computation is the bottleneck (about 0.33 s per batch in the smoke test).
- **Inference is unchanged** (6.42 vs 6.39 s). The forward pass is the same network and the same
  amount of computation; only the weight values differ.
- **GPU memory:** peak 7.39 GiB allocated / 8.22 GiB reserved of 15.9 GiB, the same in every run.
  The original batch size of 100 fits without any change to the protocol. The very first
  fine-tuning step took 18.6 s (one-time kernel tuning for the new backward passes) and was
  absorbed by the untimed warm-up.

## Validation

- **Runs:** 3 runs completed (exit code 0; all 54 training-step lines logged; no errors or NaN).
- **Predictions:** each run has exactly 10,000 test predictions with indices 0–9999 and labels
  matching MNIST. The accuracies were recounted from the prediction files.
- **Comparison:** recomputed from the Question 1 and Question 6 `runs.csv` files. The SHA-256 of
  the Question 1 baseline file, recorded in `config.json`, still matches.
- **Model checks:** a GPU smoke test at batch size 100 confirmed that every parameter receives a
  gradient and that Adam updates all 158 parameter tensors, including the feature layers.
- **Integrity:** the three original scripts are byte-identical, and the Question 1–5 results are
  unchanged.

## Limitations

- **3 runs only**, so the standard deviations are rough estimates.
- **3 epochs with learning rate 0.001**, as in the original code. This is a fairly high learning
  rate for fine-tuning a pretrained network. A lower rate or more epochs might change the result,
  but would no longer match the assignment's configuration.
- **BatchNorm statistics are updated during training in both questions** (`model.train()`), as in
  the original script. In Question 1 this also happens for the frozen feature layers, so even the
  "frozen" baseline is not completely fixed.
- **Different sessions:** the Question 1 baseline was measured on 2026-11-04 and Question 6 on
  2026-11-05, on the same machine. Small timing differences, e.g. in inference, are within
  session-to-session variation.

## Generated files

| Path | Content |
|---|---|
| `results/question6/runs.csv` | per run: seed, correct, accuracy, training/inference time, trainable/total parameters, peak GPU memory, per-epoch time and mean loss |
| `results/question6/comparison_vs_q1_mobilenet.csv` | per metric (accuracy, training time, inference time, trainable parameters): Q1 and Q6 values, difference, ratio |
| `results/question6/summary.json` | hyperparameters, protocol, parameter counts, aggregates, comparison, runs |
| `results/question6/config.json` | change vs Q1, weights, preprocessing, hyperparameters, parameter counts (Q1 and Q6), protocol, environment and GPU, SHA-256 of the Q1 baseline |
| `results/question6/predictions_run{1,2,3}_seed{0,1,2}.csv` | per-image test predictions with MNIST index (for Question 8) |
| `results/question6/logs/q6_gpu_run.log` | console log |

## Reproduce

```bash
source .venv/bin/activate
python q6_mobilenet_full_finetune.py --overwrite     # ~31 min on the GPU (3 runs)
python -m unittest discover -s tests -t . -v
```
