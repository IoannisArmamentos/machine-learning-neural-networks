# Question 7: MobileNetV2 with partial fine-tuning (last 3 layers)

The supplied MobileNetV2 code was changed so that **only the last 3 layers of the pretrained
network** continue training (fine-tuning) on the MNIST training set, together with the new
10-class classifier. It was run 3 times on the GPU and compared with Question 1 (frozen features,
classifier only) and Question 6 (full fine-tuning). Neither of those was rerun.

## Which layers are "the last 3 layers"

torchvision's `model.features` is a `Sequential` of 19 top-level modules:

- `features[0]`: the first 3×3 convolution.
- `features[1]`–`features[17]`: 17 `InvertedResidual` bottleneck blocks.
- `features[18]`: the final 1×1 convolution with BatchNorm and ReLU6.

Each top-level module is one layer unit of the pretrained network, so the last three are:

| Module | Type | Shape | Parameters | Trainable |
|---|---|---|---|---|
| `features[16]` | `InvertedResidual` (expand 1×1, depthwise 3×3, project 1×1, each with BatchNorm) | 160 → 160 | 320,000 | yes |
| `features[17]` | `InvertedResidual` | 160 → 320 | 473,920 | yes |
| `features[18]` | `Conv2dNormActivation` (Conv 1×1, BatchNorm, ReLU6) | 320 → 1280 | 412,160 | yes |
| `classifier[1]` | `Linear` (new 10-class head, as in Question 1) | 1280 → 10 | 12,810 | yes |
| `features[0]`–`features[15]` | | | 1,017,792 | **no** (frozen) |

**Trainable: 1,218,890 of 2,236,682 parameters (54.5 %).**

This reading includes the true final layer of the network and keeps each bottleneck block whole.
It is also the common `model.features[-3:]` convention. Two alternatives were considered and
rejected; both are recorded in `config.json`:

| Alternative | Modules | Pretrained parameters | Rejected because |
|---|---|---|---|
| Last 3 `InvertedResidual` blocks | `features[15]`–`[17]` | 1,113,920 | leaves the actual last layer, `features[18]`, frozen |
| Last 3 parameterised leaf operations | BN `17.conv.3`, Conv `18.0`, BN `18.1` | 412,800 | splits a bottleneck block and counts BatchNorm as a separate layer |

## Everything else is identical to Question 1

- **Model and data:** torchvision `IMAGENET1K_V1` weights, the same preprocessing (224×224,
  grayscale → 3 channels, ImageNet normalisation) and `classifier[1] = Linear(1280, 10)`.
- **Training:** Adam over `model.parameters()`, learning rate 0.001, batch size 100, 3 epochs,
  `CrossEntropyLoss`, all 60,000 training images; evaluation on the 10,000 test images.
- **Protocol:** seeds 0, 1, 2; 20 untimed warm-up batches; `num_workers=0`; GPU-synchronized
  timing. No mixed precision or other changes.
- **Building each model:** each run builds a fresh pretrained model, freezes `model.features` and
  replaces the classifier exactly as in Question 1, then sets `requires_grad=True` for
  `features[16:19]`. For the same seed, the initial weights are identical to the Question 1 model
  (checked for all three seeds).
- **Gradients:** a GPU smoke test at batch size 100 showed that gradients reach exactly the 23
  trainable parameter tensors, and that Adam left every frozen weight exactly unchanged.
- **BatchNorm:** as in the original script, training uses `model.train()`. The running
  mean/variance of every BatchNorm layer therefore keep updating during training, including those
  in frozen blocks; only their weights and biases stay fixed. Question 1 behaves the same way.

## Results (AMD Radeon RX 9070 XT)

| Run | Seed | Accuracy | Training (s) | Inference (s) | Peak GPU memory |
|---|---|---|---|---|---|
| 1 | 0 | 98.74 % | 127.67 | 6.585 | 1.05 GiB allocated / 1.44 GiB reserved |
| 2 | 1 | 99.09 % | 128.15 | 6.384 | 1.05 / 1.44 GiB |
| 3 | 2 | 99.25 % | 124.88 | 6.508 | 1.05 / 1.44 GiB |
| **Mean ± sample SD** | | **99.03 ± 0.26 %** | **126.90 ± 1.76** | **6.492 ± 0.101** | |

## Comparison with Questions 1 and 6

| Variant | Trainable parameters | Accuracy | Training (s) | Inference (s) | Peak GPU memory |
|---|---|---|---|---|---|
| Q1: frozen features, classifier only | 12,810 | 95.99 ± 0.18 % | 120.36 | 6.388 | not recorded |
| **Q7: last 3 layers + classifier** | **1,218,890** | **99.03 ± 0.26 %** | **126.90** | **6.492** | **1.05 GiB** |
| Q6: full fine-tuning | 2,236,682 | 99.03 ± 0.21 % | 603.39 | 6.424 | 7.39 GiB |

| Q7 vs | Accuracy difference | Training time ratio | Inference time ratio | Trainable parameters ratio |
|---|---|---|---|---|
| Q1 frozen | **+3.04 percentage points** | 1.054 | 1.016 | 95.2× |
| Q6 full fine-tuning | −0.003 percentage points | **0.210** (4.8× faster) | 1.011 | 0.545 |

- **Observed mean accuracy is nearly identical to full fine-tuning.** Partial fine-tuning reaches
  99.03 % vs 99.03 % for Question 6, a difference of 0.003 percentage points, far below either
  standard deviation. With 3 runs per model this shows no detectable difference under this
  protocol; it is not a proof of statistical equivalence. Test errors drop from about 401 (Q1) to about 97 (both Q6 and Q7) per
  10,000 images, and every Q7 run is above every Q1 run. The two fine-tuned models are similar
  but not identical: for the same seed they give the same prediction on 98.2–99.0 % of the test
  images.
- **Why:** the early layers of an ImageNet network detect generic edges and textures, which also
  suit handwritten strokes. The last layers combine these into class-specific, high-level
  patterns, and that is where the ImageNet representation fits MNIST worst. Adapting only the last
  blocks (54.5 % of the parameters, at the 7×7 spatial resolution) is therefore enough. Retraining
  the early layers as well (Question 6) gave no observable additional benefit in these runs.
- **Training costs almost nothing extra.** Training takes 126.9 s, only 5.4 % longer than the
  frozen model and 4.8 times faster than full fine-tuning (603.4 s). The backward pass stops at
  `features[16]`: no gradients are computed for the 16 frozen modules, and their high-resolution
  activations need not be stored. As in Question 1, CPU-side image resizing (about 32 of the
  about 42 s per epoch) again dominates the epoch time.
- **Memory:** peak GPU memory is 1.05 GiB, against 7.39 GiB for full fine-tuning, for the same
  reason.
- **Inference is unchanged** (about 6.4–6.5 s for all three variants), because the forward network
  is the same.
- **Trade-off:** of the three variants, partial fine-tuning reached an observed accuracy nearly
  identical to full fine-tuning at almost the cost of the frozen model.

## Validation

- **Runs:** 3 runs completed (exit code 0; all 54 training-step lines logged; no errors or NaN).
- **Predictions:** each run has exactly 10,000 test predictions with indices 0–9999 and labels
  matching MNIST. The accuracies were recounted from the prediction files.
- **Comparisons:** recomputed from the Question 1 and Question 6 `runs.csv` files. The SHA-256
  hashes of both baseline files, recorded in `config.json`, still match.
- **Integrity:** the three original scripts are byte-identical, and the Question 1–6 results are
  unchanged.

## Limitations

- **"Last 3 layers" has to be interpreted.** The result depends on that reading. Under the
  rejected alternatives (412,800 or 1,113,920 trainable pretrained parameters), the accuracy and
  time would differ; they were not run.
- **BatchNorm statistics in frozen blocks are updated during training**, because of
  `model.train()`, as in the original code. The frozen layers are therefore not completely
  fixed: their normalisation adapts to MNIST. Putting those layers in `eval()` mode would freeze
  them fully, but would deviate from the supplied code.
- **3 runs only, with the original configuration** (learning rate 0.001, 3 epochs). The
  near-identical accuracy of Questions 6 and 7 is an observation for this configuration, not an
  equivalence test, and should not be generalised.
- **Different sessions:** the Question 1 baseline was measured on 2026-11-04, and Questions 6 and
  7 on 2026-11-05, all on the same machine. Small timing differences (e.g. +1.6 % in inference)
  are within session-to-session variation. Peak memory was not recorded in Question 1.

## Generated files

| Path | Content |
|---|---|
| `results/question7/runs.csv` | per run: seed, correct, accuracy, training/inference time, trainable/total parameters, unfrozen layers, peak GPU memory, per-epoch time and mean loss |
| `results/question7/comparison.csv` | Q7 vs Q1 and vs Q6 for accuracy, training time, inference time and trainable parameters: means, SDs, difference, ratio |
| `results/question7/summary.json` | hyperparameters, protocol, unfrozen layers, parameter counts, aggregates, comparison, runs |
| `results/question7/config.json` | interpretation and rejected alternatives, unfrozen layers with parameter counts, BatchNorm behaviour, weights, preprocessing, parameter counts for Q1/Q6/Q7, protocol, environment, SHA-256 of the Q1 and Q6 baselines |
| `results/question7/predictions_run{1,2,3}_seed{0,1,2}.csv` | per-image test predictions with MNIST index (for Question 8) |
| `results/question7/logs/q7_gpu_run.log` | console log |

## Reproduce

```bash
source .venv/bin/activate
python q7_mobilenet_partial_finetune.py --overwrite     # ~7 min on the GPU (3 runs)
python -m unittest discover -s tests -t . -v
```
