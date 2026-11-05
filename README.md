# Machine Learning — Neural Networks

[English](#english) · [Ελληνικά](#ελληνικά)

<a id="english"></a>

Assignment 2 of the Machine Learning course, part of the MSc in Internet of Things: Artificial Intelligence and Next Generation Networks at the University of the Aegean.

The assignment uses neural networks to recognise handwritten digits in the MNIST dataset: 28×28 grayscale images, 60,000 for training and 10,000 for testing. Three networks are provided as PyTorch scripts:

| Model | Script | Architecture |
|---|---|---|
| MLP | `mnist_MLP.py` | Linear 784→100, ReLU, Linear 100→10, log-softmax |
| CNN | `mnist_CNN.py` | Conv 32 (3×3, stride 1), ReLU, max-pool 2×2; Conv 64 (3×3, stride 1), ReLU, max-pool 2×2; Linear 1600→10, log-softmax |
| MobileNetV2 | `mnist_MobileNetV2.py` | ImageNet-pretrained, frozen feature layers, new 10-class classifier; inputs resized to 224×224 RGB |

All three use mini-batch size 100, learning rate 0.001, Adam and cross-entropy (negative log-likelihood) loss. MLP and CNN train for 5 epochs, MobileNetV2 for 3. The original scripts are kept unchanged. The experiments reuse their architectures and hyperparameters through the `a2/` package.

## Questions

1. Run each network 3 times on the GPU: test accuracy (mean and standard deviation), training time and test-set inference time. **Done.**
2. Run each network once without a GPU and compare the times with Question 1. **Done.**
3. Analyse misclassified test images per network and those misclassified by all three. **Done.**
4. New MLP with two hidden layers (500 and 200, ReLU, dropout 0.2). **Done.**
5. Original MLP with dropout: choose the dropout probability and hidden size on a 10% validation set. **Done.**
6. MobileNetV2 with the whole pretrained network fine-tuned. **Done.**
7. MobileNetV2 with only the last 3 pretrained layers fine-tuned. **Done.**
8. Summary table of all experiments, best model, statistical significance and the usefulness of the GPU. **Done.**

All eight questions are complete.

## Setup

Python 3.14. Install `torch`/`torchvision` from the PyTorch index for your hardware first, then the remaining pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
# AMD GPU (ROCm 7.14), as used for the reported results:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/rocm7.14
# or CPU only:
# pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

MNIST is downloaded to `data/` on first run and is not tracked. The first MobileNetV2 run downloads the ImageNet weights to `~/.cache/torch/hub`.

## Running

```bash
# Question 1: 3 GPU runs per model (seeds 0, 1, 2), ~7 min
python q1_gpu_baseline.py --model all
# Question 2: 1 CPU run per model (seed 0), GPU hidden; MobileNetV2 takes ~46 min
python q2_cpu_comparison.py --model all
# Question 3: misclassification analysis of Question 1 run 1 (seed 0); reads saved predictions, seconds
python q3_misclassification_analysis.py
# Question 4: enhanced MLP, 3 GPU runs (seeds 0, 1, 2), ~40 s
python q4_enhanced_mlp.py
# Question 5: 25-configuration validation search, then 3 GPU runs of the winner (seeds 0, 1, 2), ~5 min
python q5_mlp_hyperparameter_search.py
# Question 6: MobileNetV2 with full fine-tuning, 3 GPU runs (seeds 0, 1, 2), ~31 min
python q6_mobilenet_full_finetune.py
# Question 7: MobileNetV2 with the last 3 feature layers fine-tuned, 3 GPU runs (seeds 0, 1, 2), ~7 min
python q7_mobilenet_partial_finetune.py
# Question 8: final comparison and McNemar test from the saved results (no training), seconds
python q8_final_comparison.py
# Quick pipeline check of the Question 1/2/4/5/6/7 scripts (truncated runs, written to results/smoke/, not tracked)
python q1_gpu_baseline.py --model all --smoke
python q2_cpu_comparison.py --model all --smoke
python q4_enhanced_mlp.py --smoke
python q5_mlp_hyperparameter_search.py --smoke
python q6_mobilenet_full_finetune.py --smoke
python q7_mobilenet_partial_finetune.py --smoke
```

Both scripts accept `--model mlp cnn mobilenetv2`, `--warmup-batches` (default 20), `--num-workers` (default 0, as in the original scripts), `--out` and `--overwrite`. They refuse to replace existing results without `--overwrite`. `q2_cpu_comparison.py` reads the Question 1 results as its GPU baseline and never writes to them. `q3_misclassification_analysis.py` (`--run 1`, `--per-digit 6`, `--top-pairs 5`, `--out`, `--overwrite`) only reads the Question 1 prediction files and the MNIST test images. `q4_enhanced_mlp.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) reads the Question 1 MLP results as its baseline and refuses to run with a different protocol. `q5_mlp_hyperparameter_search.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) reads the Question 1 and Question 4 results as baselines. `q6_mobilenet_full_finetune.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) reads the Question 1 MobileNetV2 results as its baseline and refuses to run with a different protocol or training configuration. `q7_mobilenet_partial_finetune.py` (same options) reads the Question 1 and Question 6 MobileNetV2 results as baselines. `q8_final_comparison.py` (`--out`, `--overwrite`) only reads the saved results of Questions 1–7 and the MNIST test labels.

## Results

Hardware: AMD Radeon RX 9070 XT (ROCm 7.14), AMD Ryzen 9 7900X (12 cores), PyTorch 2.15.0.

### Question 1: GPU, 3 runs per model

| Model | Accuracy (mean ± sample SD) | Training (s) | Inference, 10,000 images (s) |
|---|---|---|---|
| MLP | 96.97 ± 0.06 % | 9.66 | 0.274 |
| CNN | 98.89 ± 0.09 % | 9.54 | 0.257 |
| MobileNetV2 | 95.99 ± 0.18 % | 120.36 | 6.388 |

Details: [`results/question1/SUMMARY.md`](results/question1/SUMMARY.md)

### Question 2: CPU, 1 run per model, compared with the Question 1 GPU mean

Speedup = CPU time / GPU time (above 1×: GPU faster).

| Model | CPU accuracy | CPU / GPU training (s) | Training speedup | CPU / GPU inference (s) | Inference speedup |
|---|---|---|---|---|---|
| MLP | 96.98 % | 8.58 / 9.66 | 0.89× | 0.243 / 0.274 | 0.89× |
| CNN | 98.98 % | 32.59 / 9.54 | 3.42× | 0.424 / 0.257 | 1.65× |
| MobileNetV2 | 95.94 % | 2745.10 / 120.36 | 22.81× | 143.71 / 6.388 | 22.50× |

Details: [`results/question2/SUMMARY.md`](results/question2/SUMMARY.md)

### Question 3: misclassifications of Question 1 run 1 (seed 0)

| Model | Misclassified | Highest error rate (true digit) | Most frequent confusion (true → predicted) |
|---|---|---|---|
| MLP | 302 | 2 (4.36 %) | 2 → 8 (17) |
| CNN | 100 | 9 (2.08 %) | 3 → 5 (7) |
| MobileNetV2 | 394 | 2 (6.69 %) | 5 → 3 (25) |

29 test images are misclassified by all three models. Error rates are per true digit (errors / test images of that digit), because the classes have different sizes. The figures in `results/question3/figures/` show misclassified images with their MNIST index and true and predicted labels.

Details: [`results/question3/SUMMARY.md`](results/question3/SUMMARY.md)

### Question 4: enhanced MLP, 3 GPU runs, compared with the Question 1 MLP

Architecture: Linear 784→500, ReLU, Dropout 0.2, Linear 500→200, ReLU, Dropout 0.2, Linear 200→10, log-softmax (494,710 parameters, compared with 79,510). Training configuration as for the original MLP.

| Model | Accuracy (mean ± sample SD) | Training (s) | Inference (s) |
|---|---|---|---|
| Original MLP (Question 1) | 96.97 ± 0.06 % | 9.66 | 0.274 |
| Enhanced MLP (Question 4) | 98.04 ± 0.12 % | 10.48 | 0.275 |
| Difference | +1.07 percentage points | +0.82 (×1.085) | +0.001 (×1.004) |

Details: [`results/question4/SUMMARY.md`](results/question4/SUMMARY.md)

### Question 5: hyperparameter search for the original MLP with dropout

Architecture: Linear 784→h, ReLU, Dropout p, Linear h→10, log-softmax. The 25 combinations h ∈ {200, 400, 600, 800, 1000} × p ∈ {0.1, 0.2, 0.3, 0.4, 0.5} were trained on 54,000 training images and evaluated on a fixed validation set of 6,000 (10%). **Winner: h = 800, p = 0.2** (validation accuracy 98.07%). It was then retrained from scratch on all 60,000 training images and tested 3 times.

| Model | Accuracy (mean ± sample SD) | Training (s) | Inference (s) |
|---|---|---|---|
| Original MLP (Question 1) | 96.97 ± 0.06 % | 9.66 | 0.274 |
| Tuned MLP, h = 800, p = 0.2 (Question 5) | 97.97 ± 0.14 % | 9.22 | 0.255 |
| Enhanced MLP, two hidden layers (Question 4) | 98.04 ± 0.12 % | 10.48 | 0.275 |

The tuned MLP is 1.00 percentage points more accurate than the original MLP. Its difference from the Question 4 model (−0.07 percentage points) is smaller than the run-to-run standard deviation.

Details: [`results/question5/SUMMARY.md`](results/question5/SUMMARY.md)

### Question 6: MobileNetV2 with full fine-tuning, 3 GPU runs, compared with Question 1

The only change from Question 1 is that no layer is frozen: all 2,236,682 parameters are trainable, compared with 12,810 (the classifier only) in Question 1. Weights, preprocessing, classifier and training configuration are the same as in Question 1.

| Model | Accuracy (mean ± sample SD) | Training (s) | Inference (s) |
|---|---|---|---|
| MobileNetV2, frozen features (Question 1) | 95.99 ± 0.18 % | 120.36 | 6.388 |
| MobileNetV2, full fine-tuning (Question 6) | 99.03 ± 0.21 % | 603.39 | 6.424 |
| Difference | +3.04 percentage points | ×5.01 | ×1.006 |

Full fine-tuning reduces test errors from about 401 to about 97 per 10,000 images. Training takes 5 times longer, because the backward pass covers the whole network. Peak GPU memory was 7.39 GiB at batch size 100.

Details: [`results/question6/SUMMARY.md`](results/question6/SUMMARY.md)

### Question 7: MobileNetV2 with the last 3 layers fine-tuned, 3 GPU runs, compared with Questions 1 and 6

"The last 3 layers of the pretrained network" are the last three top-level modules of `model.features`: `features[16]` and `features[17]` (`InvertedResidual` blocks) and `features[18]` (final 1×1 Conv-BatchNorm-ReLU6). They are trained together with the 10-class classifier: 1,218,890 of 2,236,682 parameters (54.5%). `features[0]`–`features[15]` stay frozen. Everything else is the same as in Question 1.

| Model | Trainable parameters | Accuracy (mean ± sample SD) | Training (s) | Inference (s) |
|---|---|---|---|---|
| Frozen features (Question 1) | 12,810 | 95.99 ± 0.18 % | 120.36 | 6.388 |
| Last 3 layers + classifier (Question 7) | 1,218,890 | 99.03 ± 0.26 % | 126.90 | 6.492 |
| Full fine-tuning (Question 6) | 2,236,682 | 99.03 ± 0.21 % | 603.39 | 6.424 |

Partial fine-tuning is 3.04 percentage points more accurate than the frozen model. Its observed mean accuracy is nearly identical to full fine-tuning (difference −0.003 percentage points, well within the run-to-run variation); with 3 runs per model this is not a proof of statistical equivalence. It trains 4.8 times faster than full fine-tuning (×0.210) and only 5.4% slower than the frozen model. Peak GPU memory was 1.05 GiB, compared with 7.39 GiB for full fine-tuning.

Details: [`results/question7/SUMMARY.md`](results/question7/SUMMARY.md)

### Question 8: final comparison of all configurations (GPU, 3 runs each)

| Rank | Configuration | Accuracy (mean ± sample SD) | Training (s) | Inference (s) | Trainable parameters |
|---|---|---|---|---|---|
| 1 | Q6 MobileNetV2, full fine-tuning | 99.03 ± 0.21 % | 603.39 | 6.424 | 2,236,682 |
| 2 | Q7 MobileNetV2, last 3 layers fine-tuned | 99.03 ± 0.26 % | 126.90 | 6.492 | 1,218,890 |
| 3 | Q1 CNN (original) | 98.89 ± 0.09 % | 9.54 | 0.257 | 34,826 |
| 4 | Q4 MLP 500-200, dropout 0.2 | 98.04 ± 0.12 % | 10.48 | 0.275 | 494,710 |
| 5 | Q5 MLP 800, dropout 0.2 | 97.97 ± 0.14 % | 9.22 | 0.255 | 636,010 |
| 6 | Q1 MLP (original) | 96.97 ± 0.06 % | 9.66 | 0.274 | 79,510 |
| 7 | Q1 MobileNetV2, frozen features | 95.99 ± 0.18 % | 120.36 | 6.388 | 12,810 |

- **Best vs second best:** Q6 and Q7 have nearly identical observed mean accuracy. An exact McNemar test on their seed-0 runs (89 vs 82 discordant test images) gives p = 0.6465 > α = 0.05: no significant difference, which is not a proof of equivalence. Seeds 1 and 2, tested separately, give p = 0.1843 and 0.0645.
- **Speed and trade-off:** MLPs and the CNN train in about 10 s and test in about 0.26 s; MobileNetV2 needs 120–603 s and about 6.4 s. The original CNN offers the best accuracy/training-time trade-off (0.14 percentage points below the best, 13 times faster to train than Q7).
- **GPU usefulness (Question 2):** training speedup 0.89× for the MLP (no benefit), 3.42× for the CNN and 22.81× for MobileNetV2.

Details: [`results/question8/SUMMARY.md`](results/question8/SUMMARY.md)

## Methodology

- Training time covers all epochs, including data loading. Inference time covers one full pass over the test set. Dataset loading and model construction are excluded.
- On the GPU, `torch.cuda.synchronize()` runs before every clock read, so asynchronous work is fully counted.
- Before the timed runs, 20 untimed warm-up batches run on a throw-away model, on both devices. On the GPU they absorb one-time kernel compilation.
- Each run is seeded (weight initialisation and shuffle order). Test predictions are saved with their MNIST test index.
- For Question 2, the GPU is hidden from PyTorch before `torch` is imported, and the script stops if a GPU is still visible.
- For Question 3, the saved predictions of Question 1 run 1 are used without retraining. Every figure is drawn from the original MNIST test images, matched by test index. Examples are chosen by lowest test index.
- For Question 4, the enhanced MLP uses the same seeds, warm-up, timing and training configuration as the Question 1 MLP. Dropout is active only in training mode.
- For Question 5, the split uses `random_split` with a fixed seed and is the same for all 25 configurations. The selection uses validation accuracy only, with ties broken by fewer hidden neurons, then by lower dropout. The test set is not loaded until the winner has been written to `best_config.json`.
- For Question 6, each run builds a fresh pretrained MobileNetV2 and replaces the classifier after seeding, so its starting weights equal those of the Question 1 run with the same seed. Every parameter has `requires_grad=True`.
- For Question 7, each run freezes `model.features` and replaces the classifier as in Question 1, then unfreezes `features[16]`, `features[17]` and `features[18]`. Frozen parameters receive no gradients. As in the original script, `model.train()` also updates the BatchNorm statistics of frozen layers.
- For Question 8, all accuracies are recounted from the saved predictions, and no model is trained or run. The two most accurate configurations are compared with an exact two-sided McNemar test on their seed-0 runs, paired by MNIST test index, with α = 0.05. The other seeds are reported separately and are not pooled.

Changes compared with the original scripts: none to architectures, hyperparameters, loss, optimiser, transforms or data loaders. The harness adds timing, seeding, repetition, warm-up and result files. The test loader also returns each image's index. MobileNetV2's per-epoch loss is accumulated on the device instead of with a per-step `.item()` call.

Differences between the PDF and the code: the PDF lists the CNN as conv → pool → ReLU, but the code applies ReLU before pooling. These are mathematically equivalent, and the code is kept as provided. The PDF's MLP description also omits the ReLU that the code contains.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

114 unit tests cover timing, aggregation, prediction files, model fidelity, speedup calculation, baseline validation, CPU-only enforcement, misclassification counting, confusion pairs, index alignment, common errors, figure generation, the enhanced MLP architecture, dropout behaviour, baseline comparison, the hyperparameter grid, the validation split, winner selection, protection against test-set leakage, the trainable layers and parameter counts of the fine-tuned MobileNetV2, preservation of its preprocessing and training configuration, the layers selected for partial fine-tuning, with no gradients for frozen parameters, and the final comparison: loading of the seven configurations, ranking, prediction alignment and the exact McNemar test. They use synthetic data on the CPU: no GPU, no MNIST download, no pretrained weights. GitHub Actions (`.github/workflows/python-tests.yml`) runs them with CPU-only PyTorch on every pull request and push to `master`.

## Directories

- `mnist_*.py`: the original scripts (unchanged).
- `a2/`: shared code (`models`, `data`, `timing`, `experiment`, `stats`, `io_utils`, `environment`, `compare`, `misclassification`, `figures`, `search`, `final_comparison`).
- `q1_gpu_baseline.py`, `q2_cpu_comparison.py`, `q3_misclassification_analysis.py`, `q4_enhanced_mlp.py`, `q5_mlp_hyperparameter_search.py`, `q6_mobilenet_full_finetune.py`, `q7_mobilenet_partial_finetune.py`, `q8_final_comparison.py`: scripts for Questions 1–8.
- `tests/`: unit tests.
- `results/question1/` to `results/question8/`: per-question CSV/JSON results, per-image predictions, logs, figures and `SUMMARY.md`.
- `docs/`: the assignment description (PDF).

---

<a id="ελληνικά"></a>

# Machine Learning — Neural Networks

Εργασία 2 του μαθήματος Μηχανική Μάθηση, στο πλαίσιο του ΠΜΣ «Διαδίκτυο των Πραγμάτων: Τεχνητή Νοημοσύνη και Σύγχρονα Δίκτυα» του Πανεπιστημίου Αιγαίου.

Η εργασία χρησιμοποιεί νευρωνικά δίκτυα για την αναγνώριση χειρόγραφων ψηφίων στο MNIST dataset: grayscale εικόνες 28×28, 60.000 για εκπαίδευση και 10.000 για έλεγχο. Δίνονται τρία δίκτυα ως scripts σε PyTorch:

| Μοντέλο | Script | Αρχιτεκτονική |
|---|---|---|
| MLP | `mnist_MLP.py` | Linear 784→100, ReLU, Linear 100→10, log-softmax |
| CNN | `mnist_CNN.py` | Conv 32 (3×3, stride 1), ReLU, max-pool 2×2· Conv 64 (3×3, stride 1), ReLU, max-pool 2×2· Linear 1600→10, log-softmax |
| MobileNetV2 | `mnist_MobileNetV2.py` | Προ-εκπαιδευμένο στο ImageNet, παγωμένα (frozen) επίπεδα χαρακτηριστικών, νέος classifier 10 κλάσεων· είσοδοι 224×224 RGB |

Και τα τρία χρησιμοποιούν mini-batch size 100, learning rate 0.001, Adam και cross-entropy (negative log-likelihood) loss. Τα MLP και CNN εκπαιδεύονται για 5 epochs και το MobileNetV2 για 3. Τα αρχικά scripts διατηρούνται αμετάβλητα. Τα πειράματα επαναχρησιμοποιούν τις αρχιτεκτονικές και τις υπερ-παραμέτρους τους μέσω του πακέτου `a2/`.

## Ερωτήματα

1. Εκτέλεση κάθε δικτύου 3 φορές με GPU: test accuracy (μέσος όρος και τυπική απόκλιση), χρόνος εκπαίδευσης και χρόνος εφαρμογής στο test set. **Ολοκληρώθηκε.**
2. Εκτέλεση κάθε δικτύου μία φορά χωρίς GPU και σύγκριση των χρόνων με το Ερώτημα 1. **Ολοκληρώθηκε.**
3. Ανάλυση των εικόνων του test set που ταξινομούνται λάθος από κάθε δίκτυο και από τα τρία μαζί. **Ολοκληρώθηκε.**
4. Νέο MLP με δύο κρυμμένα επίπεδα (500 και 200, ReLU, dropout 0.2). **Ολοκληρώθηκε.**
5. Αρχικό MLP με dropout: επιλογή πιθανότητας dropout και διάστασης κρυμμένου επιπέδου με validation set 10%. **Ολοκληρώθηκε.**
6. MobileNetV2 με fine-tuning ολόκληρου του προ-εκπαιδευμένου δικτύου. **Ολοκληρώθηκε.**
7. MobileNetV2 με fine-tuning μόνο των 3 τελευταίων προ-εκπαιδευμένων επιπέδων. **Ολοκληρώθηκε.**
8. Συγκεντρωτικός πίνακας όλων των πειραμάτων, καλύτερο μοντέλο, στατιστική σημαντικότητα και χρησιμότητα της GPU. **Ολοκληρώθηκε.**

Και τα οκτώ ερωτήματα έχουν ολοκληρωθεί.

## Εγκατάσταση

Python 3.14. Εγκαταστήστε πρώτα τα `torch`/`torchvision` από το PyTorch index που αντιστοιχεί στο υλικό σας και μετά τις υπόλοιπες εξαρτήσεις με σταθερές εκδόσεις:

```bash
python -m venv .venv && source .venv/bin/activate
# AMD GPU (ROCm 7.14), όπως για τα αποτελέσματα που παρουσιάζονται:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/rocm7.14
# ή μόνο CPU:
# pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Το MNIST κατεβαίνει στον φάκελο `data/` στην πρώτη εκτέλεση και δεν περιλαμβάνεται στο αποθετήριο. Η πρώτη εκτέλεση του MobileNetV2 κατεβάζει τα βάρη ImageNet στο `~/.cache/torch/hub`.

## Εκτέλεση

```bash
# Ερώτημα 1: 3 εκτελέσεις ανά μοντέλο σε GPU (seeds 0, 1, 2), ~7 λεπτά
python q1_gpu_baseline.py --model all
# Ερώτημα 2: 1 εκτέλεση ανά μοντέλο σε CPU (seed 0), με κρυμμένη GPU· το MobileNetV2 διαρκεί ~46 λεπτά
python q2_cpu_comparison.py --model all
# Ερώτημα 3: ανάλυση λαθών της εκτέλεσης 1 (seed 0) του Ερωτήματος 1· διαβάζει τις αποθηκευμένες προβλέψεις, δευτερόλεπτα
python q3_misclassification_analysis.py
# Ερώτημα 4: βελτιωμένο MLP, 3 εκτελέσεις σε GPU (seeds 0, 1, 2), ~40 δευτερόλεπτα
python q4_enhanced_mlp.py
# Ερώτημα 5: αναζήτηση 25 συνδυασμών στο validation set και 3 εκτελέσεις του καλύτερου σε GPU (seeds 0, 1, 2), ~5 λεπτά
python q5_mlp_hyperparameter_search.py
# Ερώτημα 6: MobileNetV2 με πλήρες fine-tuning, 3 εκτελέσεις σε GPU (seeds 0, 1, 2), ~31 λεπτά
python q6_mobilenet_full_finetune.py
# Ερώτημα 7: MobileNetV2 με fine-tuning των 3 τελευταίων επιπέδων χαρακτηριστικών, 3 εκτελέσεις σε GPU (seeds 0, 1, 2), ~7 λεπτά
python q7_mobilenet_partial_finetune.py
# Ερώτημα 8: τελική σύγκριση και έλεγχος McNemar από τα αποθηκευμένα αποτελέσματα (χωρίς εκπαίδευση), δευτερόλεπτα
python q8_final_comparison.py
# Γρήγορος έλεγχος της ροής για τα scripts των Ερωτημάτων 1/2/4/5/6/7 (περικομμένες εκτελέσεις, στο results/smoke/, εκτός αποθετηρίου)
python q1_gpu_baseline.py --model all --smoke
python q2_cpu_comparison.py --model all --smoke
python q4_enhanced_mlp.py --smoke
python q5_mlp_hyperparameter_search.py --smoke
python q6_mobilenet_full_finetune.py --smoke
python q7_mobilenet_partial_finetune.py --smoke
```

Και τα δύο scripts δέχονται `--model mlp cnn mobilenetv2`, `--warmup-batches` (προεπιλογή 20), `--num-workers` (προεπιλογή 0, όπως στα αρχικά scripts), `--out` και `--overwrite`. Δεν αντικαθιστούν υπάρχοντα αποτελέσματα χωρίς `--overwrite`. Το `q2_cpu_comparison.py` διαβάζει τα αποτελέσματα του Ερωτήματος 1 ως GPU baseline και δεν γράφει ποτέ σε αυτά. Το `q3_misclassification_analysis.py` (`--run 1`, `--per-digit 6`, `--top-pairs 5`, `--out`, `--overwrite`) διαβάζει μόνο τα αρχεία προβλέψεων του Ερωτήματος 1 και τις εικόνες του MNIST test set. Το `q4_enhanced_mlp.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) διαβάζει τα αποτελέσματα του MLP του Ερωτήματος 1 ως baseline και δεν εκτελείται αν το πρωτόκολλο διαφέρει. Το `q5_mlp_hyperparameter_search.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) διαβάζει τα αποτελέσματα των Ερωτημάτων 1 και 4 ως baselines. Το `q6_mobilenet_full_finetune.py` (`--seeds`, `--warmup-batches`, `--num-workers`, `--out`, `--overwrite`) διαβάζει τα αποτελέσματα του MobileNetV2 του Ερωτήματος 1 ως baseline και δεν εκτελείται αν το πρωτόκολλο ή οι ρυθμίσεις εκπαίδευσης διαφέρουν. Το `q7_mobilenet_partial_finetune.py` (ίδιες επιλογές) διαβάζει τα αποτελέσματα του MobileNetV2 των Ερωτημάτων 1 και 6 ως baselines. Το `q8_final_comparison.py` (`--out`, `--overwrite`) διαβάζει μόνο τα αποθηκευμένα αποτελέσματα των Ερωτημάτων 1–7 και τις ετικέτες του MNIST test set.

## Αποτελέσματα

Υλικό: AMD Radeon RX 9070 XT (ROCm 7.14), AMD Ryzen 9 7900X (12 πυρήνες), PyTorch 2.15.0.

### Ερώτημα 1: GPU, 3 εκτελέσεις ανά μοντέλο

| Μοντέλο | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή, 10.000 εικόνες (s) |
|---|---|---|---|
| MLP | 96,97 ± 0,06 % | 9,66 | 0,274 |
| CNN | 98,89 ± 0,09 % | 9,54 | 0,257 |
| MobileNetV2 | 95,99 ± 0,18 % | 120,36 | 6,388 |

Λεπτομέρειες: [`results/question1/SUMMARY.md`](results/question1/SUMMARY.md)

### Ερώτημα 2: CPU, 1 εκτέλεση ανά μοντέλο, σε σύγκριση με τον μέσο όρο GPU του Ερωτήματος 1

Επιτάχυνση (speedup) = χρόνος CPU / χρόνος GPU (πάνω από 1×: ταχύτερη η GPU).

| Μοντέλο | Accuracy CPU | Εκπαίδευση CPU / GPU (s) | Επιτάχυνση εκπαίδευσης | Εφαρμογή CPU / GPU (s) | Επιτάχυνση εφαρμογής |
|---|---|---|---|---|---|
| MLP | 96,98 % | 8,58 / 9,66 | 0,89× | 0,243 / 0,274 | 0,89× |
| CNN | 98,98 % | 32,59 / 9,54 | 3,42× | 0,424 / 0,257 | 1,65× |
| MobileNetV2 | 95,94 % | 2745,10 / 120,36 | 22,81× | 143,71 / 6,388 | 22,50× |

Λεπτομέρειες: [`results/question2/SUMMARY.md`](results/question2/SUMMARY.md)

### Ερώτημα 3: λάθη της εκτέλεσης 1 (seed 0) του Ερωτήματος 1

| Μοντέλο | Λάθος ταξινομήσεις | Υψηλότερο ποσοστό λάθους (πραγματικό ψηφίο) | Συχνότερη σύγχυση (πραγματικό → πρόβλεψη) |
|---|---|---|---|
| MLP | 302 | 2 (4,36 %) | 2 → 8 (17) |
| CNN | 100 | 9 (2,08 %) | 3 → 5 (7) |
| MobileNetV2 | 394 | 2 (6,69 %) | 5 → 3 (25) |

29 εικόνες του test set ταξινομούνται λάθος και από τα τρία μοντέλα. Τα ποσοστά λάθους υπολογίζονται ανά πραγματικό ψηφίο (λάθη / εικόνες του test set με αυτό το ψηφίο), επειδή οι κλάσεις έχουν διαφορετικό μέγεθος. Τα σχήματα στο `results/question3/figures/` δείχνουν εικόνες που ταξινομήθηκαν λάθος, με τον δείκτη τους στο MNIST, την πραγματική ετικέτα και την πρόβλεψη.

Λεπτομέρειες: [`results/question3/SUMMARY.md`](results/question3/SUMMARY.md)

### Ερώτημα 4: βελτιωμένο MLP, 3 εκτελέσεις σε GPU, σε σύγκριση με το MLP του Ερωτήματος 1

Αρχιτεκτονική: Linear 784→500, ReLU, Dropout 0.2, Linear 500→200, ReLU, Dropout 0.2, Linear 200→10, log-softmax (494.710 παράμετροι, έναντι 79.510). Ρυθμίσεις εκπαίδευσης όπως στο αρχικό MLP.

| Μοντέλο | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή (s) |
|---|---|---|---|
| Αρχικό MLP (Ερώτημα 1) | 96,97 ± 0,06 % | 9,66 | 0,274 |
| Βελτιωμένο MLP (Ερώτημα 4) | 98,04 ± 0,12 % | 10,48 | 0,275 |
| Διαφορά | +1,07 ποσοστιαίες μονάδες | +0,82 (×1,085) | +0,001 (×1,004) |

Λεπτομέρειες: [`results/question4/SUMMARY.md`](results/question4/SUMMARY.md)

### Ερώτημα 5: αναζήτηση υπερ-παραμέτρων για το αρχικό MLP με dropout

Αρχιτεκτονική: Linear 784→h, ReLU, Dropout p, Linear h→10, log-softmax. Οι 25 συνδυασμοί h ∈ {200, 400, 600, 800, 1000} × p ∈ {0.1, 0.2, 0.3, 0.4, 0.5} εκπαιδεύτηκαν σε 54.000 εικόνες εκπαίδευσης και αξιολογήθηκαν σε σταθερό validation set 6.000 εικόνων (10%). **Καλύτερος συνδυασμός: h = 800, p = 0.2** (validation accuracy 98,07%). Στη συνέχεια εκπαιδεύτηκε από την αρχή σε όλες τις 60.000 εικόνες εκπαίδευσης και ελέγχθηκε 3 φορές.

| Μοντέλο | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή (s) |
|---|---|---|---|
| Αρχικό MLP (Ερώτημα 1) | 96,97 ± 0,06 % | 9,66 | 0,274 |
| Βελτιστοποιημένο MLP, h = 800, p = 0.2 (Ερώτημα 5) | 97,97 ± 0,14 % | 9,22 | 0,255 |
| Βελτιωμένο MLP, δύο κρυμμένα επίπεδα (Ερώτημα 4) | 98,04 ± 0,12 % | 10,48 | 0,275 |

Το βελτιστοποιημένο MLP είναι 1,00 ποσοστιαίες μονάδες ακριβέστερο από το αρχικό MLP. Η διαφορά του από το μοντέλο του Ερωτήματος 4 (−0,07 ποσοστιαίες μονάδες) είναι μικρότερη από την τυπική απόκλιση μεταξύ εκτελέσεων.

Λεπτομέρειες: [`results/question5/SUMMARY.md`](results/question5/SUMMARY.md)

### Ερώτημα 6: MobileNetV2 με πλήρες fine-tuning, 3 εκτελέσεις σε GPU, σε σύγκριση με το Ερώτημα 1

Η μόνη αλλαγή σε σχέση με το Ερώτημα 1 είναι ότι κανένα επίπεδο δεν είναι παγωμένο: εκπαιδεύονται και οι 2.236.682 παράμετροι, έναντι 12.810 (μόνο ο classifier) στο Ερώτημα 1. Τα βάρη, η προεπεξεργασία, ο classifier και οι ρυθμίσεις εκπαίδευσης είναι ίδια με το Ερώτημα 1.

| Μοντέλο | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή (s) |
|---|---|---|---|
| MobileNetV2, παγωμένα χαρακτηριστικά (Ερώτημα 1) | 95,99 ± 0,18 % | 120,36 | 6,388 |
| MobileNetV2, πλήρες fine-tuning (Ερώτημα 6) | 99,03 ± 0,21 % | 603,39 | 6,424 |
| Διαφορά | +3,04 ποσοστιαίες μονάδες | ×5,01 | ×1,006 |

Το πλήρες fine-tuning μειώνει τα λάθη στο test set από περίπου 401 σε περίπου 97 ανά 10.000 εικόνες. Η εκπαίδευση διαρκεί 5 φορές περισσότερο, επειδή το backward pass καλύπτει ολόκληρο το δίκτυο. Η μέγιστη χρήση μνήμης GPU ήταν 7,39 GiB με batch size 100.

Λεπτομέρειες: [`results/question6/SUMMARY.md`](results/question6/SUMMARY.md)

### Ερώτημα 7: MobileNetV2 με fine-tuning των 3 τελευταίων επιπέδων, 3 εκτελέσεις σε GPU, σε σύγκριση με τα Ερωτήματα 1 και 6

«Τα 3 τελευταία επίπεδα του προ-εκπαιδευμένου δικτύου» είναι τα τρία τελευταία modules πρώτου επιπέδου του `model.features`: τα `features[16]` και `features[17]` (blocks `InvertedResidual`) και το `features[18]` (τελικό 1×1 Conv-BatchNorm-ReLU6). Εκπαιδεύονται μαζί με τον classifier 10 κλάσεων: 1.218.890 από 2.236.682 παραμέτρους (54,5%). Τα `features[0]`–`features[15]` παραμένουν παγωμένα. Όλα τα υπόλοιπα είναι ίδια με το Ερώτημα 1.

| Μοντέλο | Εκπαιδεύσιμες παράμετροι | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή (s) |
|---|---|---|---|---|
| Παγωμένα χαρακτηριστικά (Ερώτημα 1) | 12.810 | 95,99 ± 0,18 % | 120,36 | 6,388 |
| 3 τελευταία επίπεδα + classifier (Ερώτημα 7) | 1.218.890 | 99,03 ± 0,26 % | 126,90 | 6,492 |
| Πλήρες fine-tuning (Ερώτημα 6) | 2.236.682 | 99,03 ± 0,21 % | 603,39 | 6,424 |

Το μερικό fine-tuning είναι 3,04 ποσοστιαίες μονάδες ακριβέστερο από το παγωμένο μοντέλο. Η παρατηρούμενη μέση ακρίβειά του είναι σχεδόν ίδια με αυτή του πλήρους fine-tuning (διαφορά −0,003 ποσοστιαίες μονάδες, πολύ μικρότερη από τη διακύμανση μεταξύ εκτελέσεων)· με 3 εκτελέσεις ανά μοντέλο αυτό δεν αποτελεί απόδειξη στατιστικής ισοδυναμίας. Εκπαιδεύεται 4,8 φορές ταχύτερα από το πλήρες fine-tuning (×0,210) και μόλις 5,4% πιο αργά από το παγωμένο μοντέλο. Η μέγιστη χρήση μνήμης GPU ήταν 1,05 GiB, έναντι 7,39 GiB στο πλήρες fine-tuning.

Λεπτομέρειες: [`results/question7/SUMMARY.md`](results/question7/SUMMARY.md)

### Ερώτημα 8: τελική σύγκριση όλων των διαμορφώσεων (GPU, 3 εκτελέσεις η καθεμία)

| Κατάταξη | Διαμόρφωση | Accuracy (μέσος όρος ± δειγματική τυπική απόκλιση) | Εκπαίδευση (s) | Εφαρμογή (s) | Εκπαιδεύσιμες παράμετροι |
|---|---|---|---|---|---|
| 1 | Ερ. 6 MobileNetV2, πλήρες fine-tuning | 99,03 ± 0,21 % | 603,39 | 6,424 | 2.236.682 |
| 2 | Ερ. 7 MobileNetV2, fine-tuning 3 τελευταίων επιπέδων | 99,03 ± 0,26 % | 126,90 | 6,492 | 1.218.890 |
| 3 | Ερ. 1 CNN (αρχικό) | 98,89 ± 0,09 % | 9,54 | 0,257 | 34.826 |
| 4 | Ερ. 4 MLP 500-200, dropout 0.2 | 98,04 ± 0,12 % | 10,48 | 0,275 | 494.710 |
| 5 | Ερ. 5 MLP 800, dropout 0.2 | 97,97 ± 0,14 % | 9,22 | 0,255 | 636.010 |
| 6 | Ερ. 1 MLP (αρχικό) | 96,97 ± 0,06 % | 9,66 | 0,274 | 79.510 |
| 7 | Ερ. 1 MobileNetV2, παγωμένα χαρακτηριστικά | 95,99 ± 0,18 % | 120,36 | 6,388 | 12.810 |

- **Καλύτερο έναντι δεύτερου καλύτερου:** τα Ερ. 6 και Ερ. 7 έχουν σχεδόν ίδια παρατηρούμενη μέση ακρίβεια. Ο ακριβής έλεγχος McNemar στις εκτελέσεις με seed 0 (89 έναντι 82 εικόνων ασυμφωνίας) δίνει p = 0,6465 > α = 0,05: δεν υπάρχει στατιστικά σημαντική διαφορά, κάτι που δεν αποτελεί απόδειξη ισοδυναμίας. Τα seeds 1 και 2, ελεγμένα χωριστά, δίνουν p = 0,1843 και 0,0645.
- **Ταχύτητα και ισορροπία:** τα MLP και το CNN εκπαιδεύονται σε περίπου 10 s και εφαρμόζονται στο test set σε περίπου 0,26 s· το MobileNetV2 χρειάζεται 120–603 s και περίπου 6,4 s. Το αρχικό CNN προσφέρει την καλύτερη ισορροπία ακρίβειας/χρόνου εκπαίδευσης (0,14 ποσοστιαίες μονάδες κάτω από το καλύτερο, 13 φορές ταχύτερη εκπαίδευση από το Ερ. 7).
- **Χρησιμότητα της GPU (Ερώτημα 2):** επιτάχυνση εκπαίδευσης 0,89× για το MLP (κανένα όφελος), 3,42× για το CNN και 22,81× για το MobileNetV2.

Λεπτομέρειες: [`results/question8/SUMMARY.md`](results/question8/SUMMARY.md)

## Μεθοδολογία

- Ο χρόνος εκπαίδευσης καλύπτει όλα τα epochs, μαζί με τη φόρτωση δεδομένων. Ο χρόνος εφαρμογής καλύπτει ένα πλήρες πέρασμα του test set. Η φόρτωση του dataset και η δημιουργία του μοντέλου εξαιρούνται.
- Στη GPU εκτελείται `torch.cuda.synchronize()` πριν από κάθε ανάγνωση του χρονομέτρου, ώστε να μετράται πλήρως η ασύγχρονη εκτέλεση.
- Πριν από τις χρονομετρούμενες εκτελέσεις, εκτελούνται 20 μη χρονομετρούμενα warm-up batches σε ένα μοντέλο που απορρίπτεται, και στις δύο συσκευές. Στη GPU απορροφούν την εφάπαξ μεταγλώττιση των kernels.
- Κάθε εκτέλεση έχει seed (αρχικοποίηση βαρών και σειρά ανακατέματος). Οι προβλέψεις στο test set αποθηκεύονται μαζί με τον δείκτη κάθε εικόνας στο MNIST test set.
- Στο Ερώτημα 2 η GPU αποκρύπτεται από το PyTorch πριν από το import του `torch`, και το script σταματά αν η GPU παραμένει ορατή.
- Στο Ερώτημα 3 χρησιμοποιούνται οι αποθηκευμένες προβλέψεις της εκτέλεσης 1 του Ερωτήματος 1, χωρίς επανεκπαίδευση. Κάθε σχήμα δημιουργείται από τις αρχικές εικόνες του MNIST test set, με αντιστοίχιση μέσω του δείκτη τους. Τα παραδείγματα επιλέγονται με βάση τον μικρότερο δείκτη στο test set.
- Στο Ερώτημα 4 το βελτιωμένο MLP χρησιμοποιεί τα ίδια seeds, warm-up, χρονομέτρηση και ρυθμίσεις εκπαίδευσης με το MLP του Ερωτήματος 1. Το dropout είναι ενεργό μόνο κατά την εκπαίδευση.
- Στο Ερώτημα 5 ο διαχωρισμός γίνεται με `random_split` και σταθερό seed και είναι ίδιος για όλους τους 25 συνδυασμούς. Η επιλογή βασίζεται μόνο στο validation accuracy, με ισοπαλίες να λύνονται υπέρ λιγότερων κρυμμένων νευρώνων και μετά χαμηλότερου dropout. Το test set φορτώνεται μόνο αφού ο καλύτερος συνδυασμός έχει γραφτεί στο `best_config.json`.
- Στο Ερώτημα 6 κάθε εκτέλεση δημιουργεί νέο προ-εκπαιδευμένο MobileNetV2 και αντικαθιστά τον classifier μετά τον ορισμό του seed, ώστε τα αρχικά βάρη να είναι ίδια με αυτά της εκτέλεσης του Ερωτήματος 1 με το ίδιο seed. Όλες οι παράμετροι έχουν `requires_grad=True`.
- Στο Ερώτημα 7 κάθε εκτέλεση παγώνει το `model.features` και αντικαθιστά τον classifier όπως στο Ερώτημα 1, και στη συνέχεια ξεπαγώνει τα `features[16]`, `features[17]` και `features[18]`. Οι παγωμένες παράμετροι δεν λαμβάνουν gradients. Όπως στο αρχικό script, το `model.train()` ενημερώνει και τα στατιστικά BatchNorm των παγωμένων επιπέδων.
- Στο Ερώτημα 8 όλες οι ακρίβειες υπολογίζονται εκ νέου από τις αποθηκευμένες προβλέψεις, χωρίς εκπαίδευση ή εκτέλεση μοντέλων. Οι δύο ακριβέστερες διαμορφώσεις συγκρίνονται με ακριβή αμφίπλευρο έλεγχο McNemar στις εκτελέσεις με seed 0, με αντιστοίχιση μέσω του δείκτη στο MNIST test set και α = 0,05. Τα υπόλοιπα seeds αναφέρονται χωριστά και δεν συνενώνονται.

Αλλαγές σε σχέση με τα αρχικά scripts: καμία στις αρχιτεκτονικές, τις υπερ-παραμέτρους, τη συνάρτηση loss, τον optimizer, τους μετασχηματισμούς ή τους data loaders. Το πλαίσιο πειραμάτων προσθέτει χρονομέτρηση, seeds, επαναλήψεις, warm-up και αρχεία αποτελεσμάτων. Ο test loader επιστρέφει επίσης τον δείκτη κάθε εικόνας. Το loss ανά epoch του MobileNetV2 αθροίζεται στη συσκευή αντί για κλήση `.item()` σε κάθε βήμα.

Διαφορές μεταξύ PDF και κώδικα: το PDF περιγράφει το CNN ως conv → pool → ReLU, ενώ ο κώδικας εφαρμόζει το ReLU πριν από το pooling. Οι δύο σειρές είναι μαθηματικά ισοδύναμες, και ο κώδικας διατηρείται όπως δόθηκε. Η περιγραφή του MLP στο PDF παραλείπει επίσης το ReLU που περιέχει ο κώδικας.

## Έλεγχοι (tests)

```bash
python -m unittest discover -s tests -t . -v
```

114 unit tests ελέγχουν τη χρονομέτρηση, τη συγκέντρωση αποτελεσμάτων, τα αρχεία προβλέψεων, την πιστότητα των μοντέλων, τον υπολογισμό της επιτάχυνσης, τον έλεγχο του GPU baseline, την εκτέλεση μόνο σε CPU, την καταμέτρηση λαθών, τα ζεύγη σύγχυσης, την αντιστοίχιση δεικτών, τα κοινά λάθη, τη δημιουργία σχημάτων, την αρχιτεκτονική του βελτιωμένου MLP, τη συμπεριφορά του dropout, τη σύγκριση με το baseline, το πλέγμα υπερ-παραμέτρων, τον διαχωρισμό validation, την επιλογή του καλύτερου συνδυασμού, την προστασία από διαρροή του test set, τα εκπαιδεύσιμα επίπεδα και το πλήθος παραμέτρων του MobileNetV2 με fine-tuning, τη διατήρηση της προεπεξεργασίας και των ρυθμίσεων εκπαίδευσής του, τα επίπεδα που επιλέγονται για μερικό fine-tuning, χωρίς gradients για τις παγωμένες παραμέτρους, καθώς και την τελική σύγκριση: φόρτωση των επτά διαμορφώσεων, κατάταξη, αντιστοίχιση προβλέψεων και ακριβή έλεγχο McNemar. Χρησιμοποιούν συνθετικά δεδομένα σε CPU: χωρίς GPU, χωρίς download του MNIST, χωρίς προ-εκπαιδευμένα βάρη. Το GitHub Actions (`.github/workflows/python-tests.yml`) τα εκτελεί με PyTorch μόνο για CPU σε κάθε pull request και push στο `master`.

## Φάκελοι

- `mnist_*.py`: τα αρχικά scripts (αμετάβλητα).
- `a2/`: κοινός κώδικας (`models`, `data`, `timing`, `experiment`, `stats`, `io_utils`, `environment`, `compare`, `misclassification`, `figures`, `search`, `final_comparison`).
- `q1_gpu_baseline.py`, `q2_cpu_comparison.py`, `q3_misclassification_analysis.py`, `q4_enhanced_mlp.py`, `q5_mlp_hyperparameter_search.py`, `q6_mobilenet_full_finetune.py`, `q7_mobilenet_partial_finetune.py`, `q8_final_comparison.py`: scripts για τα Ερωτήματα 1–8.
- `tests/`: unit tests.
- `results/question1/` έως `results/question8/`: αποτελέσματα CSV/JSON ανά ερώτημα, προβλέψεις ανά εικόνα, logs, σχήματα και `SUMMARY.md`.
- `docs/`: η εκφώνηση της εργασίας (PDF).
