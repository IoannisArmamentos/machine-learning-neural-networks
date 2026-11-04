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
3. Analyse misclassified test images per network and those misclassified by all three.
4. New MLP with two hidden layers (500 and 200, ReLU, dropout 0.2).
5. Original MLP with dropout: choose the dropout probability and hidden size on a 10% validation set.
6. MobileNetV2 with the whole pretrained network fine-tuned.
7. MobileNetV2 with only the last 3 pretrained layers fine-tuned.
8. Summary table of all experiments, best model, statistical significance and the usefulness of the GPU.

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
# Quick pipeline check of either script (truncated runs, written to results/smoke/, not tracked)
python q1_gpu_baseline.py --model all --smoke
python q2_cpu_comparison.py --model all --smoke
```

Both scripts accept `--model mlp cnn mobilenetv2`, `--warmup-batches` (default 20), `--num-workers` (default 0, as in the original scripts), `--out` and `--overwrite`. They refuse to replace existing results without `--overwrite`. `q2_cpu_comparison.py` reads the Question 1 results as its GPU baseline and never writes to them.

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

## Methodology

- Training time covers all epochs, including data loading. Inference time covers one full pass over the test set. Dataset loading and model construction are excluded.
- On the GPU, `torch.cuda.synchronize()` runs before every clock read, so asynchronous work is fully counted.
- Before the timed runs, 20 untimed warm-up batches run on a throw-away model, on both devices. On the GPU they absorb one-time kernel compilation.
- Each run is seeded (weight initialisation and shuffle order). Test predictions are saved with their MNIST test index.
- For Question 2, the GPU is hidden from PyTorch before `torch` is imported, and the script stops if a GPU is still visible.

Changes compared with the original scripts: none to architectures, hyperparameters, loss, optimiser, transforms or data loaders. The harness adds timing, seeding, repetition, warm-up and result files. The test loader also returns each image's index. MobileNetV2's per-epoch loss is accumulated on the device instead of with a per-step `.item()` call.

Differences between the PDF and the code: the PDF lists the CNN as conv → pool → ReLU, but the code applies ReLU before pooling. These are mathematically equivalent, and the code is kept as provided. The PDF's MLP description also omits the ReLU that the code contains.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

27 unit tests cover timing, aggregation, prediction files, model fidelity, speedup calculation, baseline validation and CPU-only enforcement. They use synthetic data on the CPU: no GPU, no MNIST download, no pretrained weights. GitHub Actions (`.github/workflows/python-tests.yml`) runs them with CPU-only PyTorch on every pull request and push to `master`.

## Directories

- `mnist_*.py`: the original scripts (unchanged).
- `a2/`: shared code (`models`, `data`, `timing`, `experiment`, `stats`, `io_utils`, `environment`, `compare`).
- `q1_gpu_baseline.py`, `q2_cpu_comparison.py`: experiment scripts for Questions 1 and 2.
- `tests/`: unit tests.
- `results/question1/`, `results/question2/`: per-question CSV/JSON results, per-image predictions, logs and `SUMMARY.md`.
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
3. Ανάλυση των εικόνων του test set που ταξινομούνται λάθος από κάθε δίκτυο και από τα τρία μαζί.
4. Νέο MLP με δύο κρυμμένα επίπεδα (500 και 200, ReLU, dropout 0.2).
5. Αρχικό MLP με dropout: επιλογή πιθανότητας dropout και διάστασης κρυμμένου επιπέδου με validation set 10%.
6. MobileNetV2 με fine-tuning ολόκληρου του προ-εκπαιδευμένου δικτύου.
7. MobileNetV2 με fine-tuning μόνο των 3 τελευταίων προ-εκπαιδευμένων επιπέδων.
8. Συγκεντρωτικός πίνακας όλων των πειραμάτων, καλύτερο μοντέλο, στατιστική σημαντικότητα και χρησιμότητα της GPU.

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
# Γρήγορος έλεγχος της ροής για κάθε script (περικομμένες εκτελέσεις, στο results/smoke/, εκτός αποθετηρίου)
python q1_gpu_baseline.py --model all --smoke
python q2_cpu_comparison.py --model all --smoke
```

Και τα δύο scripts δέχονται `--model mlp cnn mobilenetv2`, `--warmup-batches` (προεπιλογή 20), `--num-workers` (προεπιλογή 0, όπως στα αρχικά scripts), `--out` και `--overwrite`. Δεν αντικαθιστούν υπάρχοντα αποτελέσματα χωρίς `--overwrite`. Το `q2_cpu_comparison.py` διαβάζει τα αποτελέσματα του Ερωτήματος 1 ως GPU baseline και δεν γράφει ποτέ σε αυτά.

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

## Μεθοδολογία

- Ο χρόνος εκπαίδευσης καλύπτει όλα τα epochs, μαζί με τη φόρτωση δεδομένων. Ο χρόνος εφαρμογής καλύπτει ένα πλήρες πέρασμα του test set. Η φόρτωση του dataset και η δημιουργία του μοντέλου εξαιρούνται.
- Στη GPU εκτελείται `torch.cuda.synchronize()` πριν από κάθε ανάγνωση του χρονομέτρου, ώστε να μετράται πλήρως η ασύγχρονη εκτέλεση.
- Πριν από τις χρονομετρούμενες εκτελέσεις, εκτελούνται 20 μη χρονομετρούμενα warm-up batches σε ένα μοντέλο που απορρίπτεται, και στις δύο συσκευές. Στη GPU απορροφούν την εφάπαξ μεταγλώττιση των kernels.
- Κάθε εκτέλεση έχει seed (αρχικοποίηση βαρών και σειρά ανακατέματος). Οι προβλέψεις στο test set αποθηκεύονται μαζί με τον δείκτη κάθε εικόνας στο MNIST test set.
- Στο Ερώτημα 2 η GPU αποκρύπτεται από το PyTorch πριν από το import του `torch`, και το script σταματά αν η GPU παραμένει ορατή.

Αλλαγές σε σχέση με τα αρχικά scripts: καμία στις αρχιτεκτονικές, τις υπερ-παραμέτρους, τη συνάρτηση loss, τον optimizer, τους μετασχηματισμούς ή τους data loaders. Το πλαίσιο πειραμάτων προσθέτει χρονομέτρηση, seeds, επαναλήψεις, warm-up και αρχεία αποτελεσμάτων. Ο test loader επιστρέφει επίσης τον δείκτη κάθε εικόνας. Το loss ανά epoch του MobileNetV2 αθροίζεται στη συσκευή αντί για κλήση `.item()` σε κάθε βήμα.

Διαφορές μεταξύ PDF και κώδικα: το PDF περιγράφει το CNN ως conv → pool → ReLU, ενώ ο κώδικας εφαρμόζει το ReLU πριν από το pooling. Οι δύο σειρές είναι μαθηματικά ισοδύναμες, και ο κώδικας διατηρείται όπως δόθηκε. Η περιγραφή του MLP στο PDF παραλείπει επίσης το ReLU που περιέχει ο κώδικας.

## Έλεγχοι (tests)

```bash
python -m unittest discover -s tests -t . -v
```

27 unit tests ελέγχουν τη χρονομέτρηση, τη συγκέντρωση αποτελεσμάτων, τα αρχεία προβλέψεων, την πιστότητα των μοντέλων, τον υπολογισμό της επιτάχυνσης, τον έλεγχο του GPU baseline και την εκτέλεση μόνο σε CPU. Χρησιμοποιούν συνθετικά δεδομένα σε CPU: χωρίς GPU, χωρίς download του MNIST, χωρίς προ-εκπαιδευμένα βάρη. Το GitHub Actions (`.github/workflows/python-tests.yml`) τα εκτελεί με PyTorch μόνο για CPU σε κάθε pull request και push στο `master`.

## Φάκελοι

- `mnist_*.py`: τα αρχικά scripts (αμετάβλητα).
- `a2/`: κοινός κώδικας (`models`, `data`, `timing`, `experiment`, `stats`, `io_utils`, `environment`, `compare`).
- `q1_gpu_baseline.py`, `q2_cpu_comparison.py`: scripts πειραμάτων για τα Ερωτήματα 1 και 2.
- `tests/`: unit tests.
- `results/question1/`, `results/question2/`: αποτελέσματα CSV/JSON ανά ερώτημα, προβλέψεις ανά εικόνα, logs και `SUMMARY.md`.
- `docs/`: η εκφώνηση της εργασίας (PDF).
