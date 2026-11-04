# Question 3: Misclassification analysis

For each network, one specific run was analysed: **Question 1 run 1 (seed 0)** for MLP, CNN and
MobileNetV2. The same run was chosen for every model, fixed in advance rather than picked by its
results. The analysis uses the saved per-image GPU predictions (`results/question1/<model>/predictions_run1_seed0.csv`).
No model was retrained and no inference was rerun. The images in the figures are the original
MNIST test images, matched to the predictions by their MNIST test index.

## Misclassifications per model

| Model | Test images | Misclassified | Accuracy |
|---|---|---|---|
| MLP | 10,000 | **302** | 96.98 % |
| CNN | 10,000 | **100** | 99.00 % |
| MobileNetV2 | 10,000 | **394** | 96.06 % |

The indices of all misclassified images are in `misclassified.csv` (796 rows).

## Errors by true digit: count and rate

Error rate = errors / test images of that digit. The test set is not balanced: there are 892 fives
and 1,135 ones. A raw count therefore does not show how hard a digit is; the rate is the fairer
comparison between digits.

| Digit | Test images | MLP errors | MLP rate | CNN errors | CNN rate | MobileNetV2 errors | MobileNetV2 rate |
|---|---|---|---|---|---|---|---|
| 0 | 980 | 12 | 1.22 % | 7 | 0.71 % | 32 | 3.27 % |
| 1 | 1135 | 17 | 1.50 % | 1 | 0.09 % | 11 | 0.97 % |
| 2 | 1032 | **45** | **4.36 %** | 11 | 1.07 % | **69** | **6.69 %** |
| 3 | 1010 | 36 | 3.56 % | 16 | 1.58 % | 48 | 4.75 % |
| 4 | 982 | 26 | 2.65 % | 6 | 0.61 % | 34 | 3.46 % |
| 5 | 892 | 33 | 3.70 % | 6 | 0.67 % | 49 | 5.49 % |
| 6 | 958 | 33 | 3.44 % | 15 | 1.57 % | 31 | 3.24 % |
| 7 | 1028 | 41 | 3.99 % | 8 | 0.78 % | 46 | 4.47 % |
| 8 | 974 | 26 | 2.67 % | 9 | 0.92 % | 39 | 4.00 % |
| 9 | 1009 | 33 | 3.27 % | **21** | **2.08 %** | 35 | 3.47 % |

- **MLP:** the highest error rates are for 2 (4.36 %), 7 (3.99 %) and 5 (3.70 %). Ranked by raw
  count, the order is 2, 7, 3 instead. Digit 5 has fewer test images, so its 33 errors are a higher
  rate than the 33 errors for 6 and for 9. The MLP most often wrongly *predicts* 8 (72 times) and
  9 (52 times).
- **CNN:** the highest rates are for 9 (2.08 %), 3 (1.58 %) and 6 (1.57 %). All counts are small
  (1–21 per digit), so the ranking is uncertain. Its most frequent wrong prediction is 5 (23 times).
- **MobileNetV2:** the highest rates are for 2 (6.69 %), 5 (5.49 %) and 3 (4.75 %). Its most
  frequent wrong prediction is 9 (73 times).
- **Digit 1** has the lowest or almost the lowest error rate for every model. Digit 2 (MLP,
  MobileNetV2) and digit 3 (CNN, MobileNetV2) are each among the three highest rates for two of
  the three models.

## Most frequent confusions (true → predicted)

| Rank | MLP | CNN | MobileNetV2 |
|---|---|---|---|
| 1 | 2 → 8 (17) | 3 → 5 (7) | 5 → 3 (25) |
| 2 | 7 → 9 (16) | 6 → 5 (6) | 3 → 5 (22) |
| 3 | 3 → 8 (15) | 9 → 5 (6) | 7 → 9 (19) |
| 4 | 4 → 9 (13) | 9 → 7 (5) | 4 → 9 (18) |
| 5 | 9 → 4 (13) | 0 → 8 (4) | 2 → 5 (14) |

Counting both directions together, the most confused digit pairs are:

| Model | Top pairs (both directions) |
|---|---|
| MLP | 4/9 (26), 3/8 (21), 7/9 (18) |
| CNN | 3/5 (9), then 0/8, 4/9, 5/6 and 5/9 (7 each) |
| MobileNetV2 | 3/5 (47), 2/3 (27), 2/7 (25) |

The pairs involve similarly shaped digits: 4/9, 7/9, 3/5, 3/8 and 2/7. MobileNetV2's 3/5
confusion alone accounts for 47 of its 394 errors (12 %).

`confusion_pairs.csv` lists every off-diagonal pair that occurs, i.e. the whole confusion matrix
outside the diagonal, with the share of that digit's test images and the share of the model's errors.

## Misclassified by all three models

**29 test images are misclassified by all three networks.** For 17 of them, all three predict the
same wrong digit. Their true digits are spread across all classes (six 6s, four 7s, four 9s, at
most three of any other digit). The indices and each model's prediction are in
`common_errors.csv`, and all 29 images are shown in `figures/common_errors.png`. Visually, many
are atypical or ambiguous handwriting, e.g. #2118 (a 6 written like a U or 0) and #6576 (a 7
resembling a 1).

Overlap between models (Jaccard = shared errors / all distinct errors):

| Models | Errors | Shared | Jaccard |
|---|---|---|---|
| MLP + CNN | 302 / 100 | 59 | 0.17 |
| MLP + MobileNetV2 | 302 / 394 | 108 | 0.18 |
| CNN + MobileNetV2 | 100 / 394 | 45 | 0.10 |
| All three | 302 / 100 / 394 | 29 | 0.05 |

The models mostly make *different* mistakes. More than half of the CNN's errors (59 of 100) are
also MLP errors, but only 29 images fool all three.

## Figures (`figures/`)

| File | Content |
|---|---|
| `<model>_errors_by_digit.png` | one row per true digit 0–9: the first 6 misclassified images (lowest test index) |
| `<model>_top_confusions.png` | the 5 most frequent confusion pairs: up to 6 examples each (lowest test index) |
| `common_errors.png` | all 29 images misclassified by all three models, with each model's prediction |
| `figures_manifest.csv` | every tile shown: figure, row/column, pixel box, MNIST index, true and predicted labels |

The examples are chosen by **lowest test index**, a fixed rule that does not depend on how
striking an error looks. Each caption gives the MNIST test index and `true → predicted`. Images
are shown at 3× size with nearest-neighbour scaling, so the pixels are the original ones.

## Validation

- **Predictions:** each model has exactly 10,000 predictions with indices 0–9999. Their true
  labels equal the MNIST test labels at the same index, and the three models are aligned index by
  index.
- **Error counts:** 302 / 100 / 394 equal 10,000 − `n_correct` of Question 1 run 1 for each
  model. The SHA-256 hash of every prediction file used is recorded in `summary.csv`.
- **Tables:** the per-digit counts, confusion pairs, list of misclassified images and the 29
  common errors were recomputed independently, and they match.
- **Figures:** each of the 291 tiles in the 7 figures was cropped from the saved PNG. It is
  pixel-identical to the MNIST test image at the index in `figures_manifest.csv`, and its labels
  match the prediction file.
- **Reproducibility:** re-running the script gives byte-identical outputs.

## Limitations

- **One run per model**, as the assignment asks. Another seed would change some of the individual
  misclassified images and pair counts. In Question 1 the error counts varied by up to 16 images
  (CNN), 11 (MLP) and 34 (MobileNetV2) between seeds.
- **Small counts are noisy.** For a rate around 4 % on about 1,000 images, one standard error is
  about 0.6 percentage points. Differences of that size between digits, and especially the CNN's
  single-digit counts, should not be over-interpreted.
- **Visual remarks are qualitative.** They describe the shown examples, not a measured property
  of the digits.

## Generated files

| Path | Content |
|---|---|
| `results/question3/summary.csv` | per model: run, seed, source file, SHA-256, test images, correct, errors, accuracy |
| `results/question3/misclassified.csv` | every misclassified image: model, MNIST index, true and predicted label |
| `results/question3/errors_by_digit.csv` | per model and digit: test images, errors, error rate, share of errors, times wrongly predicted |
| `results/question3/confusion_pairs.csv` | per model: all (true, predicted) pairs that occur, ranked, with percentages |
| `results/question3/common_errors.csv` | the 29 shared errors: index, true label, each model's prediction, whether all three agree |
| `results/question3/error_overlap.csv` | shared errors and Jaccard for each model pair and for all three |
| `results/question3/figures/` | 7 PNG figures and `figures_manifest.csv` |

## Reproduce

```bash
source .venv/bin/activate
python q3_misclassification_analysis.py --overwrite      # a few seconds; reads Question 1 predictions only
python -m unittest discover -s tests -t . -v
```
