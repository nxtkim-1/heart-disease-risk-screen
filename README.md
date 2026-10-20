# heart-disease-risk-screen

Educational heart-disease risk screen on the UCI Cleveland cohort.

## Train/test split

`heartscreen/split.py` holds out 20% of the rows for testing. The split is stratified on the binary target, so both halves keep about the same share of positive cases as the full file. The seed is fixed at 42, so every model trains and scores on the same rows.

```python
from heartscreen.data import features_and_target, load_cleveland
from heartscreen.split import train_test

features, target = features_and_target(load_cleveland())
X_train, X_test, y_train, y_test = train_test(features, target)
```

`SEED` and `TEST_SIZE` are module constants. `tests/test_split.py` checks that the split repeats exactly, that every row lands in one half only, and that the positive rate holds in both halves.

## Baseline and metrics

`python -m heartscreen.train` fits each model on the training rows, scores it on the test rows, and writes `reports/metrics.json`. For now the only model is a majority-class baseline. It ignores the features and predicts the most common label in the training rows, which is no heart disease. Logistic regression and random forest come next and are compared against it.

For each model the file records ROC AUC, sensitivity (the share of positive test rows the model flags), specificity (the share of negative test rows it clears), the 0.5 threshold, and the confusion-matrix counts. `tests/test_train.py` checks every key and fails if the committed file differs from a fresh run, so rerun the script after any change to the models or the split.
