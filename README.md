# heart-disease-risk-screen

Educational heart-disease risk screen on the UCI Cleveland cohort.

## Train/test split

`train_test` holds out 20% of the rows, stratified on the binary target. `SEED` is 42, so the held-out rows stay the same every run.

```python
from heartscreen.data import features_and_target, load_cleveland
from heartscreen.split import train_test

features, target = features_and_target(load_cleveland())
X_train, X_test, y_train, y_test = train_test(features, target)
```

## Baseline and metrics

`python -m heartscreen.train` fits each model on the training rows, scores it on the test rows, and writes `reports/metrics.json`. For now the only model is a majority-class baseline. It ignores the features and predicts the most common label in the training rows, which is no heart disease. Logistic regression and random forest come next and are compared against it.

For each model the file records ROC AUC, sensitivity (the share of positive test rows the model flags), specificity (the share of negative test rows it clears), the 0.5 threshold, and the confusion-matrix counts. `tests/test_train.py` checks every key and fails if the committed file differs from a fresh run, so rerun the script after any change to the models or the split.
