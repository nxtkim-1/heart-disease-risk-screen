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

## Preprocessing

`heartscreen.preprocess.preprocessor()` returns an unfitted scikit-learn `ColumnTransformer`. Numeric columns are filled with the median and standardized. Binary columns are filled with the most common value and kept as 0/1. Categorical columns are filled with the most common value and one-hot encoded; `ca` and `thal` are the only columns with gaps in the file. `with_preprocessing(model)` puts it in front of a model in one `Pipeline`. Calling `fit` on the training rows fits the imputers, the scaler and the encoder on those rows only, so the test rows are never seen. `tests/test_preprocess.py` checks that the fitted medians, means and categories come from the training rows, and that the scaler means differ from the means over all rows.
