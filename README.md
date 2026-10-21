# heart-disease-risk-screen

An educational heart-disease risk screen on the UCI Cleveland data. It is not a medical device and gives no medical advice.

Status: in progress, target release `v1.0` on Nov 14, 2026. The loader, the train/test split, the preprocessing pipeline and a majority-class baseline are done. Logistic regression, random forest, figures and a local Streamlit app come next. Results below are TBD until those models run.

## Data

`data/raw/processed.cleveland.data` is the processed Cleveland file from the UCI Heart Disease dataset: 303 patients, 13 features and the label `num`.

| Item | Value |
|---|---|
| Rows | 303 |
| Features | 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) |
| Label | `num` 0 is no disease; 1 to 4 become one positive class |
| Class balance | 164 negative, 139 positive |
| Missing values | `ca` in 4 rows, `thal` in 2 rows |

Source: https://archive.ics.uci.edu/dataset/45/heart+disease. The file is shared under CC BY 4.0. Credit and the file hash are in `data/SOURCES.md`.

## How it works

1. `heartscreen/schema.py` lists the 14 columns with their type, allowed codes or bounds, and whether a column may be missing.
2. `heartscreen/data.py` checks the file's SHA-256 and the attribution notice, then parses each row against the schema. A row with the wrong field count, an unknown code, an out-of-bounds value or a `?` outside `ca` and `thal` stops the load with the line number.
3. `features_and_target` returns the 13 features and a 0/1 label.
4. `heartscreen/split.py` holds out 20% of rows, stratified on the label, with seed 42. That gives 242 training rows and 61 test rows.
5. `heartscreen/preprocess.py` builds the preprocessing step. `with_preprocessing(model)` puts it in front of a model in one scikit-learn `Pipeline`.
6. `python -m heartscreen.train` fits each model on the training rows, scores it on the test rows and writes `reports/metrics.json`.

```python
from heartscreen.data import features_and_target, load_cleveland
from heartscreen.split import train_test

features, target = features_and_target(load_cleveland())
X_train, X_test, y_train, y_test = train_test(features, target)
```

## Design decisions

Cleveland only, out of the four hospitals in the UCI dataset. The UCI page says published work has used the Cleveland file. In the other three files, `ca` is missing in 96% to 99% of rows, and the share of patients with disease runs from 36% (Hungary) to 93% (Switzerland), against 46% in Cleveland. Mixing them would mean filling in most of two columns and would let a model pick up which hospital a row came from. One hospital keeps the question simple, at the cost of a smaller sample.

The label is 0 versus 1 to 4. `num` grades narrowing from 0 to 4, but the grades 3 and 4 have only 35 and 13 patients. The question here is "disease or not", which matches how the UCI page says the data has usually been used.

The loader checks a hash and the credit notice. If the data file changes, every number downstream changes, so a changed file should stop the run instead of producing new numbers quietly. The notice check keeps the CC BY credit next to the file.

Bounds in the schema catch damaged rows. They are not clinical reference ranges. A blood pressure of 0 or an age of 300 is a typo, and the loader says which line it is on.

`?` is allowed only in `ca` and `thal`, because those are the only columns with missing values in this file. A `?` anywhere else means the file is not the one described in `data/SOURCES.md`.

The test set is fixed: 20%, stratified, seed 42. A fixed seed means every model is scored on the same 61 patients, so any difference between two models comes from the models themselves. Stratifying keeps the positive rate in train and test close to the full data (45.9%). With only 303 rows, an unstratified split can shift that rate by several points.

The first model is a majority-class baseline. It ignores the features and calls every patient negative, because negatives are the larger class. It scores ROC AUC 0.5 and sensitivity 0. Any real model has to beat this, and a model that barely does is not learning much.

Three metrics instead of accuracy. ROC AUC measures how well the model ranks sick patients above healthy ones at every threshold. Sensitivity is the share of sick test patients the model flags; specificity is the share of healthy ones it clears. For a screen, a missed sick patient costs more than a false alarm, so sensitivity is reported on its own. Accuracy would let the baseline look fine at 54% while it misses every sick patient.

The threshold is 0.5 for now. The threshold decides the trade between missed patients and false alarms. TBD: the final threshold and the reason for it, chosen on cross-validation results, never on the test set.

`metrics.json` is committed and tested. `tests/test_train.py` fails if the committed file differs from a fresh run, so the numbers in the repo always come from the code in the repo.

Preprocessing lives inside the model's `Pipeline`. Numeric columns get the training median for gaps and are standardized. Binary columns get the most common training value. Categorical columns (cp, restecg, slope, ca, thal) get the most common training value and are one-hot encoded. Because the fill values, the scaler and the encoder are part of the pipeline, `fit` on the training rows learns them from those rows only, and the test rows cannot leak into training. `tests/test_preprocess.py` checks that the fitted medians, means and categories match the training rows and that the scaler means differ from the means over all 303 rows.

Categorical codes are one-hot encoded instead of left as numbers. Chest pain type 4 (asymptomatic) is not "twice" type 2; the codes are labels. `ca` counts vessels from 0 to 3, so it could be treated as a number; the schema lists it as categorical, and that choice is worth revisiting once the models run. The encoder learns its categories from the training rows and ignores a code it has not seen, so an unseen code becomes all zeros instead of an error.

The median fills numeric gaps because it is not pulled by a few extreme values. In this file no numeric column has gaps, so the fill matters only for rows typed into the app.

## Results

Test set: 61 patients (33 negative, 28 positive), threshold 0.5. Numbers come from `reports/metrics.json`.

<!-- results:start -->
| Model | ROC AUC | Sensitivity | Specificity | 5-fold CV ROC AUC |
|---|---|---|---|---|
| Majority class | 0.50 | 0.00 | 1.00 | TBD |
| Logistic regression | TBD | TBD | TBD | TBD |
| Random forest | TBD | TBD | TBD | TBD |
<!-- results:end -->

Figures (ROC curves, confusion matrices): TBD.

## Demo

TBD: screenshots of the local Streamlit app.

## Run it

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m heartscreen.train      # rewrites reports/metrics.json
```

## Tests

`pytest` runs on every push and pull request in GitHub Actions. The tests cover the schema and label, rejected rows, the hash and notice checks, the split (repeatable, disjoint, stratified, right size), preprocessing fitted on training rows only, the metric math on a hand-checked case, and the committed `metrics.json`.

## Limitations

- Educational only. Not a medical device, not medical advice, not validated for clinical use.
- 303 patients from one hospital, in a dataset published in 1989. Results say nothing about other hospitals or today's patients.
- Every label comes from an angiogram, so every patient was someone sent for one. This is not the general population.
- 206 of the 303 patients are male, so results for women rest on 97 people.
- The test set has 61 patients. One patient moving changes sensitivity by about 3.6 points, which is why CV spread is reported too.
- Grades 1 to 4 are merged, so the model says nothing about severity.

## Next

Part B will ask whether MedGemma, zero-shot and then fine-tuned, can match these baselines on the same split.

## License and credit

Code: MIT (see `LICENSE`). Data: CC BY 4.0, from Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
