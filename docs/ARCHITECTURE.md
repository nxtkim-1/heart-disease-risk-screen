# heart-disease-risk-screen: architecture and decision record

The constraint: if the data file changes, the run stops; and every number in the repository comes from the code in the repository.

## System walk

The processed Cleveland file (303 patients, 13 features) is hash-checked and notice-checked before parsing. Every row is validated against a typed schema; the first bad row stops the load with its line number. One stratified split (20%, seed 42) produces 242 training and 61 test rows used by every model. Preprocessing lives inside each model's Pipeline so nothing is fitted on test rows. A majority-class baseline runs first; the random forest is measured against it. Results are written to reports/metrics.json, which is committed and protected by a test that fails if the committed file differs from a fresh run. The Streamlit app refits the same random forest on the same training rows for interactive queries.

## Decision records

### D1. Cleveland only, not all four hospitals

- Evidence (README design notes): in the other three files `ca` is missing in 96-99% of rows, and disease share runs from 36% (Hungary) to 93% (Switzerland) against 46% in Cleveland; mixing them would mean filling most of two columns and would let a model learn which hospital a row came from. Cost accepted: a smaller sample.

### D2. The label is 0 vs 1-4

- Evidence: grades 3 and 4 have only 35 and 13 patients; the question "disease or not" matches how the UCI page says the data has usually been used. Cost accepted: the model says nothing about severity (listed in limits).

### D3. A loader that stops, loudly

- Choice: SHA-256 and CC BY notice checks before parsing; schema bounds catch damaged rows (they are typo catchers, not clinical ranges); `?` is allowed only in `ca` and `thal`, the only columns with missing values in this file. A changed file stops the run instead of quietly producing new numbers.
- Evidence: tests/test_data.py covers rejected rows, the hash check and the notice check.

### D4. A fixed, stratified split and a baseline first

- Choice: 20% held out, stratified, seed 42. With 303 rows an unstratified split can shift the positive rate by several points; a fixed seed means two models differ only by the models. The majority-class baseline (ROC AUC 0.5, sensitivity 0) exists so a model that barely beats it is visibly not learning much.
- Measured: random forest, test set (61 patients): ROC AUC 0.944, sensitivity 0.929, specificity 0.818 at threshold 0.5; 5-fold CV ROC AUC 0.896 +/- 0.028 (reports/metrics.json). The gap between CV and test AUC is itself a reason to report both.
- Not yet measured: logistic regression and a final threshold rationale (chosen on CV, never the test set) are still planned; the README's results table keeps those cells TBD.

### D5. Metrics that cannot hide a miss

- Choice: ROC AUC, sensitivity and specificity, never accuracy alone; accuracy would let the baseline look fine at 54% while missing every sick patient. Sensitivity is reported on its own because for a screen a missed sick patient costs more than a false alarm.
- Evidence: tests/test_metrics.py checks the metric math on a hand-checked case.

### D6. One-hot, not ordinal, for categorical codes

- Choice: chest-pain type 4 is not "twice" type 2; codes are labels. The encoder learns categories from training rows and maps an unseen code to all zeros instead of erroring. `ca` could be treated as numeric; the schema lists it as categorical, and the README marks that choice as worth revisiting once more models run.

### D7. Numbers in the repo come from the code in the repo

- Choice: reports/metrics.json is committed, and tests/test_train.py fails if a fresh run produces different numbers. tests/test_readme.py keeps the README consistent with the repository state. CI on every push runs pytest plus a gitleaks scan over the full git history (.github/workflows/ci.yml).

## Interview questions this doc should let the owner answer

- The test AUC is 0.944 but CV says 0.896. Which one do you quote, and why?
- Someone edits two rows in the data file and pushes. What fails, and where?
- Why is accuracy not in the results table?
