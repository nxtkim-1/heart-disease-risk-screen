# Setup

How to get the project running on your own machine and check that it works.

## What you need

- Python 3.12. Check with `python --version` (on some systems `python3 --version`).
- Git.
- A terminal. On Windows, PowerShell works.

## 1. Get the code

```bash
git clone https://github.com/nxtkim-1/heart-disease-risk-screen.git
cd heart-disease-risk-screen
```

## 2. Make a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Your prompt should now start with `(.venv)`.

## 3. Install the packages

```bash
pip install -r requirements.txt
```

The versions are pinned, so you get the same ones the tests ran against.

## 4. Run the tests

```bash
pytest
```

All tests should pass. They check the data file's hash, the schema, the train/test split, preprocessing and the metrics file.

## 5. Regenerate the metrics

```bash
python -m heartscreen.train
```

This rewrites `reports/metrics.json`. If `git status` then shows that file as changed, something in your setup differs from the one the committed numbers came from.

## If something fails

| Symptom | Likely cause and fix |
|---|---|
| `python: command not found` | Try `python3`, or install Python 3.12 from python.org. |
| `No module named heartscreen` | You are not in the repo's top folder. Run `cd heart-disease-risk-screen`. |
| `No module named sklearn` or similar | The virtual environment is not active, or step 3 did not finish. Activate it and repeat step 3. |
| `does not match the UCI Cleveland file` | `data/raw/processed.cleveland.data` was changed. Restore it with `git checkout -- data/raw`. |
| Tests fail only on `committed_metrics_match_a_fresh_run` | Package versions differ from `requirements.txt`. Repeat step 3 in a fresh `.venv`. |
