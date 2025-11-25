# Amazon Reviews Deception Detection

This project studies suspicious review activity on Amazon (All_Beauty category of the **McAuley-Lab/Amazon-Reviews-2023** dataset) by mixing behavioral and textual indicators.

## Data Processing Notebook

Use `Data.ipynb` to run the entire preprocessing flow inside Jupyter:

1. Configure the category, sample size, and output directory.
2. Load a subset of the Hugging Face dataset with `datasets`.
   - If the active kernel cannot import `datasets` because of the macOS Anaconda/`pyarrow` binary mismatch, set `EXTERNAL_PYTHON` (at the top of the notebook) to the interpreter where `pip install datasets` succeeded (e.g., `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`). The notebook will spawn that interpreter via `subprocess` to download the raw CSV and reuse the cached file on subsequent runs.
3. Clean records (drop missing identifiers, empty reviews, rating outliers, duplicates).
4. Engineer textual indicators (length statistics, lexical diversity, sentiment polarity, punctuation/uppercase ratios).
5. Aggregate behavioral metrics per user (review frequency, rating variance, verified purchase ratio, helpful vote trends, posting cadence).
6. Save the enriched review-level table (`data/amazon_reviews_cleaned.csv`) and user-level summary (`data/user_behavior_features.csv`).

> **Dependencies:** `datasets`, `pandas`, `numpy`, `textblob` (optional; sentiment falls back to zero if absent). Install them in the notebook cell with `pip install datasets pandas numpy textblob seaborn matplotlib`. If you rely on the external interpreter fallback, ensure the same packages are installed there as well.

## Next Steps

- **Exploratory checks:** Visualize behavioral/textual feature distributions (e.g., KDE plots, time series) to confirm the cleaning worked and to spot heavy tails that might indicate anomalies.
- **Unsupervised models:** Train Isolation Forest and One-Class SVM on a standardized feature matrix built from both review-level and user-level metrics. Compare anomaly scores and inspect overlapping outliers.
- **Manual validation:** Sample top-ranked suspicious reviews/users, inspect their text/behavior, and refine feature thresholds.
- **Iterative refinement:** Incorporate additional cues (burstiness, review duplication similarity, sentiment inconsistency) and evaluate their effect on anomaly detection quality.
