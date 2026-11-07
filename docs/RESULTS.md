# ReversalWatch — Reproducible experimental results

All metrics below were computed from the **user-provided data** using chronological train/validation/test splits. Results are descriptive historical test outcomes, not real-time forecast promises.

## Selected models (selected by validation average precision, not test set)

| Horizon | Event direction | Model | Test samples | Test event rate | Test F1 | Test average precision |
|---|---|---|---:|---:|---:|---:|
| 10 | Down (prior rise) | Random Forest | 1,679 | 0.204 | 0.401 | 0.331 |
| 10 | Up (prior fall) | Logistic Regression | 1,164 | 0.296 | 0.488 | 0.564 |
| 20 | Down (prior rise) | Logistic Regression | 1,656 | 0.327 | 0.520 | 0.468 |
| 20 | Up (prior fall) | Logistic Regression | 1,137 | 0.456 | 0.625 | 0.687 |
| 30 | Down (prior rise) | Gaussian Naive Bayes | 1,611 | 0.390 | 0.489 | 0.496 |
| 30 | Up (prior fall) | Logistic Regression | 1,133 | 0.553 | 0.709 | 0.755 |
| 60 | Down (prior rise) | Logistic Regression | 1,500 | 0.489 | 0.618 | 0.622 |
| 60 | Up (prior fall) | Logistic Regression | 1,064 | 0.771 | 0.864 | 0.903 |

## Research experiments

- Six classifier families: Logistic Regression, Random Forest, Histogram Gradient Boosting, Decision Tree, KNN, Gaussian Naive Bayes.
- PCA feature reduction, filter/wrapper/embedded selection, K-Means, Isolation Forest, cosine similarity.
- Original feature-set versus price/volume-only controlled ablation: `artifacts/ablation_pca_results.csv`.
- Continuous 20-day adverse-move Ridge regression in `artifacts/regression_results.csv`.
- Raw complete metrics available in `artifacts/metrics.csv`; test-observation forecasts in `artifacts/test_predictions.csv`.

## Dataset coverage and limits

- 22,250 stock-day rows for 10 stocks through November 6, 2026.
- VIX source through November 5; French data through September 2026 (not used in supervised training).
- FINRA Consolidated NMS only five November 2026 dates, 50 relevant stock-day observations, kept for illustrative UI display.
- SEC 13F selected CUSIP holdings are descriptive; source archive filing dates are incomplete. Summed shares are not true percentage ownership.
- SEC available matched filing-date rows: 290,176 / 726,688; other records excluded from dated summary.
- Nasdaq premium historical short-interest access was not available in this environment. No artificial short-interest values were created.

## Interpretation

Longer horizons tend to have higher historical event rates because an event can occur at any time in the window. Do not compare raw accuracy/F1 across horizons without considering event prevalence. This restricted universe, parameter choices, historical nonstationarity and possible stock-selection bias restrict generalizability.

## Replication

Run `python -m reversalwatch.data`, `python -m reversalwatch.train`, `python -m reversalwatch.analysis`, `python -m reversalwatch.experiments`, and `python -m pytest -q`.
