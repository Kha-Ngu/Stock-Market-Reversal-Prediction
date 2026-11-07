# ReversalWatch — research methodology and limitations

## Research question
Can indicators observable in public US equity data help identify statistically elevated probabilities of a large move opposite the preceding trend? Does adding market-context data improve out-of-time predictions relative to a price/volume baseline?

## Units of analysis
One stock × one trading-date (end-of-day). Ten stocks; GOOG Class C used in place of planned GOOGL Class A.

## Features in the trained model
`mom5`, `mom10`, `mom20`, `mom60`, `volume_ratio`, `volume_z20`, `volatility20`, `ma_gap20`, `ma_gap50`, `drawdown20`, `spy_mom20`, `qqq_mom20`, `rel_spy20`, `vix`, `vix_change5`.

Each historical value uses that day's or prior data only. Rolling volume reference uses the *preceding* 20 days; date-appropriate price and index data are aligned by trading day. Feature transformations (`SimpleImputer`, `StandardScaler`) are fitted inside training pipelines.

Daily French factor files are parsed for future investigation but excluded from the trained models because distribution timestamps are not established for a fully point-in-time join. Short volume files for only five days in November 2026 are parsed for UI display, but not trained on.

## Reversal event design
Let P(t) denote adjusted closing stock price. Trend preceding date t is `upward` if `P(t)/P(t-20)-1 >= 0.05`, `downward` if `<= -0.05`, else `neutral`. For an upward trend, y(t,H)=1 if any adjusted close during the next H trading sessions is <= 0.92 P(t). For a downward trend, y(t,H)=1 if any future adjusted close is >= 1.08 P(t). H is 10, 20, 30, or 60. Neutral observations are not assigned an event label. We label only matured future windows.

Two trend-specific binary models are built for each horizon. The trend labels (`down` means downward reversal, i.e. prior price momentum UP) are internal codes; these are explained clearly on the dashboard.

The thresholds are experimental and should be assessed for sensitivity; they are not a finance-industry standard. An event can happen at any time within the horizon; forecasts at different horizons are not independent.

## Validation
Train: before 2023-01-01; validation: 2023–2024; test: 2025 onward through the last matured horizon. TRAIN label outcomes must end strictly before the validation start; VALID outcomes must end strictly before test start. Models use time-ordered splits with a label-window purge. Model choice uses validation average precision; the locked test set reports ROC-AUC, AP, accuracy, precision, recall, F1, Brier, and confusion matrices. A full walk-forward cross-validation loop is not included; this remains a limitation.

## Algorithms
- Supervised: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting, k-Nearest Neighbors, Gaussian Naive Bayes.
- Auxiliary continuous: Ridge regression on max opposite-direction 20-day price move.
- Feature selection: SelectKBest F-score, wrapper Recursive Feature Elimination (RFE), embedded Random Forest importance.
- Dimensionality reduction: PCA and explicit PCA-vs-unreduced supervised comparison.
- Unsupervised: K-Means (four clusters), Isolation Forest anomaly scores, cosine similarity against earlier observations.
- Controlled ablation: price-and-volume-only versus market-context features using the same splits.

## Known limitations
- Only ten retrospectively selected stocks; survivorship selection biases conclusions. The panel omits delisted stocks.
- Highly overlapping consecutive observation labels create statistical dependence; raw sample counts are not independent events.
- No feature-level short interest or institutional ownership enters the trained models because broad historical point-in-time coverage was unavailable. The main contribution is therefore NOT proven to capture hedge-fund positioning.
- Historical predictive probabilities are not assumed to be calibrated. Inspect Brier scores and perform probability calibration before treating probabilities as reliably interpretable frequencies.
- Only an initial fixed-period split is implemented; rolling walk-forward validation and confidence intervals remain future work.
- The completed application is an academic classifier, not a guarantee of forward stock movement or a trading system.

## Proposed follow-on experiments
- Reconcile SEC 13F holdings and augment only after public filing timestamps, amendments, and duplicate managers are resolved.
- Add listed-stock historical short interest with settlement AND dissemination date from an entitled source. Refit and re-evaluate on the strictly identical test window.
- Bootstrap block confidence intervals and multi-window rolling evaluation; assess alternative thresholds; investigate calibration via validation-only Platt/isotonic fit.
