# Market Reversal Predictor

An end-to-end, locally runnable **CSS 486 machine-learning course project** for studying large stock-price moves opposite to an established trend. Uses Python, scikit-learn and a Streamlit dashboard. The included outputs were generated from the user's actual uploaded stock, benchmark and volatility data, NOT synthetic model scores.

**This is academic research, not investment advice.** Historical metrics do not guarantee future performance. The downloaded ZIP contains **no personal FINRA or Nasdaq API keys**.

## Quick start — school Linux machine

Requires Python **3.10+** (recommended 3.11 or 3.12), sufficient storage (~250 MB plus Python packages), and network access to install packages initially. No paid market-data API access is required for the included saved models.

```bash
unzip ReversalWatch_Project.zip
cd ReversalWatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501` in the same computer's web browser.

### SSH remote Linux example

From **your local laptop**, after you have SSH access to your school server:

```bash
ssh -L 8501:127.0.0.1:8501 YOUR_USERNAME@YOUR_SCHOOL_HOST
```

Inside that SSH session, activate the environment and run:

```bash
cd ~/ReversalWatch
source .venv/bin/activate
streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Then open `http://localhost:8501` on your **local laptop**. Replace placeholders with your actual school account and host. Follow your university's SSH/port forwarding policies. Streamlit binds to loopback only; it is not intentionally exposed to the public internet.

### Windows / macOS

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS uses the same commands as Linux except SSH is not necessary for local execution.

## Dashboard contents

1. **Overview & Forecast:** stock selector, 10/20/30/60-day horizons, interactive price/volume/VIX hover data, end-of-day directional reversal estimate and clear limitations. Click `ⓘ` calculation controls for actual arithmetic and source-input values; click the forecast explanation for its feature vector, prediction method, and tested performance.
2. **Backtesting:** historical held-out model probabilities vs actual reversal events, per-observation prediction date, outcome and CSV export.
3. **Model Comparison:** six classifiers, precision/recall/F1/ROC-AUC/AP/Brier scores and confusion matrices.
4. **Feature Analysis:** input distributions and correlations, feature filter/wrapper/embedded selection, PCA explanation, PCA-vs-baseline ablation experiments.
5. **Similarity & Regimes:** PCA two-dimensional map, K-Means market condition clusters, Isolation Forest anomaly detection, cosine similarity to past market-feature vectors.
6. **Methodology:** feature glossary, split design, assumptions, references, limitations and reproduction commands.

Hoverable graph points are powered by Plotly; interactive controls are powered by Streamlit. No live market requests are made when opening the app.

## Files and reproducibility

```
ReversalWatch/
├── app.py
├── reversalwatch/
│   ├── data.py           # raw → features → future event labels
│   ├── train.py          # six ML classifiers, time purging, real test metrics
│   ├── analysis.py       # PCA, RFE, cosine, KMeans, IsolationForest
│   ├── experiments.py    # price-only vs market, PCA ablation, regression
│   └── explain.py        # indicator math and transparent source inputs
├── data/
│   ├── raw/              # original uploaded CSVs and 5 FINRA TXT samples
│   └── processed/        # reproducible panel and filtered FINRA samples
├── artifacts/            # model joblib files, metrics and predictions
├── scripts/              # refresh and optional authenticated API scripts
├── tests/                # pytest tests
├── docs/                 # research, references and dataset caveats
├── requirements.txt
├── .env.example
└── .gitignore
```

**Rebuild derived datasets and models** (may take several minutes depending on CPU):

```bash
python -m reversalwatch.data
python -m reversalwatch.train
python -m reversalwatch.analysis
python -m reversalwatch.experiments
python -m pytest -q
streamlit run app.py
```

The ZIP includes precomputed outputs, so rebuilding is optional just to demo the UI. Training uses a fixed 42 seed. Models are stored as joblib files and should only be opened from **trusted ZIPs** (never load arbitrary serialized model files).

## Stock universe

10 stocks: NVDA, AMD, AVGO, MSFT, **GOOG** (Class C rather than the previously planned GOOGL), AMZN, META, AAPL, TSLA, INTC. SPY, QQQ, DIA provide market benchmarks, not extra prediction targets.

Stock CSVs supplied by user span January 2018 through November 6, 2026. The FRED VIX file ends November 5, 2026; its latest prior observation is used for subsequent dates. Source French factor CSVs end September 2026 and are excluded from the trained predictor due to data-availability timing uncertainty.

## What the model actually predicts

A stock must show at least **+5% or −5% change over the prior 20 trading sessions**. If upward, event=1 when the adjusted closing price falls **8% or more** below its prediction-date adjusted close at any point in the next H trading sessions. If downward, event=1 when the stock rises **8% or more**. H ∈ {10,20,30,60} trading days. Neutral trends yield no directional prediction. Thresholds are declared research choices, not industry truths.

This models an **event occurring at any point within a future window**, rather than predicting the exact future price or the reversal date.

## Stock-positioning and API availability

The project retains FINRA Consolidated NMS five-day examples, including the correct short-sale-volume ratio (short volume / FINRA reported total volume). These do **NOT** measure outstanding short interest, and are NOT used as training features due to insufficient historical observations.

- `scripts/fetch_finra_bulk.py`: downloads public Consolidated NMS daily files starting in Aug 2018, filters the 10 tickers, and appends results. Internet required; retry/error logs are created.
- `scripts/fetch_finra.py`: authenticates to FINRA using OAuth Client ID + Client Secret from `.env`, probes datasets only as your account permits. The consolidated short interest API may be limited to OTC securities.
- `scripts/fetch_nasdaq.py`: probes Nasdaq Data Link premium NSHRT. If not entitled, it exits without pretending it succeeded.
- SEC 13F: the provided compact selected-security extracts include only partially matched filing dates; these are **descriptive**, not verified institutional ownership percentages and not model inputs. Access/aggregation and amendments must be fully reconciled before training with them.

To refresh an authenticated source:

```bash
cp .env.example .env
# Edit .env locally with your own API credentials (NEVER commit this file)
python scripts/fetch_finra.py --dataset consolidatedShortInterest
python scripts/fetch_nasdaq.py
# Public FINRA historical consolidated NMS, potentially thousands of requests:
python scripts/fetch_finra_bulk.py --start 2018-08-01 --end 2026-11-06 --dry-run
```

**Never distribute personal API credentials or premium data without permissions.** If you publish this on GitHub, review licensing for user-supplied Yahoo-style CSVs and other original data files before pushing.

## Course paper / presentation

Use `docs/METHODOLOGY.md` and `docs/REFERENCES.md` for academic discussion of motivation, related work, formulas, algorithms, results, confounding factors and limitations. The main experiment compares models and data-feature groups using the same chronological periods. The application uses historical *actual observed labels*, not fabricated success rates.

AI assistance disclosure: A large language model was used to assist with project architecture, software code, documentation, methodology, and tests. Verify and attribute such assistance according to the course academic-integrity policy. Run and inspect the code and experimental results yourself before submission.

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv`, then `pip install -r requirements.txt`.
- `Port 8501 already in use`: run `streamlit run app.py --server.port 8502`, and adjust SSH forwarding accordingly.
- No browser on remote machine: use SSH port forwarding above; do not expose the service publicly.
- Models/data missing: run the four `python -m reversalwatch...` commands above.
- API returns 401/403/402: verify credentials and subscription; offline included dataset remains usable.
- Long SEC 13F processing: use the included compact extracts for descriptive analysis; do not bundle gigabytes of original data in a Git repository.
- Results differ across platforms: check your pinned versions and scikit-learn serialization compatibility; models can be regenerated from raw CSVs using the packaged scripts.
