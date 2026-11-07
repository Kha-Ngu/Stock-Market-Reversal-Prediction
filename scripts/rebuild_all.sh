#!/usr/bin/env bash
set -euo pipefail
python -m reversalwatch.data
python -m reversalwatch.train
python -m reversalwatch.analysis
echo 'All model experiments and visualizations refreshed. Run: streamlit run app.py'
