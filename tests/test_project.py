from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from reversalwatch.data import ROOT,TICKERS,HORIZONS,FEATURES_BASE,load_french,load_finra_samples
from reversalwatch.explain import calculation
from reversalwatch.train import measures

@pytest.fixture(scope='module')
def panel():return pd.read_csv(ROOT/'data/processed/panel.csv',parse_dates=['date'])

def test_stock_universe_and_history(panel):
    assert set(panel.Ticker)==set(TICKERS)
    assert panel.groupby('Ticker').size().min()>2100
    assert panel.date.max()>=pd.Timestamp('2026-11-05')

def test_labels_never_available_without_future_window(panel):
    for h in HORIZONS:
        assert panel.loc[panel.groupby('Ticker').cumcount(ascending=False)<h,f'label_{h}'].isna().all()

def test_neutral_rows_not_labeled(panel):
    assert panel.loc[panel.trend=='neutral',[f'label_{h}' for h in HORIZONS]].isna().all().all()

def test_calculation_explains_price_momentum(panel):
    sample=panel[(panel.Ticker=='NVDA')&(panel.date==pd.Timestamp('2024-06-10'))].iloc[0]
    d=calculation(panel,'NVDA',pd.Timestamp('2024-06-10'),'mom20')
    assert np.isclose(d['result'],sample.mom20,rtol=1e-6)
    assert np.isclose(d['result'],d['inputs']['current_adjusted_close']/d['inputs']['prior_adjusted_close']-1)

def test_purge_train_val_and_test(panel):
    for h in HORIZONS:
        end=pd.to_datetime(panel[f'label_end_{h}'])
        train=(panel.date<'2023-01-01')&(end<'2023-01-01')
        val=(panel.date>='2023-01-01')&(panel.date<'2025-01-01')&(end<'2025-01-01')
        test=panel.date>='2025-01-01'
        assert (end[train]<pd.Timestamp('2023-01-01')).all()
        assert (end[val]<pd.Timestamp('2025-01-01')).all()
        assert not (train&val).any() and not (val&test).any()

def test_finra_sample_is_not_short_interest():
    frame=load_finra_samples()
    assert len(frame)==5*len(TICKERS)
    assert frame.short_sale_ratio.between(0,1).all()
    assert 'shares_short_outstanding' not in frame.columns

def test_models_and_metrics_exist():
    from joblib import load
    m=pd.read_csv(ROOT/'artifacts/metrics.csv')
    assert len(m)>=16
    assert m.horizon.nunique()==4
    assert set(m.trend)=={'up','down'}
    assert ((m.average_precision>=0)&(m.average_precision<=1)).all()
    files=list((ROOT/'artifacts').glob('*.joblib'))
    assert len(files)>=16
    mod=load(files[0]); assert hasattr(mod,'predict_proba')

def test_metrics_calculation():
    metric=measures([1,0,1,0],[.9,.1,.8,.2],.5)
    assert metric['accuracy']==1 and metric['tp']==2 and metric['tn']==2

