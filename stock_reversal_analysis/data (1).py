"""Point-in-time conscious construction of ReversalWatch's research panel."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ('NVDA','AMD','AVGO','MSFT','GOOG','AMZN','META','AAPL','TSLA','INTC')
HORIZONS = (10,20,30,60)
FEATURES_BASE = ('mom5','mom10','mom20','mom60','volume_ratio','volume_z20','volatility20','ma_gap20','ma_gap50','drawdown20','spy_mom20','qqq_mom20','rel_spy20','vix','vix_change5')
FEATURES_FACTORS = ('mom_factor_lag1','st_rev_factor_lag1','lt_rev_factor_lag1')
FEATURES_ALL = FEATURES_BASE + FEATURES_FACTORS


def load_market_file(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=['Date']).sort_values('Date').drop_duplicates('Date',keep='last')
    for name in ['Open','High','Low','Close','Adj Close','Volume']:
        if name not in d: raise ValueError(f'Missing {name} in {path}')
    d['Adj Close'] = pd.to_numeric(d['Adj Close'],errors='coerce')
    if d['Adj Close'].isna().any() or (d['Adj Close']<=0).any(): raise ValueError(f'Bad adjusted close in {path}')
    d['Volume'] = pd.to_numeric(d['Volume'], errors='coerce')
    return d.set_index('Date').sort_index()


def load_french(path:Path, column:str) -> pd.Series:
    """Parse French's header-heavy CSV. Returns daily percentage points converted to decimal."""
    rows=[]
    with path.open(encoding='utf-8-sig',errors='replace') as f:
        for line in f:
            m = re.match(r'^\s*(\d{8})\s*,\s*([-+]?\d+(?:\.\d+)?)\s*$',line)
            if m:
                dt=pd.to_datetime(m.group(1),format='%Y%m%d',errors='coerce')
                if pd.notna(dt):
                    value=float(m.group(2))
                    rows.append((dt,value/100 if value not in (-99.99,-999) else np.nan))
    if not rows: raise ValueError(f'No daily rows found in {path}')
    return pd.Series(dict(rows),name=column).sort_index()


def load_finra_samples(data_root:Path=ROOT/'data'/'raw') -> pd.DataFrame:
    files=sorted((data_root/'finra_sample').glob('*.txt'))
    frames=[]
    for p in files:
        d=pd.read_csv(p,sep='|')
        needed={'Date','Symbol','ShortVolume','ShortExemptVolume','TotalVolume'}
        if not needed.issubset(d.columns): continue
        d=d.loc[d['Symbol'].isin(TICKERS)].copy()
        d['Date']=pd.to_datetime(d['Date'].astype(str),format='%Y%m%d')
        for v in ['ShortVolume','ShortExemptVolume','TotalVolume']:
            d[v]=pd.to_numeric(d[v],errors='coerce')
        d['short_sale_ratio']=d['ShortVolume']/d['TotalVolume'].replace(0,np.nan)
        frames.append(d)
    if not frames: return pd.DataFrame(columns=['Date','Symbol','ShortVolume','ShortExemptVolume','TotalVolume','short_sale_ratio'])
    return pd.concat(frames,ignore_index=True).drop_duplicates(['Date','Symbol']).sort_values(['Symbol','Date'])


def build_panel(data_root:Path=ROOT/'data'/'raw', prior_trend:float=.05, reverse_move:float=.08) -> pd.DataFrame:
    benchmark={t:load_market_file(data_root/'benchmarks'/f'{t}.csv') for t in ('SPY','QQQ','DIA')}
    spy=benchmark['SPY']['Adj Close']; qqq=benchmark['QQQ']['Adj Close']
    vix=pd.read_csv(data_root/'market'/'VIXCLS.csv')
    vix['Date']=pd.to_datetime(vix['observation_date'])
    vix=pd.to_numeric(vix['VIXCLS'],errors='coerce').set_axis(vix['Date']).sort_index()
    factors={}
    names={'mom_factor_lag1':('F-F_Momentum_Factor_daily.csv','Mom'),
      'st_rev_factor_lag1':('F-F_ST_Reversal_Factor_daily.csv','ST_Rev'),
      'lt_rev_factor_lag1':('F-F_LT_Reversal_Factor_Daily.csv','LT_Rev')}
    for key,(file,col) in names.items():
        factors[key]=load_french(data_root/'market'/file,col).shift(1)  # previous factor-day proxy
    frames=[]
    for ticker in TICKERS:
        d=load_market_file(data_root/'stocks'/f'{ticker}.csv').copy()
        p=d['Adj Close']
        for days in [5,10,20,60]:d[f'mom{days}']=p.pct_change(days)
        v=d['Volume'].astype(float)
        prev_mean=v.shift(1).rolling(20,min_periods=15).mean()
        prev_std=v.shift(1).rolling(20,min_periods=15).std()
        d['volume_ratio']=v/prev_mean.replace(0,np.nan)
        d['volume_z20']=(v-prev_mean)/prev_std.replace(0,np.nan)
        d['volatility20']=p.pct_change().rolling(20,min_periods=15).std()
        for days in [20,50]:d[f'ma_gap{days}']=p/p.rolling(days).mean()-1
        d['drawdown20']=p/p.rolling(20).max()-1
        d['spy_mom20']=spy.reindex(d.index).pct_change(20)
        d['qqq_mom20']=qqq.reindex(d.index).pct_change(20)
        d['rel_spy20']=d['mom20']-d['spy_mom20']
        # Preserve absence of VIX on holidays; use most recent prior observation only.
        d['vix']=vix.reindex(d.index,method='ffill')
        d['vix_change5']=d['vix'].pct_change(5)
        for key,series in factors.items():d[key]=series.reindex(d.index,method='ffill',limit=5)
        d['trend']=np.select([d['mom20']>=prior_trend,d['mom20']<=-prior_trend],['down','up'],default='neutral')
        # For trend='down': min future adjusted close falls >= threshold.
        # For trend='up': max future adjusted close rises >= threshold.
        for horizon in HORIZONS:
            future = pd.concat([p.shift(-k) for k in range(1,horizon+1)],axis=1)
            future_min=future.min(axis=1);future_max=future.max(axis=1)
            valid=p.shift(-horizon).notna()
            down=(future_min/p-1<=-reverse_move)
            up=(future_max/p-1>=reverse_move)
            d[f'label_{horizon}']=np.select([(d['trend']=='down') & valid,(d['trend']=='up') & valid], [down.astype(int),up.astype(int)],default=np.nan)
            # Valid event labels available only after last endpoint observed.
            d[f'label_end_{horizon}']=pd.Series(d.index,index=d.index).shift(-horizon)
            d[f'future_min_{horizon}']=future_min/p-1
            d[f'future_max_{horizon}']=future_max/p-1
        d['Ticker']=ticker
        frames.append(d.reset_index().rename(columns={'Date':'date','Adj Close':'adj_close','Volume':'volume','Close':'close','Open':'open','High':'high','Low':'low'}))
    result=pd.concat(frames,ignore_index=True).sort_values(['date','Ticker']).reset_index(drop=True)
    # Explicitly keep a tiny sample available for inspection, NOT for training: five days insufficient.
    return result


def save_panel(dest:Path=ROOT/'data'/'processed'/'panel.csv') -> pd.DataFrame:
    df=build_panel();dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest,index=False,float_format='%.9g')
    finra=load_finra_samples();finra.to_csv(dest.parent/'finra_sample_filtered.csv',index=False)
    print(f'Saved {len(df):,} stock-day observations to {dest}; {len(finra)} short-volume sample observations.')
    return df

if __name__=='__main__': save_panel()
