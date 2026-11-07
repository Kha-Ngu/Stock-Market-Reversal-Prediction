"""Traceability for user-visible calculated values."""
from __future__ import annotations
import pandas as pd
import numpy as np

def calculation(frame:pd.DataFrame, ticker:str, date, metric:str) -> dict:
    t=frame[frame.Ticker==ticker].sort_values('date').reset_index(drop=True)
    t['date']=pd.to_datetime(t.date)
    d=pd.Timestamp(date)
    matching=t.index[t.date==d]
    if not len(matching):return {'error':'No source observation for this date'}
    i=int(matching[-1]);r=t.iloc[i]
    def px(j):return float(t.iloc[j].adj_close) if j>=0 else np.nan
    if metric in ('mom5','mom10','mom20','mom60'):
        n=int(metric[3:]); start=px(i-n);end=px(i)
        return {'title':f'{n}-day momentum','formula':f'(Adjusted Close today / Adjusted Close {n} trading sessions ago) - 1',
            'inputs':{'current_adjusted_close':end,'prior_adjusted_close':start,'prior_date':str(t.iloc[i-n].date.date()) if i>=n else None},
            'substitution':f'({end:.4f}/{start:.4f})-1' if pd.notna(start) else 'Insufficient history',
            'result':float(r[metric]),'units':'fraction (multiply by 100 for percent)','source':'Historical stock adjusted closes'}
    if metric=='volume_ratio':
        last=t.iloc[max(0,i-20):i];avg=float(last.volume.mean())
        return {'title':'Daily volume ratio','formula':'Current volume / mean(previous 20 session volumes)',
        'inputs':{'current_volume':float(r.volume),'historical_average_volume':avg,'sessions':len(last)},
        'substitution':f'{float(r.volume):,.0f} / {avg:,.0f}','result':float(r.volume_ratio),'units':'×','source':'Historical trading volume'}
    if metric=='volatility20':
        return {'title':'20-day daily return volatility','formula':'Sample standard deviation of last 20 daily adjusted-close returns',
        'inputs':{'rolling_session_count':20},'result':float(r.volatility20),'units':'daily fractional std. dev.','source':'Adjusted closing-price returns'}
    if metric=='vix':
        return {'title':'VIX','formula':'Direct observed index value (not calculated here)',
        'inputs':{'vix_level':float(r.vix)},'result':float(r.vix),'units':'index points','source':'FRED VIXCLS; last observed on or before selected trading date'}
    if metric=='short_sale_ratio':
        return {'title':'FINRA short-sale volume ratio','formula':'FINRA reported short-sale volume / FINRA reported total volume',
        'units':'fraction','source':'FINRA Consolidated NMS. Does NOT measure outstanding short interest.'}
    if metric.startswith('label_'):
        h=int(metric.split('_')[1]);res=r[metric]
        return {'title':f'{h}-day historical event label','formula':'After a +5% prior 20-day trend: min future adjusted close <= 92% of prediction-date close. After a -5% prior trend: max future adjusted close >= 108%.',
          'inputs':{'prior_trend':r.trend,'future_low_return':float(r[f'future_min_{h}']) if pd.notna(r[f'future_min_{h}']) else None,
             'future_high_return':float(r[f'future_max_{h}']) if pd.notna(r[f'future_max_{h}']) else None},
          'result':None if pd.isna(res) else int(res),'units':'binary event','source':'Future adjusted-close data, used ONLY for historical labels'}
    return {'title':metric,'formula':'See Methodology → feature glossary','result':float(r[metric]) if metric in r and pd.notna(r[metric]) else None}
