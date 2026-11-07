"""Controlled ablations, PCA comparison and auxiliary continuous-risk regression."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score,mean_absolute_error,r2_score
from .data import ROOT,HORIZONS,FEATURES_BASE
from .train import CUTOFF_VAL,CUTOFF_TEST

def run(panel=None,out=ROOT/'artifacts'):
  if panel is None:panel=pd.read_csv(ROOT/'data/processed/panel.csv',parse_dates=['date'])
  out=Path(out);out.mkdir(exist_ok=True,parents=True)
  specs={
   'price_volume_only':['mom5','mom10','mom20','mom60','volume_ratio','volume_z20','volatility20','ma_gap20','ma_gap50','drawdown20'],
   'with_market_context':list(FEATURES_BASE),
  }
  rows=[]
  for h in HORIZONS:
    end=f'label_end_{h}';label=f'label_{h}';panel[end]=pd.to_datetime(panel[end],errors='coerce')
    for trend in ['up','down']:
      data=panel[(panel.trend==trend)&panel[label].notna()]
      tr=data[(data.date<CUTOFF_VAL)&(data[end]<CUTOFF_VAL)]
      val=data[(data.date>=CUTOFF_VAL)&(data.date<CUTOFF_TEST)&(data[end]<CUTOFF_TEST)]
      test=data[(data.date>=CUTOFF_TEST)]
      for name,features in specs.items():
        variants=['linear','PCA_6_components'] if name=='with_market_context' else ['linear']
        for variant in variants:
          steps=[('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]
          if variant.startswith('PCA'):steps.append(('pca',PCA(n_components=6,random_state=42)))
          steps.append(('classifier',LogisticRegression(class_weight='balanced',max_iter=800,random_state=42)))
          mod=Pipeline(steps).fit(tr[features],tr[label].astype(int))
          pv=mod.predict_proba(val[features])[:,1]
          pt=mod.predict_proba(test[features])[:,1]
          rows.append(dict(horizon=h,trend=trend,features=name,method=variant,train_n=len(tr),test_n=len(test),
            val_average_precision=average_precision_score(val[label].astype(int),pv),
            test_average_precision=average_precision_score(test[label].astype(int),pt)))
  pd.DataFrame(rows).to_csv(out/'ablation_pca_results.csv',index=False)
  # Optional regression response: maximum opposite direction price movement in 20-day window.
  h=20;end='label_end_20'
  samples=panel[panel.trend.isin(['up','down']) & panel[end].notna()].copy()
  samples['future_adverse_20']=np.where(samples.trend=='down',-samples.future_min_20,samples.future_max_20)
  samples=samples[np.isfinite(samples.future_adverse_20)]
  tr=samples[(samples.date<CUTOFF_VAL)&(samples[end]<CUTOFF_VAL)]
  test=samples[samples.date>=CUTOFF_TEST]
  reg=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge(alpha=50))])
  reg.fit(tr[list(FEATURES_BASE)],tr.future_adverse_20)
  pred=reg.predict(test[list(FEATURES_BASE)])
  pd.DataFrame([{'task':'20-day max adverse-direction move (continuous)','mae':mean_absolute_error(test.future_adverse_20,pred),
      'r2':r2_score(test.future_adverse_20,pred),'test_n':len(test)}]).to_csv(out/'regression_results.csv',index=False)
  print('Saved ablation, PCA and secondary regression experiments.')
if __name__=='__main__':run()
