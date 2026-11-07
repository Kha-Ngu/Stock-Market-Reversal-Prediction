"""Leakage-aware chronological reversal model training + honest evaluation.

Each model is trend-conditional and horizon-specific. Observations with a
future label window crossing a subsequent split are excluded (purging).
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
  roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss)
from .data import ROOT, HORIZONS, FEATURES_BASE, TICKERS

CUTOFF_VAL=pd.Timestamp('2023-01-01')
CUTOFF_TEST=pd.Timestamp('2025-01-01')
MODELS={
'Logistic Regression':lambda:Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),('model',LogisticRegression(max_iter=700,class_weight='balanced',random_state=42))]),
'Random Forest':lambda:Pipeline([('impute',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=20,n_jobs=-1,class_weight='balanced_subsample',random_state=42))]),
'Gradient Boosting':lambda:Pipeline([('impute',SimpleImputer(strategy='median')),('model',HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=15,min_samples_leaf=25,l2_regularization=2,random_state=42))]),
'Decision Tree':lambda:Pipeline([('impute',SimpleImputer(strategy='median')),('model',DecisionTreeClassifier(max_depth=5,min_samples_leaf=50,class_weight='balanced',random_state=42))]),
'K-Nearest Neighbors':lambda:Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',KNeighborsClassifier(n_neighbors=41,weights='distance',n_jobs=-1))]),
'Gaussian Naive Bayes':lambda:Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',GaussianNB(var_smoothing=1e-7))]),
}

def measures(y,prob,threshold=.5):
    y=np.asarray(y,dtype=int);prob=np.asarray(prob,dtype=float)
    predicted=(prob>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return dict(n=len(y),events=int(y.sum()),event_rate=round(float(y.mean()),5),
      accuracy=round(float(accuracy_score(y,predicted)),5),precision=round(float(precision_score(y,predicted,zero_division=0)),5),
      recall=round(float(recall_score(y,predicted,zero_division=0)),5),f1=round(float(f1_score(y,predicted,zero_division=0)),5),
      roc_auc=round(float(roc_auc_score(y,prob)),5) if len(set(y))==2 else None,
      average_precision=round(float(average_precision_score(y,prob)),5) if len(set(y))==2 else None,
      brier=round(float(brier_score_loss(y,prob)),5),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def best_threshold(y,p):
    scores=[(f1_score(y,np.asarray(p)>=t,zero_division=0),t) for t in np.arange(.2,.801,.05)]
    return round(float(max(scores,key=lambda x:x[0])[1]),3)

def fit_all(panel:pd.DataFrame|None=None, artifact_dir:Path=ROOT/'artifacts', fast:bool=False):
    if panel is None: panel=pd.read_csv(ROOT/'data/processed/panel.csv',parse_dates=['date'])
    artifact_dir.mkdir(parents=True,exist_ok=True)
    rows=[];preds=[];forecasts=[]; meta=[]
    config= {'features':list(FEATURES_BASE),'train_before':'2023-01-01','validation_start':'2023-01-01',
       'test_start':'2025-01-01','trend_threshold':0.05,'reversal_threshold':0.08,
       'horizons':list(HORIZONS),'models':list(MODELS),'method':'trend-conditional binary reversal classification, future-label purge',
       'seed':42,'point_in_time_policy':'price features only; feature scalers and imputation fitted on training fold'}
    models= {'Logistic Regression':MODELS['Logistic Regression'],'Random Forest':MODELS['Random Forest']} if fast else MODELS
    for h in HORIZONS:
      label=f'label_{h}'; end=f'label_end_{h}'
      panel[end]=pd.to_datetime(panel[end],errors='coerce')
      for trend in ['down','up']:
        base=panel[(panel.trend==trend) & panel[label].notna()].copy()
        train=base[(base.date<CUTOFF_VAL)&(base[end]<CUTOFF_VAL)]
        val=base[(base.date>=CUTOFF_VAL)&(base.date<CUTOFF_TEST)&(base[end]<CUTOFF_TEST)]
        test=base[(base.date>=CUTOFF_TEST)]
        if any(len(x)<100 for x in [train,val,test]):
            print('Insufficient data',h,trend);continue
        xtr=train[list(FEATURES_BASE)]; ytr=train[label].astype(int)
        xv=val[list(FEATURES_BASE)];yv=val[label].astype(int)
        xt=test[list(FEATURES_BASE)];yt=test[label].astype(int)
        for name,factory in models.items():
          model=factory()
          with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning)
            model.fit(xtr,ytr)
          pv=model.predict_proba(xv)[:,1];pt=model.predict_proba(xt)[:,1]
          threshold=best_threshold(yv,pv)
          measurements=measures(yt,pt,threshold)
          mv=measures(yv,pv,threshold)
          tag=f'{name.lower().replace(" ","_").replace("-","_")}_{trend}_{h}'
          dump(model,artifact_dir/f'{tag}.joblib',compress=3)
          row=dict(model=name,trend=trend,horizon=h,threshold=threshold,train_n=len(train),val_n=len(val),**measurements,
            val_f1=mv['f1'],val_average_precision=mv['average_precision'])
          rows.append(row)
          data=test[['date','Ticker','adj_close',label,'trend']].copy()
          data['model']=name;data['horizon']=h;data['probability']=pt;data['decision_threshold']=threshold
          data=data.rename(columns={label:'actual_label'})
          preds.append(data)
          today=panel[panel.trend==trend].sort_values('date').groupby('Ticker').tail(1)
          # Actual latest observation per stock must belong to this trend.
          latest=panel.sort_values('date').groupby('Ticker').tail(1)
          latest=latest[latest.trend==trend]
          if len(latest):
            probs=model.predict_proba(latest[list(FEATURES_BASE)])[:,1]
            for (_,r),p in zip(latest.iterrows(),probs):
              forecasts.append(dict(date=str(r.date.date()),Ticker=r.Ticker,trend=trend,horizon=h,
                model=name,probability=float(p),decision_threshold=threshold))
          meta.append(dict(file=tag+'.joblib',horizon=h,trend=trend,model=name,features=list(FEATURES_BASE),threshold=threshold))
          print(f'{h:2d} {trend:4s} {name:22s}: test AP={measurements["average_precision"]} F1={measurements["f1"]} n={len(test)}',flush=True)
    pd.DataFrame(rows).to_csv(artifact_dir/'metrics.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(artifact_dir/'test_predictions.csv',index=False)
    pd.DataFrame(forecasts).to_csv(artifact_dir/'latest_forecasts.csv',index=False)
    (artifact_dir/'manifest.json').write_text(json.dumps(dict(config=config,models=meta),indent=2))
    # Model selection based on VALIDATION AP only; report the locked TEST performance in UI.
    selected=pd.DataFrame(rows).sort_values(['trend','horizon','val_average_precision'],ascending=[True,True,False]).drop_duplicates(['trend','horizon'])
    selected.to_csv(artifact_dir/'selected_models.csv',index=False)
    print('Completed: metrics and artifacts in',artifact_dir)
    return pd.DataFrame(rows)

if __name__=='__main__':fit_all()
