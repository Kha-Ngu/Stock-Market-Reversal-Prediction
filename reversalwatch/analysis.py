"""Unsupervised learning and feature-selection research helpers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from .data import ROOT,FEATURES_BASE

def run_analysis(panel=None, dest=ROOT/'artifacts'):
    if panel is None: panel=pd.read_csv(ROOT/'data/processed/panel.csv',parse_dates=['date'])
    dest=Path(dest);dest.mkdir(exist_ok=True,parents=True)
    # Fit descriptive transforms only on train period; apply across entire history.
    train=panel[panel.date<'2023-01-01']
    imp=SimpleImputer(strategy='median').fit(train[list(FEATURES_BASE)])
    scaler=StandardScaler().fit(imp.transform(train[list(FEATURES_BASE)]))
    sample=panel.loc[panel.date>='2022-01-01'].copy()
    Z=scaler.transform(imp.transform(sample[list(FEATURES_BASE)]))
    # 2-D visualization of PCA from fixed historical feature space.
    pca=PCA(n_components=2,random_state=42).fit(scaler.transform(imp.transform(train[list(FEATURES_BASE)])))
    pcs=pca.transform(Z)
    kmeans=KMeans(n_clusters=4,random_state=42,n_init=10).fit(pca.transform(scaler.transform(imp.transform(train[list(FEATURES_BASE)]))))
    iso=IsolationForest(n_estimators=100,contamination=.05,random_state=42).fit(scaler.transform(imp.transform(train[list(FEATURES_BASE)])))
    sample['pc1']=pcs[:,0];sample['pc2']=pcs[:,1]
    sample['cluster']=kmeans.predict(pcs)
    sample['anomaly_score']=-iso.score_samples(Z)
    sample[['date','Ticker','adj_close','trend','pc1','pc2','cluster','anomaly_score']].to_csv(dest/'regimes.csv',index=False)
    explanation=dict(pca_explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
      pca_feature_loadings={key:np.round(pca.components_[:,i],4).tolist() for i,key in enumerate(FEATURES_BASE)},
      meaning='PCA, KMeans, and Isolation Forest are descriptive; these are not reversal probabilities.')
    (dest/'unsupervised.json').write_text(json.dumps(explanation,indent=2))
    # Filter, wrapper, embedded rankings trained on ONLY historical labeled observations.
    labeled=train[(train.trend=='down') & train.label_20.notna()].copy()
    x=scaler.transform(imp.transform(labeled[list(FEATURES_BASE)])); y=labeled.label_20.astype(int)
    selected=SelectKBest(f_classif,k=min(8,len(FEATURES_BASE))).fit(x,y)
    forest=RandomForestClassifier(n_estimators=75,max_depth=6,min_samples_leaf=20,random_state=42,n_jobs=-1).fit(x,y)
    wrapper=RFE(estimator=RandomForestClassifier(n_estimators=35,max_depth=5,min_samples_leaf=20,random_state=42,n_jobs=-1),n_features_to_select=8,step=2).fit(x,y)
    ranking=pd.DataFrame({'feature':FEATURES_BASE,'filter_f_score':selected.scores_,
        'embedded_importance':forest.feature_importances_, 'wrapper_selected':wrapper.support_,
        'wrapper_rank':wrapper.ranking_})
    ranking.to_csv(dest/'feature_selection.csv',index=False)
    # Similar history examples to latest observation in standardized feature space.
    latest=panel.sort_values('date').groupby('Ticker').tail(1)
    historical=panel[(panel.date<'2025-01-01') & (panel.date>='2019-01-01')]
    Z_hist=scaler.transform(imp.transform(historical[list(FEATURES_BASE)]))
    Z_last=scaler.transform(imp.transform(latest[list(FEATURES_BASE)]))
    results=[]
    for k,(_,r) in enumerate(latest.iterrows()):
        sim=cosine_similarity(Z_last[k:k+1],Z_hist)[0]
        ind=np.argsort(-sim)[:8]
        for i in ind:
            match=historical.iloc[int(i)]
            results.append(dict(Ticker=r.Ticker,similar_date=match.date,similar_ticker=match.Ticker,cosine_similarity=sim[i], historical_trend=match.trend,actual_reversal_20=match.label_20))
    pd.DataFrame(results).to_csv(dest/'similar_histories.csv',index=False)
    print('PCA, clustering, anomaly scores, feature selection and cosine-similarity analysis saved.')

if __name__=='__main__':run_analysis()
