"""Streamlit dashboard: launch with `streamlit run app.py`."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load
from reversalwatch.data import ROOT,TICKERS,HORIZONS,FEATURES_BASE
from reversalwatch.explain import calculation

st.set_page_config(page_title='ReversalWatch | ML Reversal Research',page_icon='📈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1500px}
[data-testid="stMetricValue"]{font-size:1.65rem}
section[data-testid="stSidebar"]{border-right:1px solid #ddd}
.small{font-size:.82rem;color:#667085}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_data():
    p=ROOT/'data/processed/panel.csv'
    if not p.exists():
       st.error('Processed panel missing. Run `python -m reversalwatch.data` first.');st.stop()
    return pd.read_csv(p,parse_dates=['date'])
@st.cache_data(show_spinner=False)
def get_csv(file):
    p=ROOT/'artifacts'/file
    return pd.read_csv(p,parse_dates=['date'] if file in ('test_predictions.csv','latest_forecasts.csv','regimes.csv') else None) if p.exists() else pd.DataFrame()
@st.cache_resource(show_spinner=False)
def get_model(tag):
    p=ROOT/'artifacts'/tag
    return load(p) if p.exists() else None

panel=get_data()
metrics=get_csv('metrics.csv'); chosen=get_csv('selected_models.csv')
forecast=get_csv('latest_forecasts.csv'); predictions=get_csv('test_predictions.csv')
regimes=get_csv('regimes.csv'); similarities=get_csv('similar_histories.csv');rank=get_csv('feature_selection.csv')
ablation=get_csv('ablation_pca_results.csv'); regmetrics=get_csv('regression_results.csv')
config=json.loads((ROOT/'artifacts/manifest.json').read_text())['config'] if (ROOT/'artifacts/manifest.json').exists() else {}
with st.sidebar:
 st.title('📈 ReversalWatch')
 st.caption('Historical ML stock-reversal research')
 ticker=st.selectbox('Stock',TICKERS,index=0)
 horizon=st.select_slider('Forecast horizon (trading days)',options=list(HORIZONS),value=20)
 date_min=panel.date.min().date();date_max=panel.date.max().date()
 start_date=st.date_input('Chart start date',value=max(date_min,(panel.date.max()-pd.DateOffset(years=2)).date()),min_value=date_min,max_value=date_max)
 show_formula=st.toggle('Display computation details',value=True)
 st.caption('Predictions use end-of-day uploaded observations. No live feeds unless you explicitly refresh data.')

st.title('ReversalWatch')
st.caption('Interpretable stock-reversal classification • Real uploaded data • Chronological out-of-sample evaluation')
selected=panel[panel.Ticker==ticker].sort_values('date'); latest=selected.iloc[-1]
trend=latest.trend
if trend=='down':trend_word='Upward-trend stock → possible downward reversal'
elif trend=='up':trend_word='Downward-trend stock → possible upward reversal'
else:trend_word='No qualifying prior 20-day directional trend'
st.info(f'{ticker} | Last available stock observation: {latest.date.date()} | {trend_word} | Forecast window: {horizon} trading days')

pages=st.tabs(['Overview & forecast','Backtesting','Model comparison','Feature analysis','Similarity & regimes','Methodology & sources'])

def interactive_price(data, marker=None):
    x=data[data.date>=pd.Timestamp(start_date)].copy()
    x['momentum_20_pct']=100*x.mom20
    x['volatility_20_pct']=100*x.volatility20
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x.date,y=x.adj_close,mode='lines',name='Adjusted close',line=dict(color='#157f4a',width=2),
        customdata=np.column_stack((x.volume,x.momentum_20_pct,x.volatility_20_pct,x.vix)),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Adj close: $%{y:,.2f}<br>Volume: %{customdata[0]:,.0f}<br>20D momentum: %{customdata[1]:.2f}%<br>20D volatility: %{customdata[2]:.2f}%<br>VIX: %{customdata[3]:.2f}<extra></extra>'))
    if marker is not None:
       mark=data[data.date==pd.Timestamp(marker)]
       if len(mark):fig.add_vline(x=pd.Timestamp(marker).timestamp()*1000,line_dash='dot',line_color='#b45309')
    fig.update_layout(template='plotly_white',hovermode='x unified',height=390,margin=dict(l=12,r=15,t=20,b=15),
       xaxis_title='Trading date',yaxis_title='Adjusted close (USD)',legend=dict(orientation='h'))
    return fig

def metric_explain(name,value,metric):
    with st.popover(name+'  ⓘ',use_container_width=True):
        detail=calculation(panel,ticker,latest.date,metric)
        st.write('**Formula / definition:**',detail.get('formula',''))
        if detail.get('inputs'):st.json(detail['inputs'])
        if detail.get('substitution'):st.code(detail['substitution'])
        st.write('**Computed value:**',detail.get('result'))
        st.caption('Source: '+detail.get('source','data dictionary'))

with pages[0]:
 st.subheader('Latest observation and prediction')
 c1,c2,c3,c4=st.columns(4)
 with c1:st.metric('Adjusted closing price',f'${latest.adj_close:,.2f}')
 with c2:st.metric('20-day momentum',f'{100*latest.mom20:+.2f}%')
 with c3:st.metric('Volume ratio',f'{latest.volume_ratio:.2f}×')
 with c4:st.metric('VIX',f'{latest.vix:.2f}')
 if show_formula:
    e1,e2,e3,e4=st.columns(4)
    with e1:metric_explain('Price: source data',f'{latest.adj_close:.2f}','adj_close')
    with e2:metric_explain('Momentum math',f'{latest.mom20:.2%}','mom20')
    with e3:metric_explain('Volume ratio math',f'{latest.volume_ratio:.2f}','volume_ratio')
    with e4:metric_explain('VIX data source',f'{latest.vix:.2f}','vix')
 st.plotly_chart(interactive_price(selected),use_container_width=True)
 inspected=st.selectbox('Click to inspect a chart date / computation',selected.loc[selected.date>=pd.Timestamp(start_date),'date'].dt.strftime('%Y-%m-%d').iloc[::-1].tolist(),index=0)
 x1,x2=st.columns(2)
 with x1:
    metric_key=st.selectbox('Calculation for that date',['mom5','mom10','mom20','mom60','volume_ratio','volatility20','vix','label_10','label_20','label_30','label_60'])
 with x2:
    detail=calculation(panel,ticker,inspected,metric_key)
    with st.popover('View exact inputs and arithmetic  ⓘ',use_container_width=True):st.json(detail)
 st.caption('Hover over the green price line to inspect each observation. Adjusted prices are used for returns; all values reflect the uploaded historical dataset.')
 if trend=='neutral':st.warning('No qualifying prior trend (±5% over 20 sessions). A directional reversal probability is not defined for this stock today.')
 elif chosen.empty:st.warning('Trained models missing. Run `python -m reversalwatch.train`.')
 else:
    c=chosen[(chosen.trend==trend)&(chosen.horizon==horizon)]
    if len(c):
       c=c.iloc[0];fs=forecast[(forecast.Ticker==ticker)&(forecast.horizon==horizon)&(forecast.model==c.model)]
       if len(fs):
          prob=float(fs.iloc[-1].probability)
          col1,col2,col3=st.columns(3)
          col1.metric('Estimated event probability',f'{100*prob:.1f}%')
          col2.metric('Decision threshold',f'{100*c.threshold:.0f}%')
          col3.metric('Held-out test average precision',f'{c.average_precision:.3f}')
          st.write(f'**Selected model:** {c.model} (chosen using validation average precision)')
          st.write('**Interpretation:** '+('Above' if prob>=c.threshold else 'Below')+' the validation-selected decision threshold. This is an uncertain statistical estimate, not an investment recommendation.')
          with st.popover('How was this prediction calculated?  ⓘ'):
             st.write('**Algorithm:**',c.model)
             st.write('**Event definition:** An at-least-8% movement opposite a prior 20-day trend of at least 5%, within the selected horizon.')
             st.write('**Prediction calculation:** trained scikit-learn `predict_proba()` using the feature vector below. This is not a single arithmetic formula.')
             st.dataframe(pd.DataFrame({'feature':list(FEATURES_BASE),'latest_input':[latest[x] for x in FEATURES_BASE]}),hide_index=True)
             st.write('**Test evidence:**',c[['test_n' if 'test_n' in c else 'n','events','precision','recall','f1','average_precision','brier']].to_dict())
             st.caption('Click another metric elsewhere to see its arithmetic. Model output is not calibrated unless the calibration analysis supports it.')
 st.markdown('#### Forecast probabilities across all horizons')
 if trend!='neutral' and not chosen.empty:
    rows=[]
    for h in HORIZONS:
       r=chosen[(chosen.trend==trend)&(chosen.horizon==h)]
       if len(r):
          chosen_model=r.iloc[0].model
          f=forecast[(forecast.Ticker==ticker)&(forecast.horizon==h)&(forecast.model==chosen_model)]
          if len(f):rows.append(dict(horizon=f'{h} trading days',probability=float(f.iloc[-1].probability),model=chosen_model))
    if rows:
       d=pd.DataFrame(rows);fig=px.bar(d,x='horizon',y='probability',hover_data=['model'],range_y=[0,1],title='Predicted event probability by horizon')
       fig.update_layout(template='plotly_white',height=320);st.plotly_chart(fig,use_container_width=True)
 st.markdown('#### Uploaded FINRA short-sale activity — sample only')
 sample=ROOT/'data/processed/finra_sample_filtered.csv'
 if sample.exists():
    fin=pd.read_csv(sample,parse_dates=['Date'])
    f=fin[fin.Symbol==ticker]
    if len(f):
       f['short_sale_ratio_pct']=f.short_sale_ratio*100
       fig=px.line(f,x='Date',y='short_sale_ratio_pct',markers=True,hover_data=['ShortVolume','ShortExemptVolume','TotalVolume'],title='FINRA-reported short-sale volume / reported total volume')
       fig.update_layout(template='plotly_white',height=280,yaxis_title='Percent')
       st.plotly_chart(fig,use_container_width=True)
       st.caption('Only 5 sample dates are included. This measures short-sale flow across FINRA reporting facilities, NOT outstanding short interest. Not used to train reversal classifiers.')

with pages[1]:
 st.subheader('Historical forecasts vs. realized events')
 st.caption('These are predictions for held-out 2025–2026 observations, not in-sample fitted values.')
 model_list=sorted(predictions.model.dropna().unique()) if len(predictions) else []
 if model_list:
   mdl=st.selectbox('Backtesting model',model_list)
   hist=predictions[(predictions.Ticker==ticker)&(predictions.horizon==horizon)&(predictions.model==mdl)].copy()
   if len(hist):
      hist['predicted_reversal']=(hist.probability>=hist.decision_threshold).astype(int)
      st.plotly_chart(px.scatter(hist,x='date',y='probability',color='actual_label',hover_data=['adj_close','trend','predicted_reversal','decision_threshold'],title='Historical predictions (color = actual outcome)',labels={'actual_label':'Actual reversal'}).update_layout(template='plotly_white',height=375),use_container_width=True)
      a,b,c=st.columns(3);a.metric('Evaluated samples',f'{len(hist):,}');b.metric('Actual reversals',f'{int(hist.actual_label.sum()):,}');c.metric('False alarms',f'{int(((hist.predicted_reversal==1)&(hist.actual_label==0)).sum()):,}')
      st.dataframe(hist.sort_values('date',ascending=False).head(200),hide_index=True,use_container_width=True)
      st.download_button('Download these test predictions (CSV)',hist.to_csv(index=False),'backtest_predictions.csv')
   else:st.info('No qualifying trend observations for these selections in the test period.')
 else:st.info('Run training to generate historical predictions.')

with pages[2]:
 st.subheader('Model evaluation and algorithm comparison')
 st.caption('Chronological train / validation / test; label windows crossing boundaries are purged. Model selection based on validation AP only.')
 if len(metrics):
   view=metrics[metrics.horizon==horizon].copy(); direction=st.selectbox('Direction',['down','up'],format_func=lambda x:'Downward reversal after rising trend' if x=='down' else 'Upward reversal after falling trend')
   view=view[view.trend==direction].sort_values('average_precision',ascending=False)
   keep=['model','n','events','event_rate','precision','recall','f1','roc_auc','average_precision','brier','accuracy','threshold','tp','fp','tn','fn']
   st.dataframe(view[keep],hide_index=True,use_container_width=True)
   st.plotly_chart(px.bar(view,x='model',y=['precision','recall','f1'],barmode='group',title='Out-of-sample precision, recall, F1').update_layout(template='plotly_white',height=340),use_container_width=True)
   model=st.selectbox('Confusion matrix for',view.model.tolist())
   rec=view[view.model==model].iloc[0];matrix=np.array([[rec.tn,rec.fp],[rec.fn,rec.tp]])
   fig=px.imshow(matrix,text_auto=True,x=['Pred no','Pred yes'],y=['Actual no','Actual yes'],color_continuous_scale='Blues',title='Held-out test confusion matrix')
   st.plotly_chart(fig,use_container_width=True)
   st.write('**Why accuracy alone is not enough:** A model can look accurate when no-reversal cases dominate. Average precision, recall and false positives provide more context.')
   st.download_button('Download all model metrics (CSV)',metrics.to_csv(index=False),'metrics.csv')
 else:st.warning('No model metrics. Run training first.')

with pages[3]:
 st.subheader('Feature diagnostics and reduction')
 feature=st.selectbox('Feature to inspect',list(FEATURES_BASE))
 st.plotly_chart(px.histogram(panel,x=feature,color='trend',nbins=70,opacity=.65,title=f'{feature} distribution by prior trend').update_layout(template='plotly_white',height=320),use_container_width=True)
 corr=panel[list(FEATURES_BASE)].corr()
 st.plotly_chart(px.imshow(corr,aspect='auto',color_continuous_scale='RdBu',zmin=-1,zmax=1,title='Input feature correlation matrix').update_layout(template='plotly_white',height=470),use_container_width=True)
 if len(rank):
    st.write('**Feature selection methods** — F-test filter, random forest embedded importance, RFE wrapper')
    st.dataframe(rank.sort_values('embedded_importance',ascending=False),hide_index=True)
 if len(ablation):
    st.write('**Controlled experiment: baseline vs market context vs PCA**')
    filt=ablation[(ablation.horizon==horizon)]
    st.dataframe(filt[['trend','features','method','train_n','test_n','val_average_precision','test_average_precision']],hide_index=True)
 if len(regmetrics):
    st.write('**Auxiliary 20-day maximum adverse-move regression**')
    st.dataframe(regmetrics,hide_index=True)
 p=ROOT/'artifacts/unsupervised.json'
 if p.exists():
    info=json.loads(p.read_text());st.write('**PCA explained variance**',info['pca_explained_variance_ratio'])
    st.caption('PCA transforms standardized features into components, which may reduce dimensionality at the expense of interpretability.')

with pages[4]:
 st.subheader('Similarity, clustering and unusual conditions')
 if len(regimes):
    sample=regimes[regimes.Ticker==ticker].copy()
    st.plotly_chart(px.scatter(sample,x='pc1',y='pc2',color='cluster',hover_data=['date','adj_close','trend','anomaly_score'],title='PCA + K-Means historical regimes').update_layout(template='plotly_white',height=390),use_container_width=True)
    st.plotly_chart(px.line(sample,x='date',y='anomaly_score',title='Isolation Forest anomaly score (higher = more unusual)').update_layout(template='plotly_white',height=280),use_container_width=True)
 if len(similarities):
    st.write('**Most similar historical feature vectors by cosine similarity**')
    st.dataframe(similarities[similarities.Ticker==ticker].sort_values('cosine_similarity',ascending=False),hide_index=True,use_container_width=True)
 st.warning('Clusters and similarity do not independently indicate likely reversals. They are descriptive analyses, not trading signals.')
 sec_path=ROOT/'data/raw/sec13f_compact/sec13f_descriptive_sums.csv'
 if sec_path.exists():
    st.markdown('#### SEC Form 13F — descriptive holdings (not model inputs)')
    sec=pd.read_csv(sec_path,parse_dates=['FILING_DATE','PERIODOFREPORT']); sec=sec[sec.Ticker==ticker].sort_values('FILING_DATE')
    if len(sec):
       sec=sec.groupby('FILING_DATE',as_index=False)['sum_reported_shares'].sum()
       st.plotly_chart(px.line(sec,x='FILING_DATE',y='sum_reported_shares',markers=True,title='Sum of reported 13F share records by filing date (unreconciled)').update_layout(template='plotly_white',height=300),use_container_width=True)
       st.warning('13F summary is incomplete and unreconciled. Duplicate reporting and missing filing joins can inflate totals. This is NOT a verified institutional ownership percentage and is excluded from model training.')

with pages[5]:
 st.subheader('Methodology, data dictionary and reproducibility')
 st.markdown('**Research objective:** Predict ≥8% opposite-direction moves following a ≥5% prior 20-session trend within 10, 20, 30, or 60 trading days. The thresholds are research assumptions.')
 st.markdown('**Data:** 10 stocks (GOOG rather than GOOGL), SPY/QQQ/DIA benchmarks, VIXCLS, French factor files. Five FINRA Consolidated NMS daily files are available as demonstration-only short-sale-volume data. SEC 13F data is not incorporated into the main trained models pending complete audited filing-time joins.')
 st.markdown('**Label leakage protection:** Training observations with event label periods crossing the validation start are excluded. Likewise, validation observations whose labels cross into the test period are excluded. Scalers/imputers are fitted inside each model pipeline only on the training fold.')
 st.markdown('**Split:** Train 2018–2022, validation 2023–2024, test 2025–2026 (with future-label maturity and boundary purging).')
 st.markdown('**Metrics:** ROC-AUC, average precision, precision, recall, F1, accuracy, confusion matrix, Brier score; model selection by validation average precision. Probability calibration is not assumed or guaranteed.')
 st.markdown('**Features used for predictions:** '+', '.join(FEATURES_BASE))
 st.markdown('**Analyses:** PCA, K-Means, cosine similarity, isolation forest, F-statistic selection, RFE, embedded importances. These do not replace the supervised model.')
 st.markdown('**Research context:** Jegadeesh & Titman (1993), Daniel & Moskowitz (2016), Gu, Kelly & Xiu (2020). See docs/REFERENCES.md for citations and data links.')
 st.markdown('**Inputs are point-in-time assumptions:** price and volume features through the end of the selected session; VIX as last observed closing index value. These are not intraday or real-time forecasts.')
 st.markdown('**Caution:** Not financial advice. Public-volume indicators do not prove institutional crowding, and models can fail under regime shifts. Historical performance is not guaranteed to persist.')
 st.code('python -m reversalwatch.data\npython -m reversalwatch.train\npython -m reversalwatch.analysis\nstreamlit run app.py',language='bash')
 st.caption('All values come from the packaged data and outputs. No personal credentials are included.')
