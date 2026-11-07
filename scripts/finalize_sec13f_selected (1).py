"""Finalize previously extracted 13F records; DOES NOT infer true institutional ownership."""
import json
from pathlib import Path
import pandas as pd
CUSIPS={'NVDA':'67066G104','AMD':'007903107','AVGO':'11135F101','MSFT':'594918104','GOOG':'02079K107','AMZN':'023135106','META':'30303M102','AAPL':'037833100','TSLA':'88160R101','INTC':'458140100'}

def main(input_path,source,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 names=['ACCESSION_NUMBER','INFOTABLE_SK','NAMEOFISSUER','TITLEOFCLASS','CUSIP','FIGI','VALUE','SSHPRNAMT','SSHPRNAMTTYPE','PUTCALL','INVESTMENTDISCRETION','OTHERMANAGER','VOTING_AUTH_SOLE','VOTING_AUTH_SHARED','VOTING_AUTH_NONE']
 a=pd.read_csv(input_path,sep='\t',names=names,dtype=str)
 a=a.loc[a.CUSIP.isin(CUSIPS.values())].copy();a['Ticker']=a.CUSIP.map({v:k for k,v in CUSIPS.items()})
 subs=[]
 for file in Path(source).glob('SUBMISSION*.tsv'):
  d=pd.read_csv(file,sep='\t',dtype=str)
  subs.append(d)
 sub=pd.concat(subs).drop_duplicates('ACCESSION_NUMBER')
 sub['FILING_DATE']=pd.to_datetime(sub.FILING_DATE,format='%d-%b-%Y',errors='coerce')
 sub['PERIODOFREPORT']=pd.to_datetime(sub.PERIODOFREPORT,format='%d-%b-%Y',errors='coerce')
 a=a.drop_duplicates(['ACCESSION_NUMBER','INFOTABLE_SK','CUSIP','SSHPRNAMT'])
 a=a.merge(sub[['ACCESSION_NUMBER','FILING_DATE','PERIODOFREPORT','SUBMISSIONTYPE']],on='ACCESSION_NUMBER',how='left')
 # Restrict to shares rather than options; metadata missing is explicitly excluded.
 a=a[(a.SSHPRNAMTTYPE=='SH') & (a.PUTCALL.isna())].copy()
 a['SSHPRNAMT']=pd.to_numeric(a.SSHPRNAMT,errors='coerce')
 a.to_csv(out/'sec13f_selected_holdings.csv',index=False)
 complete=a[a.FILING_DATE.notna()].copy()
 b=complete.groupby(['Ticker','PERIODOFREPORT','FILING_DATE'],as_index=False).agg(
   sum_reported_shares=('SSHPRNAMT','sum'),holding_rows=('ACCESSION_NUMBER','size'),unique_filings=('ACCESSION_NUMBER','nunique'))
 b.to_csv(out/'sec13f_descriptive_sums.csv',index=False)
 summary={'all_selected_holdings':len(a),'matched_filing_date_rows':len(complete),'unmatched_rows':int(a.FILING_DATE.isna().sum()),
          'min_filing_date':str(complete.FILING_DATE.min()),'max_filing_date':str(complete.FILING_DATE.max()),
          'distinct_dates':int(b.FILING_DATE.nunique()),
          'caveat':'Descriptive SEC 13F analysis ONLY; 13F duplicates, amendments, and multiple manager reporting can inflate sums. Not used as institutional ownership percentage or predictor.'}
 (out/'quality.json').write_text(json.dumps(summary,indent=2))
 print(summary)
if __name__=='__main__':main('/tmp/rw13f_selected.tsv','/mnt/data','/mnt/data/ReversalWatch/data/raw/sec13f_compact')
