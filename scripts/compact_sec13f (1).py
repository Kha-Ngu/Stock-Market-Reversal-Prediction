"""Optional reproducible, memory-bounded SEC 13F *descriptive* feature extraction.

Requires the original INFOTABLE.tsv and corresponding SUBMISSION.tsv files. It
keeps only selected 10 stocks; never invents institutional ownership % or lets
holdings become predictive features without as-of filing-date checks.
"""
from __future__ import annotations
from pathlib import Path
import glob
import pandas as pd
import numpy as np

CUSIPS={'NVDA':'67066G104','AMD':'007903107','AVGO':'11135F101','MSFT':'594918104',
        'GOOG':'02079K107','AMZN':'023135106','META':'30303M102','AAPL':'037833100',
        'TSLA':'88160R101','INTC':'458140100'}


def compact(source:Path,output:Path):
    source=Path(source);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    submissions=[]
    for f in source.glob('SUBMISSION*.tsv'):
        d=pd.read_csv(f,sep='\t',dtype=str,usecols=['ACCESSION_NUMBER','FILING_DATE','SUBMISSIONTYPE','PERIODOFREPORT'])
        submissions.append(d)
    if not submissions:
        print('No SUBMISSION metadata; cannot establish filing availability. Skip SEC extraction.');return
    sub=pd.concat(submissions,ignore_index=True).drop_duplicates('ACCESSION_NUMBER')
    sub['FILING_DATE']=pd.to_datetime(sub.FILING_DATE,format='%d-%b-%Y',errors='coerce')
    sub['PERIODOFREPORT']=pd.to_datetime(sub.PERIODOFREPORT,format='%d-%b-%Y',errors='coerce')
    cols=['ACCESSION_NUMBER','NAMEOFISSUER','TITLEOFCLASS','CUSIP','VALUE','SSHPRNAMT','SSHPRNAMTTYPE','PUTCALL']
    pieces=[];total_matches=0
    files=sorted(source.glob('INFOTABLE*.tsv'))
    for f in files:
        part_count=0
        for chunk in pd.read_csv(f,sep='\t',dtype=str,usecols=cols,chunksize=100000,low_memory=False):
            match=chunk[chunk.CUSIP.isin(CUSIPS.values())].copy()
            if match.empty:continue
            match['ticker']=match.CUSIP.map({v:k for k,v in CUSIPS.items()})
            match['source_file']=f.name
            pieces.append(match)
            part_count+=len(match)
        total_matches+=part_count
        print(f.name,'rows in target CUSIPs:',part_count,flush=True)
    if not pieces:
        print('No matching 13F entries');return
    holdings=pd.concat(pieces,ignore_index=True).drop_duplicates(['ACCESSION_NUMBER','CUSIP','VALUE','SSHPRNAMT','PUTCALL'])
    holdings=holdings.merge(sub[['ACCESSION_NUMBER','FILING_DATE','PERIODOFREPORT','SUBMISSIONTYPE']],on='ACCESSION_NUMBER',how='left')
    holdings['SSHPRNAMT']=pd.to_numeric(holdings.SSHPRNAMT,errors='coerce')
    holdings=holdings[(holdings.PUTCALL.isna())&(holdings.SSHPRNAMTTYPE=='SH')]
    holdings.to_csv(output/'holdings_selected.csv',index=False)
    dated=holdings[holdings.FILING_DATE.notna()].copy()
    grouped=dated.groupby(['ticker','PERIODOFREPORT','FILING_DATE'],as_index=False).agg(reported_shares=('SSHPRNAMT','sum'),filing_rows=('ACCESSION_NUMBER','size'))
    grouped.to_csv(output/'holdings_by_filing_date.csv',index=False)
    report={'read_source_files':len(files),'raw_matching_rows':total_matches,'unique_target_holdings':len(holdings),
          'records_with_filing_date':len(dated),'unmatched_filing_date':len(holdings)-len(dated),
          'warning':'Descriptive only. 13F can have duplicate reporting and amendments; not necessarily institutional ownership %, cannot be used for prediction without full reconciliation and point-in-time checks.'}
    import json
    (output/'quality.json').write_text(json.dumps(report,indent=2))
    print(report)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',default='/mnt/data');p.add_argument('--out',default='data/raw/sec13f_compact');a=p.parse_args()
    compact(Path(a.source),Path(a.out))
