"""Download FINRA public CNMS consolidated NMS daily files by date and filter 10 symbols.

Uses only the free public CNMS files. Internet is required to RUN this script.
Do not confuse short sale *volume* with outstanding short interest.
Use --dry-run to inspect expected endpoints without traffic.
"""
from __future__ import annotations
import argparse,datetime as dt,io,time
from pathlib import Path
import pandas as pd
import requests
SYMBOLS={'NVDA','AMD','AVGO','MSFT','GOOG','AMZN','META','AAPL','TSLA','INTC'}
ROOT='https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt'

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',default='2018-08-01');p.add_argument('--end',default='2026-11-06')
    p.add_argument('--out',default='data/external/finra_cnms_10_stocks.csv');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--delay',type=float,default=.12); args=p.parse_args()
    dates=pd.bdate_range(args.start,args.end)
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
    done=set()
    if out.exists():
        saved=pd.read_csv(out,usecols=['Date']);done=set(saved.Date.astype(str))
    session=requests.Session();session.headers['User-Agent']='ReversalWatch university research project (contact via repository README)'
    failures=[];downloaded=0
    for day in dates:
        key=day.strftime('%Y%m%d');url=ROOT.format(date=key)
        if key in done:continue
        if args.dry_run:
            if downloaded<4:print('Would request',url)
            downloaded+=1;continue
        try:
            response=session.get(url,timeout=30)
            if response.status_code in (403,404):
                failures.append((key,response.status_code));continue
            response.raise_for_status()
            data=pd.read_csv(io.BytesIO(response.content),sep='|',dtype={'Date':str,'Symbol':str})
            if not {'Date','Symbol','ShortVolume','TotalVolume'}.issubset(data.columns):
                failures.append((key,'unexpected schema'));continue
            small=data[data.Symbol.isin(SYMBOLS)]
            if len(small):
                small.to_csv(out,mode='a',index=False,header=not out.exists())
                downloaded+=1
            time.sleep(args.delay)
        except (requests.RequestException,ValueError,pd.errors.ParserError) as e:
            failures.append((key,str(e)[:150]))
    print('Successful trading dates:',downloaded,'Unavailable dates:',len(failures))
    if failures:
      err=out.with_suffix('.errors.csv');pd.DataFrame(failures,columns=['date','reason']).to_csv(err,index=False)
      print('Error log:',err)
if __name__=='__main__':main()
