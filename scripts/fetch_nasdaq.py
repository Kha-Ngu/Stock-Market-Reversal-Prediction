"""Optional Nasdaq Data Link API probe, using local env key and never exporting keys.

NSHRT is a premium dataset. A valid API key is NOT an entitlement.
"""
import os,sys,argparse
from pathlib import Path
import requests
from dotenv import load_dotenv
load_dotenv()

def main():
    key=os.environ.get('NASDAQ_DATA_LINK_API_KEY')
    if not key:raise SystemExit('Provide NASDAQ_DATA_LINK_API_KEY in .env (do not commit).')
    p=argparse.ArgumentParser();p.add_argument('--table-code',default='NSHRT',help='Verify the exact premium datatable code with your account')
    p.add_argument('--out',default='data/external/nasdaq_entitled.csv');args=p.parse_args()
    # Data Link API returns an access-denied response when a premium entitlement is absent.
    url=f'https://data.nasdaq.com/api/v3/datatables/{args.table_code}.csv'
    r=requests.get(url,params={'api_key':key,'qopts.per_page':100},timeout=45)
    print('HTTP status:',r.status_code)
    if r.status_code in (401,402,403,404):
        print('No accessible premium table with this code/key; fall back to uploaded Nasdaq snapshots.');sys.exit(2)
    r.raise_for_status()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(r.content)
    print('Saved accessible Nasdaq export to',out)
if __name__=='__main__':main()
