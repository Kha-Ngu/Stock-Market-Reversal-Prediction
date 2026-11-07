"""Optional authenticated FINRA query API explorer, credentials from .env only.
Never put actual secrets in the repository, archives, or command line.

NOTE: FINRA consolidatedShortInterest dataset may refer to OTC securities only;
not a guaranteed source of listed-stock short interest.
"""
import os, sys, csv, io, json, argparse
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
load_dotenv()
TOKEN_URL='https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token?grant_type=client_credentials'
API='https://api.finra.org'

def token():
    cid=os.getenv('FINRA_CLIENT_ID'); secret=os.getenv('FINRA_CLIENT_SECRET')
    if not cid or not secret:raise SystemExit('FINRA_CLIENT_ID and FINRA_CLIENT_SECRET required in .env')
    resp=requests.post(TOKEN_URL,auth=HTTPBasicAuth(cid,secret),timeout=30);resp.raise_for_status()
    return resp.json()['access_token']

def query(name,group='otcMarket',limit=100,filters=None):
    t=token()
    url=f'{API}/data/group/{group}/name/{name}'
    payload={'limit':min(limit,5000)}
    if filters:payload['compareFilters']=filters
    resp=requests.post(url,headers={'Authorization':'Bearer '+t,'Accept':'application/json','Content-Type':'application/json'},json=payload,timeout=90)
    print('FINRA HTTP status:',resp.status_code)
    if not resp.ok:
        print('Request did not succeed. Verify dataset entitlement; response excerpt:',resp.text[:400]);resp.raise_for_status()
    return resp.json() if resp.status_code!=204 else []

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['consolidatedShortInterest','regShoDaily'],default='consolidatedShortInterest')
    p.add_argument('--group',default='otcMarket');p.add_argument('--out',default='data/external/finra_api_probe.json')
    p.add_argument('--limit',type=int,default=100);args=p.parse_args()
    result=query(args.dataset,args.group,args.limit)
    output=Path(args.out);output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(result,indent=2))
    print(f'Saved {len(result)} rows to {output}. Verify exchange coverage before feature integration.')
