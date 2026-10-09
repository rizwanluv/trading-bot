import os, requests, time
url = "https://api.india.delta.exchange/v2/history/candles?symbol=BTCUSD&resolution=15m&start=1791443219&end=1791533219"
r = requests.get(url).json()
if r.get('result'):
    print(r['result'][0])
