import requests, time
now = int(time.time())
from_ts = now - (100 * 60 * 15)
url = f"https://api.india.delta.exchange/v2/chart/history?symbol=BTCUSD&resolution=15&from={from_ts}&to={now}"
print(url)
r = requests.get(url).json()
print("KEYS:", r.keys())
print("RESULT TYPE:", type(r.get('result')))
if isinstance(r.get('result'), dict):
    print("RESULT KEYS:", r['result'].keys())
    print("LEN C:", len(r['result'].get('c', [])))
