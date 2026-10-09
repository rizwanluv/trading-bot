import os
import requests
import time
import pandas as pd

DELTA_CHART_API = os.getenv("DELTA_CHART_API", "https://api.india.delta.exchange/v2/history/candles")

def test_fetch(symbol="BTCUSD", res="15m"):
    now = int(time.time())
    mult = 60 * 15
    if res == "1h":
        mult = 3600
    from_ts = now - (100 * mult)
    url = f"{DELTA_CHART_API}?symbol={symbol}&resolution={res}&start={from_ts}&end={now}"
    print("URL:", url)
    resp = requests.get(url, timeout=10)
    print("STATUS:", resp.status_code)
    try:
        data = resp.json()
        res_data = data.get("result", {})
        if res_data:
            print("LEN:", len(res_data))
            if len(res_data) > 0:
                print("FIRST CLOSE:", res_data[0].get("close"))
        else:
            print("NO RESULT:", data)
    except Exception as e:
        print("ERROR:", e)

test_fetch("BTCUSD", "15m")
test_fetch("BTCUSD", "1h")
