with open('main.py', 'r') as f:
    content = f.read()

# Replace btc/gold/eth/sol and analyze/trend
content = content.replace("• /btc — Live Bitcoin ticker, range bar & 24h stats", "• /market — Global overview of BTC, ETH, SOL, GOLD\\n        • /btc — Live Bitcoin ticker, range bar & 24h stats")
content = content.replace("• /btc — Bitcoin ticker card & 24h stats", "• /market — Global overview of BTC, ETH, SOL, GOLD\\n        • /btc — Bitcoin ticker card & 24h stats")

content = content.replace("• /analyze [SYM] — 10-indicator confluence report\\n• /trend [SYM] — Multi-timeframe trend scanner", "• /scan [SYM] — Ultimate Master Scan (Trend + Levels + Entry)\\n        • /analyze [SYM] — 10-indicator confluence report\\n        • /trend [SYM] — Multi-timeframe trend scanner")
content = content.replace("• /analyze [SYM] — 10-indicator confluence report\\n• /trend [SYM] — Multi-timeframe trend scanner", "• /scan [SYM] — Ultimate Master Scan (Trend + Levels + Entry)\\n        • /analyze [SYM] — 10-indicator confluence report\\n        • /trend [SYM] — Multi-timeframe trend scanner")

with open('main.py', 'w') as f:
    f.write(content)
