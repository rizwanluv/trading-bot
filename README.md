# Telegram Trading Assistant Bot (Bitcoin & Gold) 🪙🥇

An advanced algorithmic trading and market intelligence Telegram bot powered by **Delta Exchange (India)** live market data and **Google Gemini AI (Multimodal Vision)**. Built with high-probability price-action methodologies following the **Gautam Jha Liquidity Strategy**.

---

## 🚀 Key Features

### 1. 🪙 Bitcoin (BTC) & 🥇 Gold (XAU) Native Support
- Dedicated shortcuts for instant speed on mobile Telegram.
- **Market Overview:** `/price` displays a live side-by-side comparison of Bitcoin (`BTCUSD`) and Gold (`XAUTUSD`).
- **Flexible Symbol Aliasing:** Accepts `btc`, `gold`, `xau`, `eth`, `sol`, etc.
- **Smart Symbol Detection:** Automatically infers whether a target price belongs to Bitcoin (e.g. `$85,000`) or Gold (e.g. `$4,180`).

### 2. 📊 Automatic Level Analysis (`/levels`)
- Daily Classic & Fibonacci Pivot Points (P, R1-R3, S1-S3).
- 24-hour High, Low, and Fibonacci Retracements (38.2%, 50.0% Equilibrium, 61.8% Golden Zone).
- Gautam Jha Daily Open (DO), Previous Day High (PDH), and Previous Day Low (PDL) with sweep detection.
- Nearest immediate Support and Resistance boundary zones with % distance.
- Visual range bar and Trend Bias indicator.

### 3. 🎯 Gautam Jha Liquidity & Price-Action Engine (`/gj`)
- Top-down daily liquidity analysis.
- Live liquidity sweep monitoring (PDH / PDL grabs).
- 3 systematic trade setup styles:
  1. **Break-and-Go (Momentum)**
  2. **Retrace-to-Level (Daily Open pullback)**
  3. **Level Reversal / Sweep Continuation**

### 4. 🕯️ 1m, 5m, 15m Candle Entry Scanner & Alerts (`/entry`, `/watch`)
- Multi-timeframe candlestick pattern recognition: Pin Bars, Hammers, Shooting Stars, Engulfing Candles, S/R Bounces, and Breakouts.
- Automated Trade Plans: Exact Entry, Stop Loss (SL), Take Profit 1 (TP1), Take Profit 2 (TP2), and Risk-to-Reward ratio (1:1.5 to 1:2.5+).
- **Automated Background Watcher:** Continuously scans candle closes every 25 seconds and sends push alerts to Telegram when high-probability setups trigger.

### 5. 🚨 Custom Price Alerts (`/alert`, `/alerts`)
- Persistent price alerts saved to JSON store.
- Auto-detects condition (`>=` if target > current, `<=` if target < current).
- Instant Telegram notification upon price cross.

### 6. 📸 Multimodal Chart Screenshot Analysis (Gemini Vision)
- Upload any chart screenshot to the bot.
- Gemini analyzes the image according to Gautam Jha rules: identifies Daily Open, PDH, PDL, liquidity sweeps, and produces structured trade recommendations.

---

## 📜 Commands Reference

### 🪙 Bitcoin (BTC) Shortcuts
| Command | Description |
|---|---|
| `/btc` | Live Bitcoin ticker & 24h stats |
| `/btclevels` | Bitcoin Level Analysis (Pivots, Fibs, S/R) |
| `/btcgj` | Bitcoin Gautam Jha Liquidity (DO, PDH, PDL sweeps) |
| `/btcentry` | Bitcoin 1m, 5m, 15m candle entry scan |
| `/btcwatch` | Enable automated candle alerts for Bitcoin |

### 🥇 Gold (XAU) Shortcuts
| Command | Description |
|---|---|
| `/gold` or `/xau` | Live Gold ticker & 24h stats |
| `/goldlevels` | Gold Level Analysis |
| `/goldgj` | Gold Gautam Jha Liquidity |
| `/goldentry` | Gold 1m, 5m, 15m candle entry scan |
| `/goldwatch` | Enable automated candle alerts for Gold |

### 💹 General Market & Commands
| Command | Description |
|---|---|
| `/price` | Live overview of BTC & Gold |
| `/price [SYMBOL]` | Live ticker for any coin (e.g. `/price ETH`) |
| `/levels [SYMBOL]` | Automatic S/R, Pivots, Fibs |
| `/gj [SYMBOL]` | Gautam Jha Price Action & Liquidity |
| `/alert [SYM] <PR>` | Set price alert (e.g. `/alert btc 85000` or `/alert 4180`) |
| `/alerts` | List active price alerts |
| `/delalert <ID>` | Remove a price alert |
| `/clearalerts` | Clear all active price alerts |
| `/entry [SYMBOL]` | Scan 1m, 5m, 15m candles |
| `/watch [SYM] [TFS]`| Turn ON automated candle alerts (e.g. `/watch btc 5m,15m`) |
| `/unwatch [SYM]` | Turn OFF automated candle alerts |
| `/watchers` | List active candle scanners |
| `/list` | Show full commands directory |
| `/help` | Detailed instructions & examples |
| `/reset` | Clear AI conversation history |

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Gemini API Key (optional, for AI chat and chart photo vision)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file:
```env
TELEGRAM_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run Tests
```bash
python3 test_bot.py -v
```

### 5. Start the Bot
```bash
python3 main.py
```
