# Telegram Trading Assistant Bot (Bitcoin & Gold) 🪙🥇

An advanced algorithmic trading and market intelligence Telegram bot powered by **Delta Exchange** live market data, **Delta Exchange API authenticated trading execution**, and **Google Gemini AI (Multimodal Vision)**. Built with high-probability price-action methodologies following the **Gautam Jha Liquidity Strategy**.

---

## 🚀 Key Features

### 1. 🤖 Automated Trading Bot Engine (`/autotrade`)
- **Dual Execution Modes:**
  - 🎮 **Paper Trading (Default):** Risk-free simulation with `$10,000` initial virtual capital, realistic tracking of SL/TP triggers, and realized PnL bookkeeping.
  - ⚡ **Live Delta Exchange Trading:** Direct authenticated order placement via Delta Exchange India or Global REST API v2 using HMAC-SHA256 signatures.
- **Systematic Strategy Scanner:**
  - Scans Bitcoin (`BTCUSD`) and Gold (`XAUTUSD`) on 5m and 15m timeframes.
  - Detects Gautam Jha Liquidity Setups (Daily Open color flips, PDH / PDL liquidity sweeps, Break-and-Go momentum).
  - Automatically attaches Stop Loss (SL) and dual Take-Profit targets (TP1 1:1.5, TP2 1:2.5+).
  - Background exit monitor actively watches live prices and executes profit taking and stop loss exits with instant Telegram notifications.

### 2. 🧠 Autonomous Self-Learning & Auto-Improvement (`/learn`, `/insights`)
- **Zero-Token Local Intelligence:**
  - 100% of continuous performance tracking, setup scoring, auto-suppression, and drawdown cooling algorithms run locally at **0 LLM API tokens**.
  - Persistently tracks metrics across setups, symbols (BTC/XAU), and trading sessions (Asia, London, New York).
- **Capital Protection & Auto-Suppression:**
  - Automatically identifies underperforming setups (`< 40%` win rate) and suppresses them from executing, preventing repeated losses.
- **Dynamic Setup Prioritization:**
  - Automatically boosts position size (`1.2x - 1.3x`) on high-probability winning patterns (`> 60%` win rate with positive edge).
- **Adaptive Drawdown Cooldown:**
  - Temporarily halts trading on a symbol after 3 consecutive losses to avoid tilt or adverse chop regimes.
- **Adaptive Stop-Loss Buffers:**
  - Expands SL breathing room if market volatility is prematurely tagging stops before target runs.
- **Ultra-Token-Efficient AI Synthesis (`/learn ai`):**
  - On-demand quantitative reflection compressed into `< 100` prompt tokens and capped at `250` output tokens with 15-minute caching to eliminate unnecessary token usage.

### 3. 🔑 Delta Exchange API Key Authentication (`/setkeys`, `/keys`)
- Secure HMAC-SHA256 signature generation (`METHOD + TIMESTAMP + PATH + QUERY + PAYLOAD`).
- Simple setup directly in Telegram via `/setkeys <API_KEY> <API_SECRET>`.
- Automatically persists to local `.env` file (protected from Git by `.gitignore`).
- Display connection status, masked API key, and active trading mode using `/keys`.

### 3. ⚡ Manual & Automatic Order Management
- `/trade <SYMBOL> <buy|sell> [size]` — Instant trade execution with auto-calculated Gautam Jha SL and TP targets.
- `/positions` — Real-time position monitor with live mark prices, entry prices, and unrealized PnL ($ and %).
- `/closeposition <id>` / `/closeall` — Close individual or all open positions at market.
- `/balance` — View virtual account equity (Paper mode) or live Delta Exchange wallet balances (Live mode).
- `/orders` & `/cancelorders` — View and manage open orders on Delta Exchange.

### 4. 🪙 Bitcoin (BTC) & 🥇 Gold (XAU) Native Support
- Dedicated shortcuts for instant speed on mobile Telegram.
- **Market Overview:** `/price` displays a live side-by-side comparison of Bitcoin (`BTCUSD`) and Gold (`XAUTUSD`).
- **Flexible Symbol Aliasing:** Accepts `btc`, `gold`, `xau`, `eth`, `sol`, etc.
- **Smart Symbol Detection:** Automatically infers whether a target price belongs to Bitcoin (e.g. `$85,000`) or Gold (e.g. `$4,180`).

### 5. 📊 Automatic Level Analysis (`/levels`)
- Daily Classic & Fibonacci Pivot Points (P, R1-R3, S1-S3).
- 24-hour High, Low, and Fibonacci Retracements (38.2%, 50.0% Equilibrium, 61.8% Golden Zone).
- Gautam Jha Daily Open (DO), Previous Day High (PDH), and Previous Day Low (PDL) with sweep detection.
- Nearest immediate Support and Resistance boundary zones with % distance.
- Visual range bar and Trend Bias indicator.

### 6. 🎯 Gautam Jha Liquidity & Price-Action Engine (`/gj`)
- Top-down daily liquidity analysis.
- Live liquidity sweep monitoring (PDH / PDL grabs).
- 3 systematic trade setup styles:
  1. **Break-and-Go (Momentum)**
  2. **Retrace-to-Level (Daily Open pullback)**
  3. **Level Reversal / Sweep Continuation**

### 7. 🕯️ 1m, 5m, 15m Candle Entry Scanner & Alerts (`/entry`, `/watch`)
- Multi-timeframe candlestick pattern recognition: Pin Bars, Hammers, Shooting Stars, Engulfing Candles, S/R Bounces, and Breakouts.
- Automated Trade Plans: Exact Entry, Stop Loss (SL), Take Profit 1 (TP1), Take Profit 2 (TP2), and Risk-to-Reward ratio (1:1.5 to 1:2.5+).
- **Automated Background Watcher:** Continuously scans candle closes every 25 seconds and sends push alerts to Telegram when high-probability setups trigger.

### 8. 🚨 Custom Price Alerts (`/alert`, `/alerts`)
- Persistent price alerts saved to JSON store.
- Auto-detects condition (`>=` if target > current, `<=` if target < current).
- Instant Telegram notification upon price cross.

### 9. 📸 Multimodal Chart Screenshot Analysis (Gemini Vision)
- Upload any chart screenshot to the bot.
- Gemini analyzes the image according to Gautam Jha rules: identifies Daily Open, PDH, PDL, liquidity sweeps, and produces structured trade recommendations.

---

## 📜 Commands Reference

### 🤖 Trading & Delta Exchange Commands
| Command | Description |
|---|---|
| `/starttrade` (or `/tradeon`) | **START** Gautam Jha automated trading engine 🟢 |
| `/stoptrade` (or `/tradeoff`) | **STOP** / pause automated trading bot 🔴 |
| `/autotrade [on\|off]` | Control automated strategy execution and view status dashboard |
| `/mode [paper\|live]` | Switch between Paper trading ($10,000 demo) and Live Delta execution |
| `/trade <SYM> <buy\|sell> [size]` | Execute manual trade with auto SL and TP (e.g. `/trade BTC buy 0.01`) |
| `/positions` | List open positions with live mark price and unrealized PnL |
| `/closeposition <ID>` | Close specific open position by ID |
| `/closeall` | Close all active positions immediately |
| `/balance` | Check virtual balance / Delta Exchange wallet balances |
| `/orders` | View working orders on Delta Exchange |
| `/cancelorders [SYM]` | Cancel open orders on Delta Exchange |
| `/setkey <API_KEY>` | Set Delta Exchange API Key |
| `/setsecret <API_SECRET>` | Set Delta Exchange API Secret |
| `/setkeys <KEY> <SECRET>` | Configure both Delta Exchange API Key & Secret at once |
| `/keys` | View Delta API connection status and masked key |

### 🧠 Self-Learning & Optimization Commands
| Command | Description |
|---|---|
| `/learn` | View self-learning dashboard, win-rates & setup calibrations (0 tokens) 🟢 |
| `/learn ai` | Ultra-compact quantitative AI review (&lt;250 tokens) |
| `/learn reset` | Reset learning memory and recalibrate from scratch |
| `/insights` | Quick summary of strategy improvements & calibrations |

### 🤖 Google Gemini AI & Model Commands
| Command | Description |
|---|---|
| `/model [MODEL]` | View or switch Gemini model (e.g. `/model gemini-2.5-flash`, `/model pro`) |
| `/setgemini <KEY>` | Configure Google Gemini API Key directly in chat |
| `/gemini` | Check AI status and active model |

### 🔔 Automatic Market Alerts
| Command | Description |
|---|---|
| `/alertson` | **TURN ON** automatic alerts for BTC & Gold (5m & 15m) 🟢 |
| `/alertsoff` | **TURN OFF** automatic market alerts 🔴 |
| `/autoalert [on\|off]` | Toggle automatic candle & liquidity alerts |

### 🪙 Bitcoin (BTC) Shortcuts
| Command | Description |
|---|---|
| `/btc` | Live Bitcoin ticker & 24h stats |
| `/btclevels` | Bitcoin Level Analysis (Pivots, Fibs, S/R) |
| `/btcgj` | Bitcoin Gautam Jha Liquidity (DO, PDH, PDL sweeps) |
| `/btcentry` | Bitcoin 1m, 5m, 15m candle entry scan |
| `/btcwatch` | Enable automated candle alerts for Bitcoin |

### 🥇 Gold (XAU/USD) Shortcuts
| Command | Description |
|---|---|
| `/gold`, `/xau`, `/xauusd` | Live Gold ticker & 24h stats |
| `/goldlevels`, `/xaulevels` | Gold Level Analysis (Pivots, Fibs, S/R, DO) |
| `/goldgj`, `/xaugj` | Gold Gautam Jha Liquidity (DO, PDH, PDL sweeps) |
| `/goldentry`, `/xauentry` | Gold 1m, 5m, 15m candle entry scan |
| `/goldwatch`, `/xauwatch` | Enable automated candle alerts for Gold |

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
- Delta Exchange India or Global API Key (for live trading execution)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

```env
TELEGRAM_TOKEN=your_telegram_bot_token_here
GEMINI_API_KEY=your_gemini_api_key_here

# Delta Exchange Credentials
DELTA_API_KEY=your_delta_api_key_here
DELTA_API_SECRET=your_delta_api_secret_here
DELTA_BASE_URL=https://api.india.delta.exchange

# Trading Configuration
TRADING_MODE=paper
DEFAULT_ORDER_SIZE=1
```

*Note: You can also configure Delta Exchange API keys directly inside Telegram using `/setkeys <KEY> <SECRET>`!*

### 4. Run Tests
```bash
python3 test_bot.py -v
```

### 5. Start the Bot
```bash
python3 main.py
```

---

## 🔒 Security & Safety
- **Safe by Default:** The bot defaults to `paper` trading mode. Real orders are only submitted to Delta Exchange when you explicitly configure API keys and switch mode with `/mode live`.
- **Credential Protection:** Secrets are never committed to version control (`.gitignore` protects `.env` and all trade store JSON files).
- **Telegram Privacy:** When displaying credentials in Telegram via `/keys`, keys are masked (e.g. `d_ke...987`).
