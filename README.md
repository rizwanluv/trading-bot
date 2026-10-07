# Telegram Trading Assistant Bot (Bitcoin & Gold) 🪙🥇

An advanced algorithmic trading and market intelligence Telegram bot powered by **Delta Exchange** live market data, **Delta Exchange API authenticated trading execution**, and **Google Gemini AI (Multimodal Vision)**. Built with high-probability price-action methodologies following the **Gautam Jha Liquidity Strategy**.

---

## 🚀 Key Features

### 0. 🏛️ Single-Screen Institutional Command Center (`/status`, `/dashboard`, `/dash`)
- **Everything in 1 Clean Card:**
  - 🤖 **Bot Status & Mode:** Live or Paper status with 1-click controls.
  - 🌐 **SMC / ICT Market Killzone Awareness:** Real-time tracking of London Open Killzone (07:00-10:00 UTC), New York Open Killzone (12:00-15:00 UTC), London Close (15:00-17:00 UTC), and Asian Accumulation (00:00-07:00 UTC).
  - 💼 **Portfolio Performance:** Live virtual equity or live Delta balance, realized PnL, and historical win rate.
  - 📈 **Active Positions Tracker:** Live mark price, entry price, live unrealized PnL ($ and %), Breakeven status, and trailing stops.
  - 🛡️ **Institutional Risk Rules:** Capital risk cap per trade (1.5%), max open positions (5), Breakeven protection, and trailing stop offsets.
  - 🧠 **Self-Learning Desk Health:** Scoring edge, suppressed loss patterns, and adaptive sizing multipliers.
  - 🔑 **API Connectivity:** Delta Exchange authenticated status & active Google Gemini AI model.

### 1. ⚡ Advanced Institutional Risk Management (`/trade be`, `/trade trail`)
- **🎯 Auto-Trade Lot Size Configuration (`/trade size <VAL>`, `/size <VAL>`):**
  - Configure the exact lot size for all auto-trading scanners (AMD Scalper, Master Confluence, Gautam Jha, Candle Entry) and manual trades.
  - **Global Lot Size:** `/trade size 0.05` or `/size 0.05`.
  - **Per-Symbol Overrides:** `/trade size BTC 0.01` or `/size GOLD 0.5`.
  - **Strict Bot Following:** The bot strictly adheres to your configured lot size across all background auto-trading setups.
  - **Plain Text / Chat Control:** Send `set lot size 0.05`, `btc size 0.01`, or `lot size` anytime in chat to inspect or change.
  - **Clear Overrides:** `/trade size BTC reset` reverts back to the global lot size.
- **🛡️ Auto-Breakeven Protection (`/trade be [on|off]`):**
  - When Take Profit 1 (TP1) is reached, Stop Loss is automatically shifted to Entry Price to lock in a **100% risk-free trade** while the runner continues towards TP2!
- **⚡ Dynamic Trailing Stop-Loss (`/trade trail [on|off|pct]`):**
  - Automatically trails Stop Loss behind peak profit once price advances into positive territory (e.g. 1.0% trail distance).
- **🛡️ Capital Risk Management (`/trade risk <PCT>`):**
  - Position sizing is dynamically calculated using account equity and structural invalidation levels to risk strictly e.g. `1.5%` of capital per trade.
- **⚡ Multiple Concurrent Positions (`/trade maxpos <N>`, `/trade multi [on|off]`):**
  - Configurable concurrent trades cap across Bitcoin, Gold, and other assets.

### 2. ⚡ AMD Scalp Engine (1m • 5m • 15m) (`/amd`, `/scalp`)
- **Accumulation - Manipulation - Distribution Multi-Timeframe Execution:**
  - 15m Higher-Timeframe Trend and Order Flow Bias.
  - 5m Consolidation / Range Accumulation detection.
  - 5m Judas Swing Liquidity Sweep (stop hunts above range highs / below range lows).
  - 1m Sniper Market Structure Shift (MSS) displacement trigger.
  - Automated structural Stop Loss, Range TP1, Expansion TP2, and capital risk sizing.

### 3. 🤖 Automated Trading Bot Engine (`/autotrade`, `/trade`)
- **Dual Execution Modes:**
  - 🎮 **Paper Trading (Default):** Risk-free simulation with `$10,000` initial virtual capital, realistic tracking of SL/TP triggers, and realized PnL bookkeeping.
  - ⚡ **Live Delta Exchange Trading:** Direct authenticated order placement via Delta Exchange India or Global REST API v2 using HMAC-SHA256 signatures.
- **Systematic Strategy Scanner:**
  - Scans Bitcoin (`BTCUSD`) and Gold (`XAUTUSD`) on 1m, 5m, and 15m timeframes.
  - Integrates AMD Scalp Engine as Priority #1, followed by 5-Layer Confluence and Gautam Jha Liquidity setups.
  - Automatically attaches Stop Loss (SL) and dual Take-Profit targets (TP1 & TP2).
  - Background exit monitor actively watches live prices, trails stops, manages breakeven, and sends push notifications.

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

### 10. 🎯 Multi-Strategy Master Confluence (`/confluence`, `/confluencetrade`)
- Combines every quantitative strategy layer into a single consensus model:
  1. **Gautam Jha Liquidity (30%)** — DO color flips & PDH/PDL liquidity sweeps
  2. **Multi-Timeframe Candles (25%)** — 5m & 15m Pin Bars, Engulfing patterns, S/R bounces
  3. **Order Book L2 Depth (20%)** — Real-time bid/ask imbalance ratio & liquidity wall cushions
  4. **News & Macro Sentiment (15%)** — Breaking headlines with lexical sentiment scoring
  5. **Self-Learning Risk Engine (10%)** — Auto-suppression of cold setups & adaptive sizing weights
- Executes high-conviction trades when combined confluence reaches **>= 65% - 70%**.

### 11. 📖 Order Book (L2 Depth) Analysis (`/orderbook`, `/book`)
- Real-time order book analysis from Delta Exchange API.
- Bid/Ask depth ratio & imbalance percentage (`-100%` to `+100%`).
- Micro-price vs Mid-price spread calculations.
- Automatic detection of large institutional **Buy Walls** (support) and **Sell Walls** (resistance).

### 12. 📰 Breaking News & Macro Sentiment Engine (`/news`)
- Real-time financial news headlines for Bitcoin, Gold, and macroeconomic events.
- Zero-token quantitative sentiment scoring (`STRONG_BULLISH` to `STRONG_BEARISH`).
- Optional token-efficient AI macro synthesis on demand (`/news ai`).

---

## 📜 Commands Reference

### 🎯 Multi-Strategy Confluence, News & Order Book
| Command | Description |
|---|---|
| `/confluence [SYMBOL]` | Every strategy combined consensus score & layers breakdown 🟢 |
| `/confluence trade [SYMBOL]` | Execute trade using unified master confluence plan |
| `/orderbook [SYMBOL]` (or `/book`) | Live L2 orderbook depth, bid/ask imbalance & liquidity walls |
| `/news [SYMBOL]` | Real-time breaking news headlines & financial sentiment score |
| `/news ai [SYMBOL]` | Token-capped AI macro sentiment synthesis (<180 tokens) |
| `/btcconfluence` / `/goldconfluence` | Instant shortcuts for BTC / Gold confluence |
| `/btcbook` / `/goldbook` | Instant shortcuts for BTC / Gold order book depth |
| `/btcnews` / `/goldnews` | Instant shortcuts for BTC / Gold breaking news |

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
