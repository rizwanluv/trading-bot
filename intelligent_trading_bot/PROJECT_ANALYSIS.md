# Full Project Analysis – Intelligent Trading Bot

**Source**: intelligent-trading-bot-master (commit 0f8748d)  
**Analysis date**: 9 October 2026

## 1. High-Level Summary

This is a mature research + production framework for building ML-based crypto trading signals and automated trading.  
It is **not** a simple strategy bot; it is a full **feature → label → model → signal → execution** pipeline with strong emphasis on reproducibility between offline research and online live trading.

## 2. Core Strengths

- Perfect offline ↔ online feature consistency (the hardest part of most trading systems).
- Extremely flexible configuration-driven design.
- Rolling walk-forward training (`predict_rolling`) – one of the best backtesting practices.
- Pluggable everything: features, labels, models, signals, notifiers, traders.
- Already battle-tested (public Telegram channel running for years).

## 3. Architecture Deep Dive

```
Data Sources (Binance / MT5 / Yahoo)
        ↓
   Collector + Merge  →  single continuous DataFrame
        ↓
   Feature Generators (TA-Lib, statistical, custom)
        ↓
   Label Generators (future high/low, top/bottom)
        ↓
   ML Training (lc / gb / nn / svc)
        ↓
   Prediction + Signal Aggregation
        ↓
   Output Adapters (Telegram, real trader, diagrams)
```

Online service simply re-uses the exact same generators on a sliding window.

## 4. Important Implementation Notes

- All intermediate results are stored as columns in one big DataFrame (column-oriented design).
- Models are stored in a dedicated folder and loaded by name.
- `App.py` + `server.py` form a clean APScheduler-based service.
- Feature generators can be pure Python functions referenced by string path.

## 5. Risk & Practical Considerations for Personal Use

| Risk                        | Mitigation                                      |
|-----------------------------|-------------------------------------------------|
| Overfitting                 | Always use `predict_rolling` + `simulate`      |
| Look-ahead bias             | Labels are strictly future-based                |
| API rate limits             | Built-in overlap + append logic                 |
| TA-Lib installation pain    | Use pre-built wheels or conda                   |
| Real money loss             | Start with Telegram only + paper trading        |

## 6. Recommended Personal Development Path

**Phase 1 – Signals only (1–2 weeks)**  
- Run offline pipeline on BTCUSDT 1min  
- Tune thresholds until signals look reasonable  
- Connect Telegram

**Phase 2 – Robust backtesting (2–4 weeks)**  
- Full walk-forward  
- Add more features (volume, volatility regimes)  
- Test different label thresholds

**Phase 3 – Paper trading**  
- Enable trader with size = 0 or very small  
- Monitor for at least 2 weeks

**Phase 4 – Live (optional)**  
- Real small size + strict risk rules

## 7. Files Worth Studying First

1. `common/gen_features.py` + `gen_labels_highlow.py`
2. `common/classifier_*.py`
3. `service/server.py` + `App.py`
4. `outputs/notifier_scores.py` + `trader_binance.py`
5. Sample configs in `configs/`

---

This analysis + the personal starter kit in this folder give you a solid, clean base to build *your own* trading system on top of a proven foundation.
