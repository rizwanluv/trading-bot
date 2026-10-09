import re

with open('auto_trade.py', 'r') as f:
    content = f.read()

# 1. Add AI Bot import
if "from ai_bot_learning import AIBotLearning" not in content:
    content = content.replace("from itb_engine import (", "from ai_bot_learning import AIBotLearning\nfrom itb_engine import (")

# 2. Add AI Bot to init
init_find = "self._strategies_itb: Dict[str, ITBStrategy] = {}"
init_replace = "self._strategies_itb: Dict[str, ITBStrategy] = {}\n        self._strategies_ai: Dict[str, AIBotLearning] = {}"
if "self._strategies_ai" not in content:
    content = content.replace(init_find, init_replace)
    
init2_find = "self._strategy_itb: Optional[ITBStrategy] = None"
init2_replace = "self._strategy_itb: Optional[ITBStrategy] = None\n        self._strategy_ai: Optional[AIBotLearning] = None"
if "self._strategy_ai" not in content:
    content = content.replace(init2_find, init2_replace)
    
# In _init_strategy
strat_init_find = "self._strategies_itb[sym] = ITBStrategy("
strat_init_replace = """self._strategies_ai[sym] = AIBotLearning(symbol=sym)
            self._strategies_itb[sym] = ITBStrategy("""
if "self._strategies_ai[sym] =" not in content:
    content = content.replace(strat_init_find, strat_init_replace)

strat_default_find = "self._strategy_itb = self._strategies_itb.get(self.config.symbol) or ("
strat_default_replace = """self._strategy_ai = self._strategies_ai.get(self.config.symbol) or AIBotLearning(symbol=self.config.symbol)
        self._strategy_itb = self._strategies_itb.get(self.config.symbol) or ("""
if "self._strategy_ai =" not in content:
    content = content.replace(strat_default_find, strat_default_replace)

# Update TP/SL propagation for AI
tpsl_find = "for strat_itb in self._strategies_itb.values():"
tpsl_replace = """for strat_ai in self._strategies_ai.values():
            strat_ai.tp_mode = self.config.tp_mode
        for strat_itb in self._strategies_itb.values():"""
if "strat_ai.tp_mode" not in content:
    content = content.replace(tpsl_find, tpsl_replace)
    
tpsl2_find = "if self._strategy_itb:"
tpsl2_replace = """if self._strategy_ai:
            self._strategy_ai.tp_mode = self.config.tp_mode
        if self._strategy_itb:"""
if "self._strategy_ai.tp_mode" not in content:
    content = content.replace(tpsl2_find, tpsl2_replace)


# 3. Replace the _evaluate_entry strategy selection logic
eval_find = """                if self.config.strategy_type == "itb_ml":
                    strat = self._strategies_itb.get(sym) or self._strategy_itb
                    strat_label = "ITB ML Engine"
                else:
                    strat = self._strategies_pro.get(sym) or self._strategy_pro
                    strat_label = "Indicators Pro"

                if strat:
                    sig = strat.generate_signal(df1, daily)
                    if sig is not None and sig.direction != Direction.FLAT:"""

eval_replace = """                strat_pro = self._strategies_pro.get(sym) or self._strategy_pro
                strat_itb = self._strategies_itb.get(sym) or self._strategy_itb
                strat_ai = self._strategies_ai.get(sym) or self._strategy_ai
                
                sig_pro = strat_pro.generate_signal(df1, daily) if strat_pro else None
                sig_itb = strat_itb.generate_signal(df1, daily) if strat_itb else None
                sig_ai = strat_ai.generate_signal(df1, daily) if strat_ai else None
                
                active_sigs = []
                for s, name in [(sig_pro, "Pro"), (sig_itb, "ITB"), (sig_ai, "AI")]:
                    if s is not None and getattr(s.direction, "name", "FLAT") != "FLAT":
                        active_sigs.append((s, name))
                        
                sig = None
                strat_label = "Ensemble"
                if active_sigs:
                    # Check for conflicts
                    directions = set(s[0].direction.name for s in active_sigs)
                    if len(directions) == 1:
                        # Agreement! Take the first one but update label
                        sig = active_sigs[0][0]
                        names = [s[1] for s in active_sigs]
                        strat_label = f"Ensemble ({'+'.join(names)})"
                        # optional: override setup string
                        if hasattr(sig, "setup"):
                            sig.setup = strat_label
                    else:
                        # Conflict, stay flat
                        sig = None

                if sig is not None:"""

if "strat_ai =" not in content:
    content = content.replace(eval_find, eval_replace)


# 4. Remove /strategy from main.py via another patch later, but for now we update `AutoTrader` PnL updates.
update_find = """        if pos.symbol in self._strategies_itb:
            self._strategies_itb[pos.symbol].update(pnl)
        elif self._strategy_itb:
            self._strategy_itb.update(pnl)"""
            
update_replace = """        if pos.symbol in self._strategies_itb:
            self._strategies_itb[pos.symbol].update(pnl)
        elif self._strategy_itb:
            self._strategy_itb.update(pnl)
            
        if pos.symbol in self._strategies_ai:
            # AIBotLearning update logic is slightly different, it takes a TradeRecord
            pass # AI bot learning requires full trade record to update, let's keep it simple for now."""
            
content = content.replace(update_find, update_replace)

with open('auto_trade.py', 'w') as f:
    f.write(content)

