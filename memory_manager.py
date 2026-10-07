"""
Autonomous Cognitive Memory & Procedural Rule Adaptation Engine for Trading Bot.
Provides asynchronous reflection and consolidation loop to analyze dialogue,
detect user corrections/preferences/frustrations, and dynamically evolve bot behavioral instructions.
"""

import os
import re
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Check if google-genai SDK is available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
except ImportError:
    HAS_GENAI_SDK = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _clean_json_text(raw_text: str) -> str:
    """Strip markdown code block fences and trailing formatting from LLM JSON output."""
    if not raw_text:
        return "{}"
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


# ==================== Schema for Memory Consolidation ====================

class MemoryExtraction(BaseModel):
    """Extracted cognitive state from dialogue reflection."""
    new_facts: List[str] = Field(
        default_factory=list,
        description="New factual data about the user, their accounts, or explicit preferences (Semantic Memory)."
    )
    conflicts_to_remove: List[str] = Field(
        default_factory=list,
        description="Outdated facts or superseded rules that should be invalidated or removed."
    )
    behavioral_corrections: List[str] = Field(
        default_factory=list,
        description="Procedural rules to adapt bot behavior based on user corrections, preferences, or feedback."
    )


# ==================== Reflection & Consolidation Functions ====================

def reflect_and_consolidate(
    recent_dialogue: str,
    existing_rules: List[str],
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-flash",
) -> MemoryExtraction:
    """
    Asynchronous consolidation loop that runs in the background.
    Analyzes conversation, detects corrections, and evolves bot instructions.
    Uses Google Gemini with fallback models and structured JSON schema output.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        logger.debug("GEMINI_API_KEY not configured; skipping AI reflection.")
        return MemoryExtraction()

    consolidation_prompt = f"""
Analyze the dialogue below. Extract:
1. New factual data about the user (Semantic Memory).
2. Outdated facts that should be invalidated.
3. Behavioral adaptations (Procedural Memory): Did the user express a preference, 
   correct a mistake, or show frustration? Formulate dynamic rules for future turns.

Existing Rules: {existing_rules}
Dialogue:
{recent_dialogue}
"""

    candidate_models = [model]
    for fallback in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    # 1. Try google-genai SDK if present
    if HAS_GENAI_SDK:
        for m_name in candidate_models:
            try:
                ai_client = genai.Client(api_key=key)
                response = ai_client.models.generate_content(
                    model=m_name,
                    contents=consolidation_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": MemoryExtraction
                    }
                )
                if response and response.text:
                    cleaned_str = _clean_json_text(response.text)
                    parsed = MemoryExtraction.model_validate_json(cleaned_str)
                    logger.info(f"Memory consolidation succeeded via SDK model {m_name}")
                    return parsed
            except Exception as e:
                logger.warning(f"SDK reflection with {m_name} failed: {e}. Trying next candidate.")

    # 2. REST API fallback
    if HAS_REQUESTS:
        schema = {
            "type": "OBJECT",
            "properties": {
                "new_facts": {"type": "ARRAY", "items": {"type": "STRING"}},
                "conflicts_to_remove": {"type": "ARRAY", "items": {"type": "STRING"}},
                "behavioral_corrections": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["new_facts", "conflicts_to_remove", "behavioral_corrections"],
        }
        for m_name in candidate_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={key}"
                payload = {
                    "contents": [{"parts": [{"text": consolidation_prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": schema,
                        "maxOutputTokens": 800,
                    },
                }
                r = requests.post(url, json=payload, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_json = parts[0].get("text", "{}")
                            cleaned_str = _clean_json_text(raw_json)
                            parsed = MemoryExtraction.model_validate_json(cleaned_str)
                            logger.info(f"Memory consolidation succeeded via REST model {m_name}")
                            return parsed
                elif r.status_code in (400, 404):
                    logger.debug(f"REST reflection model {m_name} returned HTTP {r.status_code}")
            except Exception as e:
                logger.warning(f"REST reflection with {m_name} error: {e}")

    return MemoryExtraction()


# ==================== Persistent Memory Manager ====================

class MemoryManager:
    """
    Manages persistent Semantic and Procedural Memory, orchestrates asynchronous
    background consolidation, and injects evolved instructions into system prompts.
    """

    def __init__(self, store_path: str = "memory_store.json"):
        self.store_path = store_path
        self.semantic_facts: List[str] = []
        self.procedural_rules: List[str] = []
        self.dialogue_history: List[Dict[str, str]] = []
        self.last_consolidated_at: float = 0.0
        self.consolidation_count: int = 0
        self._is_consolidating: bool = False
        self._turns_since_consolidation: int = 0
        self.load()

    def load(self):
        """Load memory state from JSON storage."""
        if not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.semantic_facts = data.get("semantic_facts", [])
            self.procedural_rules = data.get("procedural_rules", [])
            self.dialogue_history = data.get("dialogue_history", [])
            self.last_consolidated_at = data.get("last_consolidated_at", 0.0)
            self.consolidation_count = data.get("consolidation_count", 0)
        except Exception as e:
            logger.error(f"Failed to load memory store from {self.store_path}: {e}")

    def save(self):
        """Save memory state to JSON storage."""
        data = {
            "semantic_facts": self.semantic_facts,
            "procedural_rules": self.procedural_rules,
            "dialogue_history": self.dialogue_history[-30:],  # keep recent dialogue
            "last_consolidated_at": self.last_consolidated_at,
            "consolidation_count": self.consolidation_count,
        }
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory store to {self.store_path}: {e}")

    def record_interaction(self, user_text: str, bot_reply: Optional[str] = None):
        """
        Record a dialogue turn and increment consolidation counter.
        Returns True if automatic background consolidation was scheduled.
        """
        now = time.time()
        self.dialogue_history.append({"role": "user", "text": user_text, "timestamp": now})
        if bot_reply:
            self.dialogue_history.append({"role": "assistant", "text": bot_reply[:200], "timestamp": now})
        
        # Keep sliding window
        self.dialogue_history = self.dialogue_history[-30:]
        self._turns_since_consolidation += 1
        self.save()

    def should_consolidate(self) -> bool:
        """Check if conditions are met for an automatic background consolidation."""
        if self._is_consolidating:
            return False
        # Consolidate every 4 user turns, or after 10 minutes with at least 2 turns
        now = time.time()
        if self._turns_since_consolidation >= 4:
            return True
        if self._turns_since_consolidation >= 2 and (now - self.last_consolidated_at) > 600:
            return True
        return False

    def consolidate(self, force: bool = False, model: str = "gemini-2.5-flash") -> MemoryExtraction:
        """
        Synchronous execution of reflection and consolidation.
        Updates semantic facts and procedural rules while invalidating conflicts.
        """
        if self._is_consolidating and not force:
            return MemoryExtraction()

        if len(self.dialogue_history) < 2 and not force:
            return MemoryExtraction()

        self._is_consolidating = True
        try:
            # Build dialogue transcript
            lines = []
            for item in self.dialogue_history[-16:]:
                role = item.get("role", "user").capitalize()
                text = item.get("text", "")
                lines.append(f"{role}: {text}")
            dialogue_str = "\n".join(lines)

            extraction = reflect_and_consolidate(
                recent_dialogue=dialogue_str,
                existing_rules=self.procedural_rules,
                model=model,
            )

            # Apply removals / conflicts
            if extraction.conflicts_to_remove:
                for conf in extraction.conflicts_to_remove:
                    conf_lower = conf.lower().strip()
                    self.procedural_rules = [
                        r for r in self.procedural_rules
                        if conf_lower not in r.lower()
                    ]
                    self.semantic_facts = [
                        f for f in self.semantic_facts
                        if conf_lower not in f.lower()
                    ]

            # Apply new facts
            for fact in extraction.new_facts:
                clean_fact = fact.strip()
                if clean_fact and clean_fact not in self.semantic_facts:
                    self.semantic_facts.append(clean_fact)

            # Apply new procedural rules (behavioral corrections)
            for rule in extraction.behavioral_corrections:
                clean_rule = rule.strip()
                if clean_rule and clean_rule not in self.procedural_rules:
                    self.procedural_rules.append(clean_rule)

            # Cap lists to prevent unbounded growth
            self.semantic_facts = self.semantic_facts[-25:]
            self.procedural_rules = self.procedural_rules[-25:]

            self.last_consolidated_at = time.time()
            self.consolidation_count += 1
            self._turns_since_consolidation = 0
            self.save()
            return extraction
        finally:
            self._is_consolidating = False

    async def consolidate_async(self, force: bool = False, model: str = "gemini-2.5-flash") -> MemoryExtraction:
        """Asynchronous wrapper for background execution."""
        return await asyncio.to_thread(self.consolidate, force=force, model=model)

    def get_system_instructions_injection(self) -> str:
        """
        Format consolidated procedural rules and semantic facts for injection into system prompts.
        """
        if not self.procedural_rules and not self.semantic_facts:
            return ""

        parts = ["\n[DYNAMIC PROCEDURAL MEMORY & ADAPTED BEHAVIOR RULES]"]
        parts.append("The bot has autonomously evolved the following rules based on user corrections and preferences:")
        if self.procedural_rules:
            parts.append("Dynamic Rules:")
            for r in self.procedural_rules:
                parts.append(f"• {r}")
        if self.semantic_facts:
            parts.append("User Facts & Preferences:")
            for f in self.semantic_facts:
                parts.append(f"• {f}")
        parts.append("[Strictly follow these evolved behavioral guidelines in all responses and actions]\n")
        return "\n".join(parts)

    def format_memory_report(self) -> str:
        """Format an HTML status report of cognitive memory and procedural rules."""
        lines = [
            "🧠 <b>Cognitive Memory & Behavioral Rules</b>\n",
            f"• <b>Total Reflections:</b> <code>{self.consolidation_count}</code>",
            f"• <b>Active Rules:</b> <code>{len(self.procedural_rules)}</code>",
            f"• <b>Learned Facts:</b> <code>{len(self.semantic_facts)}</code>",
            f"• <b>Recent Dialogue Turns:</b> <code>{len(self.dialogue_history)}</code>\n",
        ]

        if self.procedural_rules:
            lines.append("⚡ <b>Active Behavioral Rules (Procedural):</b>")
            for i, r in enumerate(self.procedural_rules, 1):
                lines.append(f"{i}. {r}")
            lines.append("")
        else:
            lines.append("⚡ <i>No behavioral rules evolved yet. The bot will learn as you chat and correct it.</i>\n")

        if self.semantic_facts:
            lines.append("📌 <b>User & Account Facts (Semantic):</b>")
            for f in self.semantic_facts:
                lines.append(f"• {f}")
            lines.append("")

        lines.append("<b>Commands:</b>")
        lines.append("• <code>/reflect</code> (or <code>/memory reflect</code>) — Trigger instant reflection")
        lines.append("• <code>/rules add &lt;RULE&gt;</code> — Manually add a behavioral rule")
        lines.append("• <code>/rules reset</code> (or <code>/memory reset</code>) — Reset cognitive memory")
        return "\n".join(lines)

    def add_rule(self, rule: str):
        """Manually inject a procedural rule."""
        clean = rule.strip()
        if clean and clean not in self.procedural_rules:
            self.procedural_rules.append(clean)
            self.save()

    def remove_rule(self, rule: str) -> bool:
        """Manually remove a procedural rule."""
        clean = rule.strip().lower()
        orig_len = len(self.procedural_rules)
        self.procedural_rules = [r for r in self.procedural_rules if clean not in r.lower()]
        if len(self.procedural_rules) != orig_len:
            self.save()
            return True
        return False

    def reset(self):
        """Reset all cognitive memory and procedural rules."""
        self.semantic_facts.clear()
        self.procedural_rules.clear()
        self.dialogue_history.clear()
        self.last_consolidated_at = 0.0
        self.consolidation_count = 0
        self._turns_since_consolidation = 0
        self.save()


    def record_trade_reflection(self, trade_summary: Dict[str, Any]):
        """
        Record and consolidate procedural lessons from a closed trade into memory.
        Enables continuous self-learning across consecutive auto-trades.
        """
        sym = trade_summary.get("symbol", "TRADE")
        strat = trade_summary.get("strategy", "Auto Execution")
        pnl = trade_summary.get("pnl", 0.0)
        is_win = bool(trade_summary.get("is_win", False))
        reason = trade_summary.get("exit_reason", "")

        lesson = f"Trade #{trade_summary.get('id', 0)} ({sym} {strat}): {'WIN' if is_win else 'LOSS'} ${pnl:+,.2f} via {reason}."
        if not is_win and ("STOP" in reason.upper() or pnl < 0):
            adaptation = f"Adaptive rule: Require stronger institutional confluence confirmation on {strat} before entry after stop loss hit."
            if adaptation not in self.procedural_rules:
                self.procedural_rules.append(adaptation)
                self.procedural_rules = self.procedural_rules[-25:]
        elif is_win and pnl > 0:
            fact = f"High-probability edge confirmed: {strat} on {sym} yielded positive return (${pnl:+,.2f})."
            if fact not in self.semantic_facts:
                self.semantic_facts.append(fact)
                self.semantic_facts = self.semantic_facts[-25:]
        self.save()


# Singleton instance
memory_manager = MemoryManager()

