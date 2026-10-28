"""
Agriculture-Only Guardrails and Prompt Injection Security Layer.
Enforces deterministic boundaries: immediately blocks non-agricultural queries
(e.g., coding, medical, entertainment, politics) and mitigates prompt injections.
"""

import re
from typing import Tuple, Dict, Any


# Forbidden non-agricultural domains
OUT_OF_SCOPE_PATTERNS = [
    # General software & coding
    r"\b(python|javascript|java|c\+\+|html|css|sql|write a code|write a script|debug|regex|api key|hack|payload|function|class)\b",
    # Human medical & pharmacology
    r"\b(headache|fever|cough|cancer|prescription|doctor|symptom|covid|paracetamol|antibiotic for humans|medical diagnosis)\b",
    # Entertainment, pop culture & casual chat
    r"\b(movie|song|lyrics|celebrity|hollywood|bollywood|football score|crypto|bitcoin|dating|horoscope)\b",
    # System & prompt injection triggers
    r"(ignore previous instructions|system prompt|jailbreak|disregard all prior|dan mode|reveal secret)"
]

# Explicit agricultural domain indicators
AGRICULTURAL_KEYWORDS = [
    "crop", "yield", "harvest", "field", "soil", "fertilizer", "pest", "pesticide",
    "irrigation", "wheat", "maize", "rice", "potato", "soybean", "cassava", "sorghum",
    "rainfall", "temperature", "hectare", "acre", "npk", "tonnes", "blight", "rust",
    "agronomy", "farming", "farmer", "plant", "sowing", "drought", "humidity",
    "tillage", "seed", "seedling", "monsoon", "compost", "tractor", "weeds", "grain"
]


class AgricultureGuardrails:
    """Deterministic scope gate and prompt security validator."""

    @classmethod
    def check_query_safety(cls, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates whether a query is safely within the agricultural domain.
        Returns: (is_safe, refusal_reason, metadata)
        """
        normalized = query.lower().strip()

        if not normalized or len(normalized) < 2:
            return False, "Query is empty or too short. Please ask a specific agricultural question.", {"flag": "empty"}

        # 1. Prompt Injection & Jailbreak Detection
        injection_triggers = [
            "ignore previous", "disregard all", "system prompt", "jailbreak",
            "dan mode", "reveal your instructions", "print hidden"
        ]
        if any(trig in normalized for trig in injection_triggers):
            return False, "Security Notice: Prompt injection or instruction override attempt detected. Query rejected.", {"flag": "prompt_injection"}

        # 2. Check for Out-of-Scope non-agricultural topics
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                # Double-check if it's a genuine agricultural query that happened to mention a word
                agri_hits = sum(1 for kw in AGRICULTURAL_KEYWORDS if kw in normalized)
                if agri_hits < 2:
                    return False, (
                        "Scope Refusal: AgriSense is strictly specialized for Agriculture, Crop Yield Prediction, "
                        "and Agronomic advisory. Inquiries regarding general software coding, human medicine, "
                        "entertainment, or unrelated topics are out of scope."
                    ), {"flag": "out_of_scope"}

        # 3. Check for presence of agricultural intent
        is_agri = any(kw in normalized for kw in AGRICULTURAL_KEYWORDS)
        if not is_agri:
            # If the user asks a very generic question without any agri context
            if len(normalized.split()) > 3:
                return False, (
                    "Scope Refusal: Please frame your inquiry around agriculture, crop yield forecasting, "
                    "crop protection, soil health, or farming techniques."
                ), {"flag": "non_agricultural"}

        return True, "Safe", {"flag": "allowed"}

    @classmethod
    def sanitize_untrusted_text(cls, text: str) -> str:
        """Sanitizes text extracted from external web sources or unverified documents."""
        # Strip suspicious prompt injection patterns from ingested text
        cleaned = re.sub(r"(ignore\s+all\s+instructions|system\s+role\s*:)", "[filtered]", text, flags=re.IGNORECASE)
        return cleaned.strip()
