"""pricing.py

Token pricing helpers for local eval.

This module exists only to provide a rough, comparable cost estimate across models.

Notes:
- It is an estimate: ignores cached-input discounts and any gateway-side adjustments.
- We assume text token billing with separate input/output rates.
- For providers that return different usage shapes, the caller should normalize to:
  {prompt_tokens, completion_tokens}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelPrice:
    input_per_1m: float
    output_per_1m: float


# Prices are USD per 1M tokens.
# Keep this explicit and update when providers change public pricing.
_PRICES: dict[str, ModelPrice] = {
    # OpenAI (selected models used in this repo)
    'gpt-5.2': ModelPrice(input_per_1m=1.75, output_per_1m=14.00),
    'gpt-5.2-pro': ModelPrice(input_per_1m=21.00, output_per_1m=168.00),
    'gpt-5-mini': ModelPrice(input_per_1m=0.25, output_per_1m=2.00),
    'gpt-4o': ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    'gpt-4o-mini': ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    'o4-mini': ModelPrice(input_per_1m=1.10, output_per_1m=4.40),

    # Anthropic (Claude)
    # Prices from Anthropic pricing docs (base input/output tokens).
    # Model IDs vary; we include common aliases + a few dated IDs.
    'claude-opus-4-1': ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    'claude-opus-4': ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    'claude-sonnet-4': ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    'claude-sonnet-3-7': ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    'claude-haiku-3-5': ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
    'claude-haiku-3': ModelPrice(input_per_1m=0.25, output_per_1m=1.25),

    # Some observed dated IDs (map to the same rates as their family).
    'claude-opus-4-1-20250805': ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    'claude-opus-4-20250514': ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    'claude-sonnet-4-20250514': ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    'claude-haiku-4-20250514': ModelPrice(input_per_1m=1.00, output_per_1m=5.00),

    # Sonnet 4.5: commonly priced the same as Sonnet 4.
    'claude-sonnet-4-5': ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    'claude-sonnet-4-5-20250929': ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
}


def _normalize_model(model: str) -> str:
    m = (model or '').strip().lower()
    # Strip snapshot/date suffix if present, keep the base alias we know.
    for base in sorted(_PRICES.keys(), key=len, reverse=True):
        if m == base or m.startswith(base + '-'):
            return base
    return m


def price_for_model(model: str) -> Optional[ModelPrice]:
    return _PRICES.get(_normalize_model(model))


def estimate_cost_usd(model: str, usage: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Estimate USD cost from an OpenAI-like usage dict.

    usage should have prompt_tokens and completion_tokens (ints).
    """
    p = price_for_model(model)
    pt = int(usage.get('prompt_tokens') or 0)
    ct = int(usage.get('completion_tokens') or 0)

    if not p:
        return 0.0, {
            'error': 'unknown_model',
            'model': str(model),
            'prompt_tokens': pt,
            'completion_tokens': ct,
        }

    cost = (pt / 1_000_000.0) * p.input_per_1m + (ct / 1_000_000.0) * p.output_per_1m
    return float(cost), {
        'model': _normalize_model(model),
        'prompt_tokens': pt,
        'completion_tokens': ct,
        'input_per_1m': p.input_per_1m,
        'output_per_1m': p.output_per_1m,
    }
