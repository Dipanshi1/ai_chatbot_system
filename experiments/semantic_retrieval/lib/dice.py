"""
Exact content-token Dice overlap and benchmark population/bucket derivation.

Uses the Stage A lexical definitions read-only (spec §3.4):
tokens = clean_text -> (?u)\\b\\w\\w+\\b ; content tokens = tokens minus ENGLISH_STOP_WORDS;
Dice = 2|Q ∩ M| / (|Q| + |M|) over content-token sets.

Values are exact Fractions so that boundary comparisons (0.25, 0.50, 0.60, 0.75) are
exact. `max_dice_to_acceptable` defines P* and positive-query buckets; it is never a
retrieval acceptance gate (R2-1).
"""
from fractions import Fraction
from typing import Iterable, Optional

from src.ml.lexical import content_tokens, tokenize
from src.ml.preprocessing import clean_text

import config


def content_set(text: str) -> frozenset:
    return frozenset(content_tokens(tokenize(clean_text(text))))


def dice(a: frozenset, b: frozenset) -> Fraction:
    denom = len(a) + len(b)
    if denom == 0:
        return Fraction(0)
    return Fraction(2 * len(a & b), denom)


def max_dice(query: str, questions: Iterable[str]) -> Optional[Fraction]:
    """Maximum Dice between `query` and any of `questions`; None if there are none."""
    q = content_set(query)
    values = [dice(q, content_set(m)) for m in questions]
    return max(values) if values else None


def is_positive(category: str) -> bool:
    return category in config.POSITIVE_CATEGORIES


def dice_bucket(category: str, max_dice_to_acceptable: Optional[Fraction]) -> str:
    """B1–B4 for positive queries (from max_dice_to_acceptable); NEG for Category 4."""
    if category == config.NEGATIVE_CATEGORY:
        return "NEG"
    if max_dice_to_acceptable is None:
        raise ValueError("Positive queries need max_dice_to_acceptable to be bucketed")
    e1, e2, e3 = config.BUCKET_EDGES
    d = max_dice_to_acceptable
    if d < e1:
        return "B1"
    if d < e2:
        return "B2"
    if d < e3:
        return "B3"
    return "B4"


def reporting_slice(category: str, max_dice_to_acceptable: Optional[Fraction]) -> str:
    """Reporting slice: B1, B2, B3a, B3b, B4 or NEG (spec §3.4, §7.5)."""
    bucket = dice_bucket(category, max_dice_to_acceptable)
    if bucket != "B3":
        return bucket
    return "B3a" if max_dice_to_acceptable < config.B3_SPLIT else "B3b"


def in_primary_population(category: str, max_dice_to_acceptable: Optional[Fraction]) -> bool:
    """P*: positive queries for which no acceptable KB answer has Dice >= 0.60."""
    if not is_positive(category) or max_dice_to_acceptable is None:
        return False
    return max_dice_to_acceptable < config.PRIMARY_DICE_LT


def format_dice(value: Optional[Fraction]) -> str:
    return "" if value is None else f"{float(value):.{config.DICE_DECIMALS}f}"
