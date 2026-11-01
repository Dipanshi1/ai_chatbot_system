"""
Lexicon-based sentiment heuristic.

A rough signal used only to decide whether to offer a human contact. It is not a
validated model: no accuracy has been measured, because neither available dataset
has a usable emotion label. It never reaches intent classification or answer
selection. All lexicon entries are hand-written prototype values; not fitted.
"""
from dataclasses import dataclass
import re
from typing import List, Tuple

_NEGATIVE = frozenset(
    {
        "angry", "annoyed", "annoying", "anxious", "awful", "bad", "broken", "confused",
        "confusing", "desperate", "disappointed", "disappointing", "failed", "failing",
        "frustrated", "frustrating", "hate", "horrible", "ignored", "impossible", "incorrect",
        "overwhelmed", "pointless", "ridiculous", "rude", "scared", "stressed", "stressful",
        "struggling", "stuck", "terrible", "unacceptable", "unfair", "unhappy", "unhelpful",
        "upset", "useless", "waste", "worried", "worst", "wrong",
    }
)
_POSITIVE = frozenset(
    {
        "amazing", "appreciate", "awesome", "excellent", "fantastic", "glad", "good", "great",
        "happy", "helped", "helpful", "love", "perfect", "pleased", "thank", "thanks",
        "wonderful",
    }
)
_NEGATORS = frozenset(
    {
        "can't", "cannot", "didn't", "doesn't", "don't", "hardly", "isn't", "never", "no",
        "not", "wasn't", "won't",
    }
)
_NEGATION_WINDOW = 2
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "negative" | "neutral" | "positive"
    score: float  # (positive - negative) / matched terms, in [-1, 1]
    matched_terms: Tuple[str, ...]


def analyze_sentiment(text: object) -> SentimentResult:
    """Score a message; a tie or no matched term is neutral."""
    if not isinstance(text, str):
        return SentimentResult("neutral", 0.0, ())

    tokens = _TOKEN.findall(text.replace("’", "'").lower())
    polarities: List[int] = []
    matched: List[str] = []
    for index, token in enumerate(tokens):
        if token not in _NEGATIVE and token not in _POSITIVE:
            continue
        polarity = 1 if token in _POSITIVE else -1
        if any(t in _NEGATORS for t in tokens[max(0, index - _NEGATION_WINDOW):index]):
            polarity = -polarity
        polarities.append(polarity)
        matched.append(token)

    if not polarities:
        return SentimentResult("neutral", 0.0, ())

    score = sum(polarities) / len(polarities)
    label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
    return SentimentResult(label, score, tuple(matched))
