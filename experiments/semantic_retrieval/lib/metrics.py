"""
Per-query retrieval outcomes and aggregate metrics (spec §7).

Denominators:
  * Acceptable* metrics: queries with a non-empty acceptable set.
  * Preferred* metrics: queries with a preferred row.
  * Intent metrics (IntentRecall@K, WrongIntent@1): queries with a target intent.
  * Mean cosine margin: queries whose ranking has >= 2 candidates.
Every aggregate carries its n.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import config

Ranking = List[Tuple[int, float]]

NO_CANDIDATE = "NO_CANDIDATE"
PREFERRED_HIT = "PREFERRED_HIT"
ACCEPTABLE_HIT = "ACCEPTABLE_HIT"
INTENT_ONLY = "INTENT_ONLY"
WRONG_INTENT = "WRONG_INTENT"
OUTCOME_CLASSES = (NO_CANDIDATE, PREFERRED_HIT, ACCEPTABLE_HIT, INTENT_ONLY, WRONG_INTENT)


@dataclass(frozen=True)
class QueryLabels:
    challenge_id: str
    target_intent: Optional[str]
    acceptable: FrozenSet[int] = field(default_factory=frozenset)
    preferred: Optional[int] = None


def rank_of(ranking: Ranking, row_id: int) -> Optional[int]:
    """1-indexed rank of row_id in the ranking, or None if absent."""
    for i, (rid, _) in enumerate(ranking, start=1):
        if rid == row_id:
            return i
    return None


def first_acceptable_rank(ranking: Ranking, acceptable: FrozenSet[int]) -> Optional[int]:
    for i, (rid, _) in enumerate(ranking, start=1):
        if rid in acceptable:
            return i
    return None


def outcome_class(ranking: Ranking, labels: QueryLabels, intent_of: Mapping[int, str]) -> str:
    """Rank-1 outcome class; first matching class wins (spec §7.3)."""
    if not ranking:
        return NO_CANDIDATE
    top = ranking[0][0]
    if labels.preferred is not None and top == labels.preferred:
        return PREFERRED_HIT
    if top in labels.acceptable:
        return ACCEPTABLE_HIT
    if intent_of[top] == labels.target_intent:
        return INTENT_ONLY
    return WRONG_INTENT


def reciprocal_rank(rank: Optional[int], depth: int = config.MRR_DEPTH) -> float:
    return 1.0 / rank if rank is not None and rank <= depth else 0.0


def per_query_values(ranking: Ranking, labels: QueryLabels, intent_of: Mapping[int, str]) -> Dict[str, float]:
    """Per-query indicator values; NaN marks 'not in this metric's denominator'."""
    nan = float("nan")
    values: Dict[str, float] = {}
    has_acc = bool(labels.acceptable)
    has_pref = labels.preferred is not None
    has_intent = labels.target_intent is not None
    acc_rank = first_acceptable_rank(ranking, labels.acceptable) if has_acc else None
    pref_rank = rank_of(ranking, labels.preferred) if has_pref else None
    for k in config.TOP_K_DEPTHS:
        values[f"AcceptableRecall@{k}"] = float(acc_rank is not None and acc_rank <= k) if has_acc else nan
        values[f"PreferredRecall@{k}"] = float(pref_rank is not None and pref_rank <= k) if has_pref else nan
        if has_intent:
            values[f"IntentRecall@{k}"] = float(any(intent_of[r] == labels.target_intent for r, _ in ranking[:k]))
        else:
            values[f"IntentRecall@{k}"] = nan
    values[f"AcceptableMRR@{config.MRR_DEPTH}"] = reciprocal_rank(acc_rank) if has_acc else nan
    values[f"PreferredMRR@{config.MRR_DEPTH}"] = reciprocal_rank(pref_rank) if has_pref else nan
    if has_intent:
        values["WrongIntent@1"] = float(outcome_class(ranking, labels, intent_of) == WRONG_INTENT)
    else:
        values["WrongIntent@1"] = nan
    values["CosineMargin"] = ranking[0][1] - ranking[1][1] if len(ranking) >= 2 else nan
    return values


METRIC_NAMES = tuple(
    [f"AcceptableRecall@{k}" for k in config.TOP_K_DEPTHS]
    + [f"AcceptableMRR@{config.MRR_DEPTH}"]
    + [f"PreferredRecall@{k}" for k in config.TOP_K_DEPTHS]
    + [f"PreferredMRR@{config.MRR_DEPTH}"]
    + [f"IntentRecall@{k}" for k in config.TOP_K_DEPTHS]
    + ["WrongIntent@1", "CosineMargin"]
)


def aggregate(per_query: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean of each metric over its own denominator, with n."""
    out: Dict[str, Dict[str, float]] = {}
    for name in METRIC_NAMES:
        vals = np.array([pq[name] for pq in per_query], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        out[name] = {"value": float(vals.mean()) if len(vals) else None, "n": int(len(vals))}
    return out


def outcome_counts(ranking_labels: Sequence[Tuple[Ranking, QueryLabels]], intent_of: Mapping[int, str]) -> Dict[str, int]:
    counts = {c: 0 for c in OUTCOME_CLASSES}
    for ranking, labels in ranking_labels:
        counts[outcome_class(ranking, labels, intent_of)] += 1
    return counts
