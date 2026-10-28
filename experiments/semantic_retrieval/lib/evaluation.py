"""
B1 Phase 3 evaluation logic: benchmark items, retrieval passes, per-population metrics,
paired deltas, outcome category and failure-mode tagging.

Sources: parameters and gates come from config.py and the freeze record in DECISIONS.md;
metric definitions come from lib/metrics.py. Rules the repository did not already define
(outcome-category assignment, failure-signature precedence) are documented as implementation
decisions in DECISIONS.md — they are not claimed to come from the unavailable specification.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.ml.lexical import tokenize
from src.ml.preprocessing import clean_text

import config
from lib import benchmark as bm
from lib import dice as b1_dice
from lib import metrics as m
from lib import stats as b1_stats

PASSES = ("pass1_global", "pass2_oracle", "pass3_predicted")
PRIMARY_PASS = "pass1_global"
DELTA_METRICS = tuple(n for n in m.METRIC_NAMES if n != "CosineMargin")


@dataclass(frozen=True)
class BenchmarkItem:
    challenge_id: str
    query: str
    category: str
    labels: m.QueryLabels
    positive: bool
    primary: bool
    slice: str


def load_items(df, split) -> List[BenchmarkItem]:
    """Benchmark rows -> items with labels, P* membership and reporting slice (exact Dice)."""
    items = []
    for _, row in df.iterrows():
        fields = bm.exact_dice_fields(row, split)
        mda = fields["max_dice_to_acceptable"]
        category = row["difficulty_category"]
        labels = m.QueryLabels(
            row["challenge_id"],
            row["target_intent"] or None,
            frozenset(bm.parse_row_ids(row["acceptable_row_ids"])),
            bm.parse_optional_int(row["preferred_row_id"]),
        )
        items.append(BenchmarkItem(
            row["challenge_id"], row["query_text"], category, labels,
            positive=b1_dice.is_positive(category),
            primary=b1_dice.in_primary_population(category, mda),
            slice=b1_dice.reporting_slice(category, mda),
        ))
    return items


def make_row(it: BenchmarkItem, pass_name: str, retriever_name: str, ranking, intent_of, predicted) -> Dict:
    values = m.per_query_values(ranking, it.labels, intent_of)
    acc_rank = m.first_acceptable_rank(ranking, it.labels.acceptable) if it.labels.acceptable else None
    pref_rank = m.rank_of(ranking, it.labels.preferred) if it.labels.preferred is not None else None
    return {
        "challenge_id": it.challenge_id,
        "query_text": it.query,
        "pass": pass_name,
        "retriever": retriever_name,
        "category": it.category,
        "slice": it.slice,
        "primary": it.primary,
        "positive": it.positive,
        "target_intent": it.labels.target_intent,
        "predicted_intent": predicted,
        "acceptable_row_ids": ";".join(map(str, sorted(it.labels.acceptable))),
        "preferred_row_id": "" if it.labels.preferred is None else it.labels.preferred,
        "topk_row_ids": ";".join(str(r) for r, _ in ranking),
        "topk_scores": ";".join(f"{s:.6f}" for _, s in ranking),
        "topk_intents": ";".join(intent_of[r] for r, _ in ranking),
        "rank1_row_id": ranking[0][0] if ranking else "",
        "rank1_score": ranking[0][1] if ranking else None,
        "first_acceptable_rank": acc_rank if acc_rank is not None else "",
        "preferred_rank": pref_rank if pref_rank is not None else "",
        "outcome_class": m.outcome_class(ranking, it.labels, intent_of),
        "_values": values,
    }


def run_passes(retrievers: Sequence, items: Sequence[BenchmarkItem], predicted_intent: Mapping[str, Optional[str]],
               intent_of: Mapping[int, str], k: int = config.RETRIEVAL_DEPTH) -> List[Dict]:
    """Pass 1 global, Pass 2 oracle target intent, Pass 3 Stage A predicted intent; all retrievers, all items."""
    rows = []
    for it in items:
        pred = predicted_intent.get(it.challenge_id)
        restrict = {"pass1_global": None, "pass2_oracle": it.labels.target_intent, "pass3_predicted": pred}
        for pass_name in PASSES:
            for ret in retrievers:
                ranking = ret.rank(it.query, k, intent=restrict[pass_name])
                rows.append(make_row(it, pass_name, ret.name, ranking, intent_of, pred))
    return rows


def _values(rows, pass_name, retriever, ids) -> Dict[str, Dict[str, float]]:
    return {r["challenge_id"]: r["_values"] for r in rows
            if r["pass"] == pass_name and r["retriever"] == retriever and r["challenge_id"] in ids}


def population_summary(rows, items, pass_name, baseline: str, candidate: str,
                       member: Callable[[BenchmarkItem], bool], bootstrap: bool = True) -> Dict:
    """Metrics for both retrievers plus paired deltas (candidate − baseline) over one population."""
    ids = [it.challenge_id for it in items if member(it)]
    out = {"n_queries": len(ids), "retrievers": {}, "deltas": {}, "outcome_counts": {}}
    if not ids:
        return out
    vals = {name: _values(rows, pass_name, name, set(ids)) for name in (baseline, candidate)}
    for name in (baseline, candidate):
        out["retrievers"][name] = m.aggregate([vals[name][c] for c in ids])
        counts = {c: 0 for c in m.OUTCOME_CLASSES}
        for r in rows:
            if r["pass"] == pass_name and r["retriever"] == name and r["challenge_id"] in ids:
                counts[r["outcome_class"]] += 1
        out["outcome_counts"][name] = counts
    for metric in DELTA_METRICS:
        a = np.array([vals[baseline][c][metric] for c in ids], dtype=float)
        b = np.array([vals[candidate][c][metric] for c in ids], dtype=float)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if not mask.any():
            continue
        if bootstrap:
            out["deltas"][metric] = b1_stats.paired_bootstrap(a[mask], b[mask])
        else:
            out["deltas"][metric] = {"n": int(mask.sum()), "delta": float(b[mask].mean() - a[mask].mean())}
    return out


def primary_wilcoxon(rows, items, baseline, candidate) -> Dict:
    ids = [it.challenge_id for it in items if it.primary]
    vb = _values(rows, PRIMARY_PASS, baseline, set(ids))
    vc = _values(rows, PRIMARY_PASS, candidate, set(ids))
    key = f"AcceptableMRR@{config.MRR_DEPTH}"
    return b1_stats.paired_wilcoxon([vb[c][key] for c in ids], [vc[c][key] for c in ids])


def assign_outcome_category(ar1: Dict, mrr: Dict, wrong_intent: Dict, wilcoxon: Dict,
                            resources_ok: Optional[bool]) -> Dict:
    """Ordered outcome categories (implementation decision; see DECISIONS.md)."""
    t1, t2, s = config.T1_PP / 100, config.T2_MRR, config.S_PP / 100
    safety = wrong_intent["delta"] <= s + 1e-12
    p = wilcoxon.get("p_value")
    significant = isinstance(p, float) and p < config.WILCOXON_ALPHA
    thresholds = ar1["delta"] >= t1 - 1e-12 and mrr["delta"] >= t2 - 1e-12
    ci_positive = ar1["ci_low"] > 0 and mrr["ci_low"] > 0
    any_ci_positive = ar1["ci_low"] > 0 or mrr["ci_low"] > 0
    any_ci_negative = ar1["ci_high"] < 0 or mrr["ci_high"] < 0
    if any_ci_negative or not safety:
        category = "Regression"
    elif thresholds and ci_positive and significant:
        if resources_ok is None:
            category = "Meaningful retrieval improvement — resources not measured"
        elif resources_ok:
            category = "Meaningful improvement"
        else:
            category = "Meaningful retrieval improvement — resource limit exceeded"
    elif any_ci_positive:
        category = "Modest improvement"
    else:
        category = "No meaningful difference / inconclusive"
    underpowered = category == "No meaningful difference / inconclusive" and (
        ar1["delta"] >= t1 - 1e-12 or mrr["delta"] >= t2 - 1e-12)
    return {
        "category": category,
        "underpowered": underpowered,
        "thresholds_met": thresholds,
        "ci_lower_bounds_positive": ci_positive,
        "wilcoxon_significant": significant,
        "safety_guardrail_holds": safety,
        "resource_gates_hold": resources_ok,
    }


def score_distribution(scores: Sequence[Optional[float]]) -> Dict:
    vals = np.array([s for s in scores if s is not None], dtype=float)
    if not len(vals):
        return {"n": 0}
    return {"n": int(len(vals)), "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)), "max": float(vals.max())}


def is_failure(outcome_class: str, negative: bool) -> bool:
    if negative:
        return outcome_class == m.WRONG_INTENT
    return outcome_class in (m.INTENT_ONLY, m.WRONG_INTENT)


def failure_signatures(query: str, rank1_score: Optional[float], outcome_class: str, negative: bool) -> List[str]:
    """Automatic F-mode signatures; F2 (entity collapse) cannot be detected automatically."""
    sig = []
    if negative:
        sig.append("F5")
    if len(tokenize(clean_text(query))) <= config.F3_MAX_TOKENS:
        sig.append("F3")
    if rank1_score is not None and rank1_score > config.F1_COSINE_GT and outcome_class == m.WRONG_INTENT:
        sig.append("F1")
    if rank1_score is not None and rank1_score < config.F4_COSINE_LT:
        sig.append("F4")
    return sig


FAILURE_PRECEDENCE = ("F5", "F3", "F1", "F4")


def primary_failure_tag(signatures: Sequence[str]) -> str:
    for tag in FAILURE_PRECEDENCE:
        if tag in signatures:
            return tag
    return "UNASSIGNED"


# ---------------------------------------------------------------------------
# Human-review preservation across reruns
# ---------------------------------------------------------------------------
GUARDRAIL_EVIDENCE = {1: "guardrail_1_cases_stage_a_hit_b1_miss", 2: "guardrail_2_rank1_score_distributions"}
HUMAN_OUTCOME_KEYS = ("guardrails_human_review", "overall_acceptance", "overall_acceptance_reason",
                      "reporting_decision", "provisional", "provisional_reason")
PENDING = "pending_human_confirmation"


def preserve_human_review(new: Dict, old: Optional[Dict]):
    """
    Carry human guardrail decisions from a previous result into a freshly computed one.

    A guardrail's status/note is carried only if its evidence is unchanged; outcome-level human fields
    only if both guardrails were carried and the computed category is unchanged. Anything not carried
    is kept under `previous_human_review` (never silently dropped). Returns (new, messages).
    """
    msgs: List[str] = []
    if not old:
        return new, msgs
    g_new, g_old = new["guardrails"], old.get("guardrails", {})
    if g_old.get("previous_human_review"):  # history from earlier reruns is append-only, never dropped
        g_new["previous_human_review"] = {k: list(v) for k, v in g_old["previous_human_review"].items()}
    carried = []
    for i, key in GUARDRAIL_EVIDENCE.items():
        status = g_old.get(f"guardrail_{i}_status")
        if not status or status == PENDING:
            continue
        prev = {k: g_old[k] for k in (f"guardrail_{i}_status", f"guardrail_{i}_human_note") if k in g_old}
        if g_old.get(key) == g_new.get(key):
            g_new.update(prev)
            carried.append(i)
        else:
            g_new.setdefault("previous_human_review", {}).setdefault(f"guardrail_{i}", []).append(
                dict(prev, reason="evidence changed on rerun; re-review required"))
            msgs.append(f"guardrail {i}: evidence changed, previous human review kept under previous_human_review")
    o_new = new["primary"]["outcome"]
    o_old = old.get("primary", {}).get("outcome", {})
    if o_old.get("previous_human_review"):
        o_new["previous_human_review"] = list(o_old["previous_human_review"])
    human = {k: o_old[k] for k in HUMAN_OUTCOME_KEYS if k in o_old}
    has_decision = any(k in o_old for k in ("guardrails_human_review", "overall_acceptance"))
    if has_decision:
        if carried == [1, 2] and o_old.get("category") == o_new.get("category"):
            o_new.update(human)
        else:
            o_new.setdefault("previous_human_review", []).append(
                dict(human, category=o_old.get("category"),
                     reason="guardrail evidence or category changed; re-review required"))
            msgs.append("outcome: human acceptance decision not carried; kept under previous_human_review")
    return new, msgs


def merge_human_tags(new_rows: List[Dict], old_rows: Sequence[Dict]):
    """Carry human_tag for failures whose rank-1 row is unchanged; return (rows, dropped human-tagged rows)."""
    old = {r["challenge_id"]: r for r in old_rows if r.get("human_tag")}
    for r in new_rows:
        prev = old.get(r["challenge_id"])
        if prev and str(prev.get("rank1_row_id")) == str(r.get("rank1_row_id")):
            r["human_tag"] = prev["human_tag"]
            del old[r["challenge_id"]]
    return new_rows, list(old.values())
