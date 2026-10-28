"""
Challenge benchmark I/O, annotation, validation and freeze helpers (spec §3, §2.2).

Validation is split into:
  * schema checks (spec §3.3), which work on draft and verified files;
  * contamination checks (spec §3.2 step 4);
  * composition checks (size, category counts, |P*|), which apply to the final
    verified benchmark.
Each check returns a list of human-readable error strings (empty = pass).
"""
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.ml.preprocessing import clean_text

import config
from lib import dice as b1_dice
from lib.data import CanonicalSplit

CHALLENGE_ID_RE = re.compile(r"^CH_\d{3}$")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_benchmark(path: Path = config.BENCHMARK_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return df


def write_benchmark(df: pd.DataFrame, path: Path = config.BENCHMARK_PATH) -> None:
    df = df.loc[:, list(config.BENCHMARK_COLUMNS)]
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def parse_row_ids(value: str) -> Tuple[int, ...]:
    value = (value or "").strip()
    if not value:
        return ()
    return tuple(int(v.strip()) for v in value.split(";"))


def parse_optional_int(value: str) -> Optional[int]:
    value = (value or "").strip()
    return int(value) if value else None


# ---------------------------------------------------------------------------
# Annotation (Dice fields and bucket)
# ---------------------------------------------------------------------------
def exact_dice_fields(row: pd.Series, split: CanonicalSplit) -> Dict[str, Optional[Fraction]]:
    """Exact Dice diagnostics for one row (ids must already be valid training ids)."""
    question = {rid: q for rid, q in zip(split.train_ids, split.train_df["User Message"])}
    query = row["query_text"]
    acceptable = parse_row_ids(row["acceptable_row_ids"])
    preferred = parse_optional_int(row["preferred_row_id"])
    q_set = b1_dice.content_set(query)
    return {
        "dice_to_preferred": (
            b1_dice.dice(q_set, b1_dice.content_set(question[preferred])) if preferred is not None else None
        ),
        "max_dice_to_acceptable": b1_dice.max_dice(query, [question[r] for r in acceptable]),
        "max_dice_to_any_kb_row": b1_dice.max_dice(query, question.values()),
    }


def annotate(df: pd.DataFrame, split: CanonicalSplit) -> pd.DataFrame:
    """Fill dice_to_preferred, max_dice_to_acceptable, max_dice_to_any_kb_row, dice_bucket."""
    out = df.copy()
    for i, row in out.iterrows():
        fields = exact_dice_fields(row, split)
        out.at[i, "dice_to_preferred"] = b1_dice.format_dice(fields["dice_to_preferred"])
        out.at[i, "max_dice_to_acceptable"] = b1_dice.format_dice(fields["max_dice_to_acceptable"])
        out.at[i, "max_dice_to_any_kb_row"] = b1_dice.format_dice(fields["max_dice_to_any_kb_row"])
        category = row["difficulty_category"]
        if category == config.NEGATIVE_CATEGORY or fields["max_dice_to_acceptable"] is not None:
            out.at[i, "dice_bucket"] = b1_dice.dice_bucket(category, fields["max_dice_to_acceptable"])
        else:
            out.at[i, "dice_bucket"] = ""
    return out


# ---------------------------------------------------------------------------
# Schema validation (spec §3.3)
# ---------------------------------------------------------------------------
def schema_errors(df: pd.DataFrame, split: CanonicalSplit) -> List[str]:
    errors: List[str] = []
    if tuple(df.columns) != config.BENCHMARK_COLUMNS:
        return [f"Columns must be exactly {list(config.BENCHMARK_COLUMNS)}; got {list(df.columns)}"]
    train_ids = set(split.train_ids)
    all_intents = set(split.all_intents())
    eligible = set(split.eligible_intents())
    seen_ids = set()
    for _, row in df.iterrows():
        cid = row["challenge_id"]
        where = cid or "<missing id>"
        if not CHALLENGE_ID_RE.match(cid):
            errors.append(f"{where}: challenge_id must match CH_###")
        if cid in seen_ids:
            errors.append(f"{where}: duplicate challenge_id")
        seen_ids.add(cid)
        if not row["query_text"].strip():
            errors.append(f"{where}: empty query_text")
        category = row["difficulty_category"]
        if category not in config.CATEGORY_COUNTS:
            errors.append(f"{where}: unknown difficulty_category '{category}'")
            continue
        status = row["review_status"]
        if status not in config.REVIEW_STATUSES:
            errors.append(f"{where}: review_status must be one of {config.REVIEW_STATUSES}")
        try:
            acceptable = parse_row_ids(row["acceptable_row_ids"])
            preferred = parse_optional_int(row["preferred_row_id"])
        except ValueError:
            errors.append(f"{where}: row ids must be integers (acceptable 'a;b', preferred 'n')")
            continue
        if list(acceptable) != sorted(set(acceptable)):
            errors.append(f"{where}: acceptable_row_ids must be unique and sorted ascending")
        for rid in acceptable + ((preferred,) if preferred is not None else ()):
            if rid not in train_ids:
                errors.append(f"{where}: row id {rid} is not a training row id")
        if preferred is not None and preferred not in acceptable:
            errors.append(f"{where}: preferred_row_id must be one of acceptable_row_ids")
        if category == config.NEGATIVE_CATEGORY and preferred is not None:
            errors.append(f"{where}: Category 4 rows must have an empty preferred_row_id")
        intent = row["target_intent"].strip()
        if status == "verified":
            if intent not in all_intents:
                errors.append(f"{where}: target_intent must be one of the 26 Stage A intents")
            elif category in config.POSITIVE_CATEGORIES and intent not in eligible:
                errors.append(f"{where}: positive queries must target an eligible intent (>= 3 training rows)")
            if category in config.POSITIVE_CATEGORIES and not acceptable:
                errors.append(f"{where}: Categories 1–3 need at least one acceptable_row_id")
        elif intent and intent not in all_intents:
            errors.append(f"{where}: target_intent must be one of the 26 Stage A intents")
    return errors


def annotation_errors(df: pd.DataFrame, split: CanonicalSplit) -> List[str]:
    """Stored Dice/bucket fields must equal the recomputed values."""
    expected = annotate(df, split)
    errors = []
    for col in ("dice_to_preferred", "max_dice_to_acceptable", "max_dice_to_any_kb_row", "dice_bucket"):
        diff = df[col] != expected[col]
        for cid in df.loc[diff, "challenge_id"]:
            errors.append(f"{cid}: {col} does not match recomputed value (run annotate)")
    return errors


# ---------------------------------------------------------------------------
# Contamination checks (spec §3.2 step 4, §2.2.6)
# ---------------------------------------------------------------------------
def contamination_errors(df: pd.DataFrame, split: CanonicalSplit) -> List[str]:
    errors: List[str] = []
    train_q = {clean_text(q) for q in split.train_df["User Message"]}
    test_q = {clean_text(q) for q in split.test_df["User Message"]}
    test_ids = set(split.test_ids)
    seen = {}
    for _, row in df.iterrows():
        cid = row["challenge_id"]
        cq = clean_text(row["query_text"])
        if cq in train_q:
            errors.append(f"{cid}: query equals a training (KB) question after clean_text")
        if cq in test_q:
            errors.append(f"{cid}: query equals a Stage A test question after clean_text")
        if cq in seen:
            errors.append(f"{cid}: duplicate of {seen[cq]} after clean_text")
        seen[cq] = cid
        try:
            ids = parse_row_ids(row["acceptable_row_ids"]) + tuple(
                x for x in (parse_optional_int(row["preferred_row_id"]),) if x is not None
            )
        except ValueError:
            continue
        for rid in ids:
            if rid in test_ids:
                errors.append(f"{cid}: row id {rid} is a Stage A TEST row id")
    return errors


# ---------------------------------------------------------------------------
# Composition (final verified benchmark)
# ---------------------------------------------------------------------------
def population_counts(df: pd.DataFrame, split: CanonicalSplit) -> Dict[str, object]:
    """Counts of Q, P, P*, N, reporting slices, categories and intents (needs labels)."""
    slices: Dict[str, int] = {}
    n_primary = 0
    for _, row in df.iterrows():
        category = row["difficulty_category"]
        fields = exact_dice_fields(row, split)
        mda = fields["max_dice_to_acceptable"]
        if category in config.POSITIVE_CATEGORIES and mda is None:
            slices["unlabelled"] = slices.get("unlabelled", 0) + 1
            continue
        s = b1_dice.reporting_slice(category, mda)
        slices[s] = slices.get(s, 0) + 1
        n_primary += int(b1_dice.in_primary_population(category, mda))
    categories = df["difficulty_category"].value_counts().to_dict()
    intents = df.loc[df["target_intent"] != "", "target_intent"].value_counts().to_dict()
    positives = int(df["difficulty_category"].isin(config.POSITIVE_CATEGORIES).sum())
    return {
        "Q": int(len(df)),
        "P": positives,
        "P_star": n_primary,
        "N": int((df["difficulty_category"] == config.NEGATIVE_CATEGORY).sum()),
        "slices": dict(sorted(slices.items())),
        "categories": {k: int(v) for k, v in sorted(categories.items())},
        "intents": {k: int(v) for k, v in sorted(intents.items())},
        "n_positive_intents_covered": int(
            df.loc[df["difficulty_category"].isin(config.POSITIVE_CATEGORIES) & (df["target_intent"] != ""),
                   "target_intent"].nunique()
        ),
    }


def composition_errors(df: pd.DataFrame, split: CanonicalSplit) -> List[str]:
    errors: List[str] = []
    if len(df) != config.BENCHMARK_SIZE:
        errors.append(f"Benchmark must have exactly {config.BENCHMARK_SIZE} rows; has {len(df)}")
    counts = df["difficulty_category"].value_counts().to_dict()
    for category, expected in config.CATEGORY_COUNTS.items():
        if counts.get(category, 0) != expected:
            errors.append(f"Category {category}: expected {expected}, found {counts.get(category, 0)}")
    not_verified = df.loc[df["review_status"] != "verified", "challenge_id"].tolist()
    if not_verified:
        errors.append(f"{len(not_verified)} rows are not verified: {not_verified[:10]}")
    counts_pop = population_counts(df, split)
    if counts_pop["P_star"] < config.PRIMARY_MIN:
        errors.append(f"|P*| = {counts_pop['P_star']} < required minimum {config.PRIMARY_MIN}")
    return errors


def composition_warnings(df: pd.DataFrame, split: CanonicalSplit) -> List[str]:
    lo, hi = config.INTENT_COVERAGE_RANGE
    n = population_counts(df, split)["n_positive_intents_covered"]
    if not lo <= n <= hi:
        return [f"Positive queries cover {n} intents; spec target is {lo}–{hi} where feasible"]
    return []


# ---------------------------------------------------------------------------
# Freeze helpers (Phase 2)
# ---------------------------------------------------------------------------
def set_config_benchmark_sha(sha: str, config_path: Path = config.EXPERIMENT_DIR / "config.py") -> None:
    text = config_path.read_text(encoding="utf-8")
    new, n = re.subn(r"^BENCHMARK_SHA256 = .*$", f'BENCHMARK_SHA256 = "{sha}"', text, flags=re.M)
    if n != 1:
        raise RuntimeError("Could not locate BENCHMARK_SHA256 in config.py")
    config_path.write_text(new, encoding="utf-8")


def decisions_mentions_sha(sha: str, decisions_path: Path = config.DECISIONS_PATH) -> bool:
    return decisions_path.exists() and sha in decisions_path.read_text(encoding="utf-8")
