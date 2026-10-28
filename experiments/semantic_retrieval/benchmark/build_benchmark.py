"""
B1 challenge benchmark tool (spec §3, §16 Step 3).

Modes:
  review-pack    Write a reviewer aid to /tmp/b1_review_pack.md (outside the repo).
                 Only training rows appear in it; test rows never do.
  annotate       Fill the Dice fields and dice_bucket (no row is rejected for its Dice value).
  validate       Schema + annotation consistency + contamination + composition checks.
  freeze         Validate, record the CSV sha256 in config.py and append the
                 pre-registration entry to DECISIONS.md.
  verify-freeze  Check the CSV sha256 against config.py and DECISIONS.md.

Usage (from the repo root):
  HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python \
      experiments/semantic_retrieval/benchmark/build_benchmark.py <mode>
"""
import argparse
import sys
from datetime import date
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from lib import benchmark as bm  # noqa: E402
from lib import data as b1_data  # noqa: E402
from lib import provenance  # noqa: E402


def cmd_review_pack(split) -> int:
    df = bm.read_benchmark()
    errors = bm.schema_errors(df, split) + bm.contamination_errors(df, split)
    if errors:
        print("Benchmark has schema/contamination errors; fix before review:")
        print("\n".join(f"  - {e}" for e in errors))
        return 1
    train = split.train_df
    eligible = split.eligible_intents()
    lines = [
        "# B1 Challenge Benchmark — Reviewer Pack",
        "",
        "Generated from experiments/semantic_retrieval/benchmark/challenge_set_v1.csv.",
        "Only TRAINING rows (the searchable KB) are shown. Stage A test rows are excluded.",
        "",
        "## What to fill in challenge_set_v1.csv (human reviewer only)",
        "- Keep exactly 72 rows: 22 synonym_substitution, 18 syntactic_inversion,",
        "  18 slang_informal, 14 distractor_negative. Delete rejected rows; rewording is allowed.",
        "- target_intent: one of the 26 Stage A intents (positive queries: one of the 17 eligible intents below).",
        "- acceptable_row_ids: training rows whose Bot Response correctly answers the query,",
        "  semicolon-separated, ascending (e.g. 115;131). Categories 1–3: at least one.",
        "  Category 4: may be empty. Check at least every training row of the target intent.",
        "- preferred_row_id: optional for Categories 1–3 (must be one of acceptable_row_ids);",
        "  ALWAYS empty for Category 4.",
        "- review_status: set to verified.",
        "- Leave dice_to_preferred, max_dice_to_acceptable, max_dice_to_any_kb_row and dice_bucket",
        "  EMPTY — Phase 2 computes them.",
        "- Aim for >= 40 positive queries where no acceptable row shares much wording with the query",
        "  (Phase 2 computes the exact count and reports any shortfall).",
        "",
        "## Draft queries",
        "",
    ]
    by_id = {rid: (it, q) for rid, it, q in zip(train.index, train["Intent"], train["User Message"])}
    for _, row in df.iterrows():
        seed = row["notes"].replace("drafted from row", "").strip()
        seed_txt = ""
        if seed.isdigit() and int(seed) in by_id:
            it, q = by_id[int(seed)]
            seed_txt = f"row {seed} [{it}] \"{q}\""
        lines += [
            f"### {row['challenge_id']} — {row['difficulty_category']}",
            f"- Query: **{row['query_text']}**",
            f"- Drafted from: {seed_txt or row['notes']}",
            f"- Current labels: target_intent=`{row['target_intent']}` "
            f"acceptable=`{row['acceptable_row_ids']}` preferred=`{row['preferred_row_id']}` "
            f"status=`{row['review_status']}`",
            "",
        ]
    lines += ["## Training rows of the 17 eligible intents (row_id | question | Bot Response)", ""]
    for intent in eligible:
        g = train[train["Intent"] == intent]
        lines.append(f"### {intent} ({len(g)} rows)")
        for rid, q, resp in zip(g.index, g["User Message"], g["Bot Response"]):
            lines.append(f"- **{rid}** | {q} | {resp}")
        lines.append("")
    other = sorted(set(split.all_intents()) - set(eligible))
    lines += [
        "## Other intents (<= 2 training rows; usable only as Category 4 target_intent)",
        "",
    ]
    for intent in other:
        g = train[train["Intent"] == intent]
        for rid, q, resp in zip(g.index, g["User Message"], g["Bot Response"]):
            lines.append(f"- **{rid}** [{intent}] | {q} | {resp}")
    config.REVIEW_PACK_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Review pack written to {config.REVIEW_PACK_PATH} ({len(df)} draft queries)")
    return 0


def cmd_annotate(split) -> int:
    df = bm.read_benchmark()
    errors = bm.schema_errors(df, split) + bm.contamination_errors(df, split)
    if errors:
        print("Cannot annotate; fix these first:")
        print("\n".join(f"  - {e}" for e in errors))
        return 1
    bm.write_benchmark(bm.annotate(df, split))
    print(f"Annotated {len(df)} rows -> {config.BENCHMARK_PATH}")
    return 0


def _validate(split):
    df = bm.read_benchmark()
    errors = bm.schema_errors(df, split)
    if not errors:
        errors += bm.annotation_errors(df, split)
        errors += bm.contamination_errors(df, split)
        errors += bm.composition_errors(df, split)
    return df, errors


def cmd_validate(split) -> int:
    df, errors = _validate(split)
    if not bm.schema_errors(df, split):
        counts = bm.population_counts(df, split)
        print("Population counts:", counts)
        for w in bm.composition_warnings(df, split):
            print("WARNING:", w)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} problems):")
        print("\n".join(f"  - {e}" for e in errors))
        return 1
    print("VALIDATION PASSED")
    return 0


def _decisions_entry(sha: str, counts: dict) -> str:
    req_sha = provenance.sha256_file(config.REQUIREMENTS_B1_PATH)
    e1, e2, e3 = config.BUCKET_EDGES
    return f"""
---

## B1 Pre-registration & Benchmark Freeze ({date.today().isoformat()})
- **Specification**: {config.SPEC_VERSION}; metric definitions `{config.METRIC_DEFINITION_VERSION}`.
- **Benchmark**: `experiments/semantic_retrieval/benchmark/challenge_set_v1.csv`, sha256 `{sha}`.
- **Population counts**: Q={counts['Q']}, P={counts['P']}, P*={counts['P_star']}, N={counts['N']};
  slices {counts['slices']}; categories {counts['categories']};
  intents {counts['intents']}.
- **Canonical lexical-overlap variable**: `max_dice_to_acceptable` — a non-gating benchmark-definition
  field. P* = positive queries with max_dice_to_acceptable < {float(config.PRIMARY_DICE_LT):.2f}.
  Buckets B1 < {float(e1):.2f} <= B2 < {float(e2):.2f} <= B3 < {float(e3):.2f} <= B4; B3 split at
  {float(config.B3_SPLIT):.2f} into B3a (in P*) and B3b. `dice_to_preferred` and
  `max_dice_to_any_kb_row` are purely diagnostic. No Dice field is a retrieval acceptance gate.
- **Primary comparison**: Stage A `ResponseRetriever(df=train_df)` vs `{config.MODEL_NAME}`
  (revision `{config.MODEL_REVISION}`) on P*, Pass 1 (global search over the 160 training rows).
- **Decision gates (engineering, not universal claims)**: T1 ΔAcceptableRecall@1 >= +{config.T1_PP} pp;
  T2 ΔAcceptableMRR@{config.MRR_DEPTH} >= +{config.T2_MRR}; S ΔWrongIntent@1 <= +{config.S_PP} pp on P*.
- **Outcome categories (assigned in order)**: Regression; Meaningful improvement; Modest improvement;
  No meaningful difference / inconclusive — per spec §13.2.
- **Statistics**: paired bootstrap, {config.BOOTSTRAP_RESAMPLES} resamples, seed {config.SEED},
  percentile CI {config.BOOTSTRAP_CI}; Wilcoxon signed-rank {config.WILCOXON} (supporting only).
- **Resource gates**: end-to-end p95 <= {config.LATENCY_P95_LIMIT_MS} ms (p50 target <= {config.LATENCY_P50_TARGET_MS} ms);
  incremental peak RSS <= {config.INCREMENTAL_RSS_LIMIT_MB} MB; {config.RESOURCE_REPEATS} fresh-process repeats.
- **Environment pins**: `requirements-b1.txt` sha256 `{req_sha}`.
"""


def cmd_freeze(split) -> int:
    df, errors = _validate(split)
    if errors:
        print("Cannot freeze; validation failed:")
        print("\n".join(f"  - {e}" for e in errors))
        return 1
    if config.BENCHMARK_SHA256 is not None:
        print(f"Already frozen (config BENCHMARK_SHA256 = {config.BENCHMARK_SHA256}). Refusing to re-freeze.")
        return 1
    sha = provenance.sha256_file(config.BENCHMARK_PATH)
    counts = bm.population_counts(df, split)
    bm.set_config_benchmark_sha(sha)
    with open(config.DECISIONS_PATH, "a", encoding="utf-8") as f:
        f.write(_decisions_entry(sha, counts))
    print(f"FROZEN: sha256 {sha}")
    print("config.py BENCHMARK_SHA256 set; DECISIONS.md entry appended.")
    return 0


def cmd_verify_freeze(split) -> int:
    sha = provenance.sha256_file(config.BENCHMARK_PATH)
    problems = []
    if config.BENCHMARK_SHA256 != sha:
        problems.append(f"config BENCHMARK_SHA256 ({config.BENCHMARK_SHA256}) != file sha256 ({sha})")
    if not bm.decisions_mentions_sha(sha):
        problems.append("DECISIONS.md does not record the benchmark sha256")
    if problems:
        print("FREEZE VERIFICATION FAILED:")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"FREEZE VERIFIED: sha256 {sha}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["review-pack", "annotate", "validate", "freeze", "verify-freeze"])
    args = parser.parse_args(argv)
    split = b1_data.load_canonical_split()
    return {
        "review-pack": cmd_review_pack,
        "annotate": cmd_annotate,
        "validate": cmd_validate,
        "freeze": cmd_freeze,
        "verify-freeze": cmd_verify_freeze,
    }[args.mode](split)


if __name__ == "__main__":
    sys.exit(main())
