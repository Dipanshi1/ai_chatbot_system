"""
B1 Phase 3 — head-to-head evaluation on the frozen challenge benchmark.

Stage A ResponseRetriever (read-only adapter) vs. MiniLM dense retrieval, over the 160 training
rows only. Pass 1 global (primary), Pass 2 oracle intent (diagnostic), Pass 3 Stage A predicted
intent (integration context). Writes results/b1_retrieval_comparison.csv and
results/b1_head_to_head_metrics.json. Refuses to run unless the benchmark freeze verifies.

Usage (repo root):
  HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python \
      experiments/semantic_retrieval/scripts/02_run_eval.py [--skip-resources]
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import config  # noqa: E402
from lib import benchmark as bm  # noqa: E402
from lib import data as b1_data  # noqa: E402
from lib import evaluation as ev  # noqa: E402
from lib import metrics as m  # noqa: E402
from lib import provenance, resources, retrievers  # noqa: E402
from src.ml.artifacts import verify_artifacts  # noqa: E402
from src.ml.inference import IntentClassifier  # noqa: E402
from src.ml.preprocessing import clean_text  # noqa: E402

BASELINE, CANDIDATE = retrievers.StageARetrieverAdapter.name, retrievers.SemanticRetriever.name
SLICES = ("B1", "B2", "B3a", "B3b", "B4")


def fail(msg: str) -> int:
    print(f"ABORT: {msg}")
    return 1


def check_gates(split):
    """Freeze, Stage A integrity and benchmark validity must all hold before any evaluation."""
    sha = provenance.sha256_file(config.BENCHMARK_PATH)
    if config.BENCHMARK_SHA256 is None or sha != config.BENCHMARK_SHA256:
        return f"benchmark sha256 {sha} does not match frozen config value {config.BENCHMARK_SHA256}"
    if not bm.decisions_mentions_sha(sha):
        return "DECISIONS.md does not record the frozen benchmark sha256"
    verify_artifacts()
    df = bm.read_benchmark()
    errors = (bm.schema_errors(df, split) or bm.annotation_errors(df, split)
              + bm.contamination_errors(df, split) + bm.composition_errors(df, split))
    if errors:
        return f"benchmark validation failed: {errors[:3]}"
    return None


def run_probes(items):
    """Fresh-process memory/load probes; run before the parent loads the model (low-RAM safety)."""
    probe_items = [{"query": it.query, "target_intent": it.labels.target_intent} for it in items]
    probes = []
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(config.RESOURCE_REPEATS):
            probes.append(resources.run_probe(probe_items, tmp))
    return probes


def measure_resources(items, semantic, stage_a, probes):
    queries = [it.query for it in sorted(items, key=lambda i: i.challenge_id)]
    lat_sem = resources.measure_latency(semantic, queries, config.LATENCY_CALLS, config.RETRIEVAL_DEPTH)
    lat_a = resources.measure_latency(stage_a, queries, config.LATENCY_CALLS, config.RETRIEVAL_DEPTH)
    inc = [p["incremental_peak_rss_mb"] for p in probes]
    load = [p["model_load_s"] for p in probes]
    p95 = lat_sem["end_to_end"]["p95_ms"]
    gates = {
        "latency_p95_ms": p95, "latency_p95_limit_ms": config.LATENCY_P95_LIMIT_MS,
        "latency_p95_ok": p95 <= config.LATENCY_P95_LIMIT_MS,
        "latency_p50_ms": lat_sem["end_to_end"]["p50_ms"], "latency_p50_target_ms": config.LATENCY_P50_TARGET_MS,
        "incremental_peak_rss_mb_max": max(inc), "incremental_rss_limit_mb": config.INCREMENTAL_RSS_LIMIT_MB,
        "incremental_rss_ok": max(inc) <= config.INCREMENTAL_RSS_LIMIT_MB,
    }
    gates["all_resource_gates_ok"] = gates["latency_p95_ok"] and gates["incremental_rss_ok"]
    return {
        "gates": gates,
        "probes": probes,
        "model_load_s": {"median": float(np.median(load)), "max": float(max(load))},
        "index_build_s_median": float(np.median([p["index_build_s"] for p in probes])),
        "index_size_bytes": config.INDEX_PATH.stat().st_size,
        "latency": {CANDIDATE: lat_sem, BASELINE: lat_a},
    }


def guardrail_evidence(rows, items):
    hit = {m.PREFERRED_HIT, m.ACCEPTABLE_HIT}
    by = {(r["challenge_id"], r["pass"], r["retriever"]): r for r in rows}
    g1 = []
    for it in items:
        if not it.positive:
            continue
        for p in ("pass1_global", "pass3_predicted"):
            a, b = by[(it.challenge_id, p, BASELINE)], by[(it.challenge_id, p, CANDIDATE)]
            if a["outcome_class"] in hit and b["outcome_class"] not in hit:
                g1.append({"challenge_id": it.challenge_id, "pass": p, "slice": it.slice,
                           "stage_a": a["outcome_class"], "b1": b["outcome_class"]})
    g2 = {}
    for name in (BASELINE, CANDIDATE):
        for label, member in (("negatives_N", lambda i: not i.positive), ("primary_P_star", lambda i: i.primary)):
            ids = {i.challenge_id for i in items if member(i)}
            g2[f"{name}:{label}"] = ev.score_distribution(
                [r["rank1_score"] for r in rows if r["pass"] == ev.PRIMARY_PASS and r["retriever"] == name
                 and r["challenge_id"] in ids])
    return {
        "guardrail_1_cases_stage_a_hit_b1_miss": g1,
        "guardrail_1_status": "pending_human_confirmation",
        "guardrail_2_rank1_score_distributions": g2,
        "guardrail_2_status": "pending_human_confirmation",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="B1 Phase 3 evaluation")
    parser.add_argument("--skip-resources", action="store_true", help="debug only: skip resource measurement")
    args = parser.parse_args(argv)
    retrievers.require_offline()

    split = b1_data.load_canonical_split()
    problem = check_gates(split)
    if problem:
        return fail(problem)

    df = bm.read_benchmark()
    items = ev.load_items(df, split)
    intent_of = {int(r): str(it) for r, it in zip(split.train_df.index, split.train_df["Intent"])}
    clf = IntentClassifier()
    predicted = {}
    for it in items:
        res = clf.predict(clean_text(it.query))
        predicted[it.challenge_id] = res.predicted_intent if res else None

    probes = None if args.skip_resources else run_probes(items)
    stage_a = retrievers.StageARetrieverAdapter(split.train_df)
    semantic = retrievers.SemanticRetriever(split.train_df)
    for ret in (stage_a, semantic):
        b1_data.assert_kb_is_train_only(ret.row_ids, split)

    rows = ev.run_passes([stage_a, semantic], items, predicted, intent_of)

    populations = {"P_star": lambda i: i.primary, "P": lambda i: i.positive,
                   "N_negatives": lambda i: not i.positive}
    for s in SLICES:
        populations[f"slice_{s}"] = (lambda s: lambda i: i.positive and i.slice == s)(s)
    for c in config.POSITIVE_CATEGORIES:
        populations[f"category_{c}"] = (lambda c: lambda i: i.category == c)(c)
    summary = {p: {name: ev.population_summary(rows, items, p, BASELINE, CANDIDATE, member)
                   for name, member in populations.items()} for p in ev.PASSES}

    wilcoxon = ev.primary_wilcoxon(rows, items, BASELINE, CANDIDATE)
    res = None if args.skip_resources else measure_resources(items, semantic, stage_a, probes)
    primary = summary[ev.PRIMARY_PASS]["P_star"]["deltas"]
    outcome = ev.assign_outcome_category(
        primary["AcceptableRecall@1"], primary[f"AcceptableMRR@{config.MRR_DEPTH}"], primary["WrongIntent@1"],
        wilcoxon, res["gates"]["all_resource_gates_ok"] if res else None)
    outcome["provisional"] = True
    outcome["provisional_reason"] = ("guardrails 1-2 require human confirmation"
                                     + ("" if res else "; resources not measured (--skip-resources)"))

    stage_a_kb = stage_a.kb_version
    result = {
        "metric_definition_version": config.METRIC_DEFINITION_VERSION,
        "primary": {"pass": ev.PRIMARY_PASS, "population": "P_star",
                    "n": summary[ev.PRIMARY_PASS]["P_star"]["n_queries"], "deltas": primary,
                    "wilcoxon_acceptable_rr": wilcoxon, "outcome": outcome},
        "summary": summary,
        "guardrails": guardrail_evidence(rows, items),
        "resources": res,
        "provenance": {
            "timestamp_utc": provenance.timestamp_utc(),
            "git_head": provenance.git_head(),
            "environment": resources.environment_record(),
            "packages": provenance.package_versions(),
            "model": {"name": config.MODEL_NAME, "revision": config.MODEL_REVISION},
            "benchmark_sha256": config.BENCHMARK_SHA256,
            "index_sha256": provenance.sha256_file(config.INDEX_PATH),
            "stage_a_hashes": provenance.stage_a_hashes(),
            "stage_a_kb_version": stage_a_kb,
            "parameters": {"T1_PP": config.T1_PP, "T2_MRR": config.T2_MRR, "S_PP": config.S_PP,
                           "bootstrap": [config.BOOTSTRAP_RESAMPLES, config.SEED, list(config.BOOTSTRAP_CI)],
                           "wilcoxon": config.WILCOXON, "retrieval_depth": config.RETRIEVAL_DEPTH},
            "population_counts": {k: summary[ev.PRIMARY_PASS][k]["n_queries"] for k in populations},
        },
    }
    if config.METRICS_JSON_PATH.exists():
        previous = json.loads(config.METRICS_JSON_PATH.read_text(encoding="utf-8"))
        result, msgs = ev.preserve_human_review(result, previous)
        for msg in msgs:
            print(f"WARNING: {msg}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k != "_values"]
    with open(config.COMPARISON_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    config.METRICS_JSON_PATH.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    ar, mrr, wi = (primary[k] for k in ("AcceptableRecall@1", f"AcceptableMRR@{config.MRR_DEPTH}", "WrongIntent@1"))
    print(f"P* n={result['primary']['n']} (Pass 1, {BASELINE} -> {CANDIDATE})")
    for label, d in (("AcceptableRecall@1", ar), ("AcceptableMRR@10", mrr), ("WrongIntent@1", wi)):
        print(f"  {label}: {d['baseline']:.3f} -> {d['candidate']:.3f}  delta {d['delta']:+.3f} "
              f"[95% CI {d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    print(f"  Wilcoxon (acceptable RR): {wilcoxon}")
    if res:
        print(f"  resources: {res['gates']}")
    print(f"  outcome (provisional): {outcome['category']}")
    print(f"wrote {config.COMPARISON_CSV_PATH} and {config.METRICS_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
