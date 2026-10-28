"""
B1 Phase 3 — failure-mode tagging for the semantic retriever (Pass 1).

Failures: positive queries whose rank-1 outcome is INTENT_ONLY or WRONG_INTENT, and negative
probes whose rank-1 outcome is WRONG_INTENT. Automatic signatures F1/F3/F4/F5 (thresholds in
config.py); primary tag by precedence F5 > F3 > F1 > F4; anything else is UNASSIGNED for human
tagging (F2 entity collapse cannot be detected automatically). Writes results/b1_failure_analysis.csv.
The alternative-model (BGE) condition is reported only; no other model is ever run here.

Usage (repo root, after 02_run_eval.py):
  experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/scripts/03_analyze_errors.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
for _p in (str(REPO_ROOT), str(EXPERIMENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from lib import evaluation as ev  # noqa: E402
from lib.retrievers import SemanticRetriever  # noqa: E402

FIELDS = ("challenge_id", "query_text", "category", "slice", "target_intent", "acceptable_row_ids",
          "rank1_row_id", "rank1_score", "outcome_class", "failure_level", "signatures", "primary_tag", "human_tag")


def main() -> int:
    if not config.COMPARISON_CSV_PATH.exists():
        print("ABORT: run scripts/02_run_eval.py first")
        return 1
    with open(config.COMPARISON_CSV_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["retriever"] == SemanticRetriever.name and r["pass"] == ev.PRIMARY_PASS]
    out = []
    for r in rows:
        negative = r["positive"] != "True"
        if not ev.is_failure(r["outcome_class"], negative):
            continue
        score = float(r["rank1_score"]) if r["rank1_score"] else None
        sig = ev.failure_signatures(r["query_text"], score, r["outcome_class"], negative)
        out.append({**{k: r[k] for k in FIELDS if k in r},
                    "failure_level": "intent" if r["outcome_class"] == "WRONG_INTENT" else "row_only",
                    "signatures": ";".join(sig), "primary_tag": ev.primary_failure_tag(sig), "human_tag": ""})
    with open(config.FAILURE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    counts = Counter(o["primary_tag"] for o in out)
    category = None
    if config.METRICS_JSON_PATH.exists():
        category = json.loads(config.METRICS_JSON_PATH.read_text())["primary"]["outcome"]["category"]
    representation = sum(counts.get(t, 0) for t in ("F1", "F4"))
    print(f"failures: {len(out)} | primary tags: {dict(counts)}")
    print(f"UNASSIGNED (need human tag, e.g. F2): {counts.get('UNASSIGNED', 0)}")
    print(f"outcome category (provisional): {category}")
    print(f"alternative-model condition: MiniLM meets success criteria = {category == 'Meaningful improvement'}; "
          f"representation-related failures (F1+F4, plus any human F2) = {representation}. "
          "Decision is human; no alternative model was run.")
    print(f"wrote {config.FAILURE_CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
