# B1 — Semantic Retrieval Experiment

**Status:** Phase 1 complete; benchmark human-reviewed (72/72 explicit decisions) and **frozen**
(sha256 `b80b9b95d1f085a1f1537077f72ae485da2ab2cdb6d12cbdc92928f2b184bd6b`, recorded in `config.py` and
`DECISIONS.md`). Phase 3 evaluation has been run. The authoritative Revision 2 document was not available,
so Phase 3 follows the repository's recorded parameters; the rules it had to add are documented in
`DECISIONS.md`.

**Source of truth:** B1 Experimental Design Specification, **Revision 2 (final)**.
This is an isolated experiment; Stage A (`src/`, `artifacts/`, `tests/`, `reports/`, `app.py`,
Stage A requirements) is frozen and is not modified.

## Methodology summary

- **Question:** can dense semantic retrieval meaningfully improve retrieval for student-support
  queries with weak lexical overlap with the knowledge base, at the exact-row and intent level,
  without an unacceptable increase in wrong-intent retrieval?
- **Comparison:** the actual Stage A `ResponseRetriever(df=train_df)` (stopword-free TF-IDF,
  ranked through a read-only adapter with exact parity to `best_match`) vs.
  `sentence-transformers/all-MiniLM-L6-v2` (pinned revision, CPU, unit-normalised embeddings,
  dot-product search over a 160 × 384 float32 NumPy matrix).
  **No vector database or external retrieval server is required.**
- **Knowledge base:** the 160 canonical training rows only. The 40 Stage A test rows are never
  indexed, embedded, labelled or used as drafting seeds.
- **Benchmark:** 72 human-reviewed queries — 22 synonym substitution, 18 syntactic inversion,
  18 slang/informal (positive), 14 distractor/negative probes.
- **Primary population P\*:** positive queries for which no acceptable KB answer has Dice ≥ 0.60
  (`max_dice_to_acceptable` < 0.60, at least 40 queries). `max_dice_to_acceptable` is a non-gating
  benchmark-definition field; `dice_to_preferred` and `max_dice_to_any_kb_row` are diagnostics.
  No Dice field is a retrieval acceptance gate.
- **Primary evaluation:** Pass 1 (global search over the 160-row KB). Pass 2 (oracle intent) is
  diagnostic; Pass 3 (Stage A predicted intent) is integration context.
- **Primary metric:** AcceptableRecall@1. Engineering decision gates on P\* / Pass 1:
  ΔAcceptableRecall@1 ≥ +15 pp, ΔAcceptableMRR@10 ≥ +0.15, ΔWrongIntent@1 ≤ +5 pp.
  Paired percentile bootstrap (1,000 resamples, seed 42); Wilcoxon signed-rank as supporting
  evidence only. Resource gates: p95 end-to-end latency ≤ 50 ms, incremental peak RSS ≤ 500 MB.
- **Interpretation rule:** "semantic retrieval is better" is never claimed from raw Recall@1 alone
  (spec §13.4).

All pre-registered parameters live in [config.py](config.py).

## Layout

| Path | Purpose |
|---|---|
| `requirements-b1.txt` | B1-only dependencies (includes `../../requirements.txt`; torch pinned to `+cpu`) |
| `config.py` | Pre-registered parameters, paths, model revision |
| `lib/` | Shared code: data split & leakage guards, exact Dice, retrievers, metrics, statistics, resource probe, provenance, benchmark validation |
| `benchmark/build_benchmark.py` | `review-pack`, `annotate`, `validate`, `freeze`, `verify-freeze` |
| `benchmark/challenge_set_v1.csv` | Frozen challenge benchmark (72 human-verified queries) |
| `benchmark/challenge_set_v1_draft90.csv`, `human_review_decisions.json`, review/audit files | Audit evidence: original 90-row draft, recorded human decisions, review reports |
| `scripts/01_build_index.py` | Builds `artifacts/kb_train_embeddings.npy` (git-ignored, regenerable) |
| `scripts/02_run_eval.py`, `scripts/03_analyze_errors.py` | Phase 3 evaluation and failure-mode tagging (`results/`) |
| `lib/evaluation.py` | Phase 3 logic: passes, population summaries, outcome category, failure tagging |
| `tests/test_semantic_retriever.py` | B1 unit tests |
| `results/` | Phase 3 outputs |

## Setup

From the repository root (Python 3.12):

```bash
python3 -m venv experiments/semantic_retrieval/.venv-b1
experiments/semantic_retrieval/.venv-b1/bin/pip install -r experiments/semantic_retrieval/requirements-b1.txt
```

Download the pinned model once (the only network use):

```bash
experiments/semantic_retrieval/.venv-b1/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
```

Every later command runs with `HF_HUB_OFFLINE=1`.

## Commands

Build the KB index:

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/scripts/01_build_index.py
```

Benchmark tool (`review-pack` | `annotate` | `validate` | `freeze` | `verify-freeze`):

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/benchmark/build_benchmark.py review-pack
```

B1 tests:

```bash
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python -m unittest discover -s experiments/semantic_retrieval/tests -v
```

Stage A tests must keep passing in the Stage A environment:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Findings

Measured on WSL2 (3.8 GB RAM visible, ~0.2 GB available at start), frozen benchmark `b80b9b95…`.

**Primary — P\* (n = 58), Pass 1 global search over the 160 training rows:**

| Metric | Stage A TF-IDF | B1 MiniLM | Δ (95% paired bootstrap CI) |
|---|---|---|---|
| AcceptableRecall@1 | 0.276 | 0.828 | +0.552 (+0.414 … +0.690) |
| AcceptableMRR@10 | 0.341 | 0.891 | +0.550 (+0.434 … +0.667) |
| WrongIntent@1 | 0.483 | 0.103 | −0.379 (−0.534 … −0.241) |

Wilcoxon (supporting): p = 1.4e-8 (43 non-zero pairs). Resources: end-to-end p50 12.9 ms, p95 16.5 ms;
max incremental peak RSS 278.8 MB; model load median 0.68 s. **Provisional outcome: Meaningful
improvement** — the engineering gates hold; guardrails 1–2 still need human confirmation.

Interpretation and limitations:
- Gains are largest where lexical overlap is lowest (B1 slice: 0.056 → 0.750) and for synonym
  substitutions (0.136 → 0.909); slang/informal improves least (0.333 → 0.667).
- **Integration caveat:** behind the Stage A classifier (Pass 3), WrongIntent@1 rises 0.362 → 0.655,
  because dense retrieval always returns a row while Stage A abstains (no candidate for 18/58). Using
  B1 in the product would need a semantic similarity gate; none was calibrated here.
- Negative probes: rank-1 semantic scores (p50 0.56) barely separate from P* (p50 0.61), so scores alone
  cannot reject unanswerable queries; WrongIntent@1 on negatives 0.714 → 0.643 (CI includes 0).
- Guardrail 1 evidence: one case (CH_055) where Stage A hit and B1 missed. Slices B3b/B4 are empty.
- Failure analysis (Pass 1, B1): 19 failures — 9 negative-probe (F5), 1 low-similarity (F4),
  9 UNASSIGNED awaiting human tagging. Index build 2.46 s exceeded its non-gating 1.5 s limit under
  memory pressure. No alternative embedding model was run.
