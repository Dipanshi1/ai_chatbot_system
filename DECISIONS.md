# Phase 5E Engineering Decisions

This document records architectural, documentation, and verification decisions established during Phase 5E (Final Documentation & Fresh-Clone Verification).

---

## 1. Documentation Structure & Terminology
- **Decision**: Adhere strictly to the requested 21-section structure for `README.md`.
- **Terminology**: Never refer to the chatbot as an LLM, RAG, or deep learning system. Describe the model accurately as a classical `LinearSVC` intent classifier over `TfidfVectorizer` paired with lexical overlap retrieval and deterministic policy gating.
- **Acronym Policy**: The term "HCL" is preserved as "HCL Student Support Chatbot / Workflow" without expanding it into any unverified or invented acronym.

## 2. Distinction Between Educational Walkthrough and Production Engine
- **Decision**: In `README.md` and `docs/architecture.md`, explicitly separate `notebooks/hcl/01_hcl_student_support_workflow.ipynb` from `src/ml/`.
- **Rationale**: The notebook is an educational, reproducible companion designed for transparent walkthroughs. It calls tested library functions directly and runs candidate training in memory. It is not an alternative or competing production implementation.

## 3. Fresh-Clone Verification Environment
- **Decision**: To verify fresh clone usability without disturbing the project's existing `.venv`, create a temporary virtual environment at `/tmp/hcl-chatbot-test-venv`.
- **Scope**: Install dependencies from `requirements.txt`, run the full test suite (`python -m unittest discover -s tests -v`), run `python src/evaluate.py`, and test headless Streamlit startup (`streamlit run app.py --server.headless true`). Remove the temporary environment after verification.

## 4. Protected Canonical Artifacts
- **Decision**: The existing canonical artifacts (`artifacts/intent_model.pkl`, `artifacts/tfidf_vectorizer.pkl`, and `artifacts/manifest.json`) are immutable release baselines. Normal training (`python src/train.py`) executes entirely in memory. Disk writes require `--save` and `--overwrite` flags to guard against accidental replacement.

## 5. Security & Secret Exposure
- **Decision**: Confirm via `git grep` and `git ls-files` that no API keys, credentials, tokens, or environment files are tracked. Document the zero-secret status in release verification reports.

---

## B1 Pre-registration & Benchmark Freeze (2026-10-28)
- **Specification**: B1 Revision 2 (final); metric definitions `b1-rev2`.
- **Benchmark**: `experiments/semantic_retrieval/benchmark/challenge_set_v1.csv`, sha256 `b80b9b95d1f085a1f1537077f72ae485da2ab2cdb6d12cbdc92928f2b184bd6b`.
- **Population counts**: Q=72, P=58, P*=58, N=14;
  slices {'B1': 36, 'B2': 20, 'B3a': 2, 'NEG': 14}; categories {'distractor_negative': 14, 'slang_informal': 18, 'synonym_substitution': 22, 'syntactic_inversion': 18};
  intents {'academic_advising': 7, 'academic_policies': 7, 'academic_support': 4, 'accessibility': 4, 'administration': 6, 'campus_facilities': 7, 'career_services': 3, 'course_information': 1, 'course_registration': 4, 'financial_aid': 5, 'housing': 6, 'mental_health': 5, 'research_support': 3, 'safety': 4, 'student_services': 2, 'technical_issue': 4}.
- **Canonical lexical-overlap variable**: `max_dice_to_acceptable` — a non-gating benchmark-definition
  field. P* = positive queries with max_dice_to_acceptable < 0.60.
  Buckets B1 < 0.25 <= B2 < 0.50 <= B3 < 0.75 <= B4; B3 split at
  0.60 into B3a (in P*) and B3b. `dice_to_preferred` and
  `max_dice_to_any_kb_row` are purely diagnostic. No Dice field is a retrieval acceptance gate.
- **Primary comparison**: Stage A `ResponseRetriever(df=train_df)` vs `sentence-transformers/all-MiniLM-L6-v2`
  (revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) on P*, Pass 1 (global search over the 160 training rows).
- **Decision gates (engineering, not universal claims)**: T1 ΔAcceptableRecall@1 >= +15 pp;
  T2 ΔAcceptableMRR@10 >= +0.15; S ΔWrongIntent@1 <= +5 pp on P*.
- **Outcome categories (assigned in order)**: Regression; Meaningful improvement; Modest improvement;
  No meaningful difference / inconclusive — per spec §13.2.
- **Statistics**: paired bootstrap, 1000 resamples, seed 42,
  percentile CI (2.5, 97.5); Wilcoxon signed-rank {'alternative': 'two-sided', 'zero_method': 'wilcox'} (supporting only).
- **Resource gates**: end-to-end p95 <= 50 ms (p50 target <= 20 ms);
  incremental peak RSS <= 500 MB; 3 fresh-process repeats.
- **Environment pins**: `requirements-b1.txt` sha256 `2fd8c97f2be269cbfdedc83653617be888e1ae82492652546e2373c71dc3a88b`.

---

## Stage A UI error-handling fix (2026-10-28)
- **Decision (user-authorized submission fix)**: `app.py` no longer renders raw exception text. Query
  failures show only the generic error message; startup failures show a generic message with the local
  verification command. Details are logged server-side via `logging`.
- **Rationale**: user-facing output must not expose stack traces, paths or internal details
  (`CLAUDE.md` §7). No ML behaviour, artifacts or dependencies changed.
- **Tests**: new `tests/test_app_e2e.py` (Streamlit `AppTest`): startup, supported query, fallback,
  over-long input, example button + clear, and two leak-prevention tests (query and startup failures).

---

## B1 Phase 3 evaluation implemented without the authoritative specification (2026-10-29)
- **Context**: the "Stage B1 Research & Technical Specification, Revision 2" document is not available
  locally (targeted search found no copy). On user instruction, Phase 3 uses the repository as the source
  of truth: `experiments/semantic_retrieval/config.py` (gates, statistics, resource limits, failure
  thresholds), the freeze entry above (category names, statistics, gates), `lib/metrics.py` (metric
  definitions) and the B1 README (passes, primary population and metric). Nothing below is claimed to
  come from the missing document.
- **Implementation decisions (not established elsewhere in the repo)**:
  1. Outcome categories, assigned in order: *Regression* if the 95% CI upper bound of ΔAcceptableRecall@1
     or ΔAcceptableMRR@10 is < 0, or ΔWrongIntent@1 (point estimate) > +5 pp; *Meaningful improvement* if
     both point estimates meet T1/T2, both CI lower bounds are > 0, Wilcoxon p < 0.05 and the resource
     gates hold (labelled "resource limit exceeded" / "resources not measured" otherwise); *Modest
     improvement* if either CI lower bound is > 0; else *No meaningful difference / inconclusive*
     (flagged underpowered when a point estimate meets its threshold).
  2. Guardrails 1–2 (catastrophic regression; negative-probe score separation) are qualitative: the
     script records evidence and marks both `pending_human_confirmation`; the category is provisional.
  3. Wilcoxon runs on per-query AcceptableMRR@10 reciprocal ranks over P* in Pass 1; paired bootstrap is
     computed for every population and slice (descriptive outside P*).
  4. Resource gate uses the maximum incremental peak RSS over the 3 fresh-process probes; probes run
     before the parent process loads the model (low-RAM safety).
  5. Failures = Pass 1 semantic positives with INTENT_ONLY/WRONG_INTENT and negatives with WRONG_INTENT;
     signatures use config thresholds; primary tag precedence F5 > F3 > F1 > F4; F2 is left for human tagging.
- **Result (P*, n = 58, Pass 1)**: AcceptableRecall@1 0.276 → 0.828 (Δ +0.552, 95% CI +0.414…+0.690);
  AcceptableMRR@10 0.341 → 0.891 (Δ +0.550, CI +0.434…+0.667); WrongIntent@1 0.483 → 0.103
  (Δ −0.379); Wilcoxon p = 1.4e-8; p95 latency 16.5 ms; max incremental RSS 278.8 MB.
  Provisional category: **Meaningful improvement**, pending the guardrail confirmations.
- **Caveats recorded with the result**: Pass 3 (behind the Stage A classifier) WrongIntent@1 rises
  0.362 → 0.655 because the semantic retriever never abstains (Stage A returns no candidate for 18/58);
  negative-probe rank-1 scores (p50 0.56) barely separate from P* (p50 0.61); slices B3b/B4 are empty;
  measured on WSL2 with 3.8 GB RAM visible and ~0.2 GB available at start.

---

## B1 human review of guardrails and failure tags (2026-10-29)
- **Guardrail 1 — violated** under the implemented operational definition (a positive query where Stage A
  ranks an acceptable row first and B1 does not). One observed case: CH_055 ("need a tutor asap") — Stage A
  ranks acceptable row 162 first; B1 ranks non-acceptable row 98 first (0.529) and 162 second (0.521). The
  authoritative specification was unavailable and gave no severity threshold; this interpretation is not
  claimed to come from it.
- **Guardrail 2 — violated (qualitative)**: reliable score separation for unanswerable queries was not
  demonstrated (B1 rank-1 p95 0.806 on negatives vs 0.796 on P*; medians 0.560 vs 0.613). No threshold
  exists in the repository; "violated" is recorded because the status is binary.
- **Human failure tags**: F2 for CH_002 (major vs minor) and CH_058 (first-time ID pickup vs replacement).
  CH_009, CH_030, CH_031, CH_052, CH_055, CH_065 and CH_068 remain UNASSIGNED.
- **Outcome category**: unchanged at the quantitative result ("Meaningful improvement" on the
  pre-registered gates). The repository defines no rule for how guardrail violations change the category,
  so the final category is left open for a user decision rather than assigned post hoc.
- **Reporting decision (post-evaluation, project owner, 2026-10-29)**: the quantitative category
  "Meaningful improvement" is kept as the benchmark result; **overall acceptance is withheld** because both
  guardrails are marked violated under the current operational definitions. This is an explicit reporting
  decision made after seeing the results, not a rule from the unavailable specification; the measured
  retrieval improvement is not relabelled as a regression.
- **Review preservation**: `02_run_eval.py` now carries human guardrail decisions forward only when their
  evidence (and, for the acceptance decision, the computed category) is unchanged; otherwise it resets to
  pending and keeps the old decision under `previous_human_review`. `03_analyze_errors.py` carries
  `human_tag` values for unchanged failures and aborts rather than drop a tag (override:
  `--allow-dropped-tags`).
