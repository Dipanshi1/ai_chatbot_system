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

---

## HCL support features: Stage A freeze exception (2026-10-30)
- **Authorization**: RECORDED (the project owner explicitly authorized additive changes for the HCL submission under the `CLAUDE.md` §3 freeze exception, limited to new `src/support/` modules, `app.py`, `src/ui/helpers.py`, `tests/`, `.gitignore` and this file).
- **Remains frozen**: `src/ml/`, serialized artifacts and `artifacts/manifest.json`, raw training data, B1
  benchmark files and research results, and the empty `b1_candidate_benchmark_audit.md`. No new dependencies,
  paid services or external APIs. Nothing is staged, committed or pushed under this exception.
- **Features**: PII redaction before inference, display and persistence; crisis triage before ML processing,
  with distinct self-harm and immediate-danger guidance and no invented contact numbers; heuristic sentiment
  that cannot affect intent or answer selection; session opt-out; consent-based redacted escalation tickets
  with manual requests and automatic offers on negative sentiment plus abstention; optional feedback; an
  administrator-visible ticket queue.
- **Data rules**: no contact details and no chat transcripts are stored. Stored rows expire after a fixed
  retention period. Stored or logged support data is never used for training or tuning.
- **Recorded behaviour change**: redacting before `pipeline.answer()` changes the pipeline's input, and so
  possibly its outcome, for messages containing PII. Policy gates and thresholds are unchanged.

---

## HCL support features: Admin Analytics and Data Export (2026-10-31)
- **Authorization**: user-instructed additive administrator features (Admin Analytics dashboard, read-only Knowledge Base viewer, and sanitized CSV data export). Status: IMPLEMENTED.
- **Strict freeze preserved**: `src/ml/`, artifacts, manifests, benchmarks, raw Excel dataset, and
  previously verified support ticket lifecycle behaviour remain strictly untouched.
- **Admin Analytics architecture**:
  - Telemetry is derived strictly from actual rows in the SQLite support store (`tickets` and `feedback`).
  - No synthetic/mock records or invented statistics are displayed.
  - The application explicitly disclaims total chat volume or unconsented student telemetry: unescalated
    interactions are processed ephemerally in memory and never logged.
  - Insufficient data states are cleanly indicated when tables have 0 rows or missing sentiments.
- **Knowledge Base viewer architecture**:
  - Exposes the approved 160-record knowledge base (`split_dataset(load_dataset())[0]`) in a read-only viewer.
  - Features real-time substring search with punctuation/hyphen normalization (e.g. "wifi" matches "Wi-Fi").
  - Strictly read-only; no modification of canonical data or artifacts.
- **Admin authentication architecture**:
  - Single shared administrator credential using salted `hashlib.scrypt` (`n=16384`, `r=8`, `p=1`, `dklen=32`), not PBKDF2.
  - Constant-time verification via `hmac.compare_digest` with fail-closed behavior when unset.
  - Session-level lockout after 3 failed attempts; gate controls access to ticket queue, analytics, and data export.
- **Data Export security safeguards**:
  - Enforces application-level authentication via `require_admin_auth()` (backed by `hashlib.scrypt` verification) prior to data generation.
  - Excludes opted-out sessions (`exclude_session_ids`).
  - Redacts PII in summaries and comments using `redact()`.
  - Neutralizes spreadsheet formula injection (CWE-1236) by prefixing `=`, `+`, `-`, `@`, `\t`, `\r` with `'`.
  - Omits internal primary keys, session IDs, passwords, hashes, and student contact details.
  - Outputs strictly UTF-8 encoded CSV.

---

## Single-Turn Conversational Follow-Up and Knowledge Base Remediation (2026-10-31)
- **Authorization**: user-instructed remediation of misleading conversational follow-up behaviour in single-turn mode. Status: IMPLEMENTED.
- **Context & Issue**: The chatbot operates strictly in single-turn mode without conversational memory. Knowledge base canned answers previously ended with unfulfillable follow-up invitations (e.g., "I can check that for you.", "Would you like me to look that up?", "Tell me your major"), prompting user follow-ups like "yes" or "when?" which resulted in confusing clarification abstentions.
- **Strict freeze preserved**:
  - `src/ml/` and all serialized model artifacts (`artifacts/intent_model.pkl`, `artifacts/tfidf_vectorizer.pkl`, `artifacts/manifest.json`) remain strictly untouched.
  - Single-turn ML architecture, TF-IDF vectorization, LinearSVC classification, lexical overlap retrieval, and deterministic policy thresholds are preserved without modification.
- **Knowledge base remediation (`data/raw/AI-Powered Chatbot.xlsx`)**:
  - All 200 responses inspected and updated to replace unsupported personalized offers with complete, self-contained answers.
  - Misleading follow-up questions and multi-turn invitations removed.
  - Clear distinction between general university information and individual student-record information established (e.g., explicitly noting that the chatbot cannot access individual student records or take actions on behalf of students).
  - No financial records, student details, dates, or institutional contacts invented.
  - All 200 responses maintain pairwise uniqueness to satisfy leakage isolation assertions between train and test splits.
- **Regression testing**:
  - Added regression test cases and assertions in `tests/test_pipeline.py` and `tests/test_app_e2e.py` verifying that standalone conversational messages ("yes", "okay", "when?") abstain cleanly (`NO_LEXICAL_EVIDENCE`) in single-turn mode without claiming conversational memory.
  - Added test in `tests/test_ui.py` confirming the user-facing disclaimer explicitly disclaims conversational memory.

---

## Admin Portal Refinement: Knowledge Base Safe Text Wrapping and Analytics Charts (2026-10-31)
- **Authorization**: user-instructed refinement of the Admin Portal: safe text wrapping for long FAQ answers in the Knowledge Base Viewer, and real ticket-status and daily-trend charts in the Analytics Dashboard. Status: IMPLEMENTED.
- **Strict freeze preserved**:
  - `src/ml/`, artifacts, manifests, benchmark files, datasets (`data/raw/`), and ticket lifecycle/privacy policies remain strictly untouched.
  - No synthetic data is fabricated; honest empty states (`st.info`) are preserved when data is 0 or absent.
  - Zero claims of full chatbot query volume analytics: captions clarify that telemetry is exclusively derived from stored tickets/feedback in the 30-day SQLite store, while unescalated interactions are ephemeral and unlogged.
- **Knowledge Base Viewer enhancements**:
  - Replaced canvas-based `st.dataframe` with an accessible, semantic HTML table wrapped inside `.kb-viewer-container` and `.kb-viewer-table`.
  - Content cells enforce safe text wrapping (`white-space: normal !important`, `word-break: break-word !important`, `overflow-wrap: break-word !important`, `line-height: 1.5`), ensuring long FAQ answers are fully readable across multiple lines without truncating ellipses.
  - Table container enforces a max-height (560px) and contained horizontal/vertical scroll (`overflow-x: auto`, `overflow-y: auto`, `min-width: 720px`, sticky header), preventing page blowout on smaller viewports.
  - All text content rendered is safely HTML-escaped with `html.escape()` to prevent XSS.
  - Real-time search query filtering and intent selectbox filtering are fully preserved.
- **Analytics Dashboard enhancements**:
  - Added real ticket-status bar chart (`st.bar_chart(status_df, x="Status", y="Tickets", color="#2563eb")`) rendered when total tickets > 0.
  - Added daily ticket creation trend bar chart (`st.bar_chart(trend_df, x="Date", y="Tickets Created", color="#0284c7")`) rendered when creation trends are non-empty.
  - When tickets or trends are empty/zero, honest empty state alerts are displayed (`st.info`) rather than empty or fabricated charts.

---

## Critical UX and FAQ Coverage Audit: Sample Prompts Remediation (2026-11-01)
- **Authorization**: user-instructed audit and remediation of responses to suggested sample questions ("How do I apply for financial aid?", "When is course registration?", "Where can I find student services?"). Status: IMPLEMENTED.
- **Audit Findings**:
  - `"How do I apply for financial aid?"`: Intent `financial_aid` correctly predicted with high margin (1.2630), but no general application FAQ exists in the workbook. Nearest neighbor was Row 198 (*"how to request a meeting with financial aid?"*) with Dice 0.5714 < 0.60, correctly triggering safety gate `NO_RELIABLE_MATCH` (`INTENT_FALLBACK`).
  - `"When is course registration?"`: Intent `course_registration` correctly predicted (margin 0.4316), but no calendar start-date FAQ exists. Nearest candidate was Row 47 (*"is late registration allowed?"*) with Dice 0.4000 < 0.60, correctly triggering safety gate `NO_RELIABLE_MATCH` (`INTENT_FALLBACK`).
  - `"Where can I find student services?"`: Broad, generic phrasing caused low classifier margin (0.0886 < 0.20), correctly triggering safety abstention `LOW_MARGIN` (`ABSTAINED`).
  - `"When is the library open on weekends?"`: Exact match to Row 1 (`campus_facilities`), returning direct FAQ answer (`ANSWERED`).
- **Remediation**:
  - Replaced unsupported sample prompts in `src/ui/helpers.py` with validated questions from the existing 200-row workbook that reliably achieve `Outcome.ANSWERED` with high margin and Dice >= 0.85:
    1. `"How do I add a course to my schedule?"` (Course Registration, Row 0)
    2. `"When will my financial aid be disbursed?"` (Financial Aid, Row 65)
    3. `"When is the library open on weekends?"` (Campus Facilities / Library, Row 1)
    4. `"Who do I contact about accessibility services?"` (Student Services, Row 13)
  - Synchronized `EXAMPLE_QUESTIONS` (sidebar) and `SUGGESTED_TILES` (welcome stage) to ensure 100% of suggested prompts yield verified direct answers.
  - Preserved all safety thresholds (`MARGIN_THRESHOLD = 0.20`, `PARAPHRASE_MIN_DICE = 0.60`), abstention policies, and canonical model artifacts without modification.
- **Regression Testing**: Added `test_displayed_sample_prompts_return_direct_faq_answers` and `test_unsupported_sample_prompts_safety_abstention_audit` in `tests/test_ui.py`.

---

## Documentation alignment after workbook remediation (2026-10-31)
- **Authorization**: RECORDED (the project owner explicitly requested updates to the README and architecture documentation; `docs/model_card.md` received factual corrections for the same figures. No code, artifact, workbook, benchmark or result file changed).
- **Knowledge-base wording**: documents describe the workbook as the unified canonical source for training and runtime FAQ answers. No separate knowledge-base file exists.
- **Runtime row count**: the audit input stated a 160-row runtime knowledge base. Verified code and a live load show `get_pipeline()` indexes all 200 rows; 160 rows apply to evaluation and the admin viewer. Documents record the verified behaviour. Whether runtime should be restricted to 160 rows is an open owner decision.
- **Corrected pre-existing errors**: vocabulary size (360 unigram features, not 602 with bigrams), class counts and singleton list, policy gate order, and the Wi-Fi example output.
- **Metrics**: unchanged. `evaluate.py` was not re-run, so stored reports keep `kb-df-db0915f97b6b`; the current train-only label is `kb-df-c85647c049e3`.

---

## Runtime FAQ terminology and evaluation reconciliation (2026-10-31)
- **Authorization**: RECORDED (the project owner directed runtime FAQ terminology and evaluation reconciliation; `src/ml/`, artifacts, reports, B1 files and the workbook were not modified).
- **Knowledge-base scope**: the 200-row runtime FAQ knowledge base and the 160-row evaluation subset are both preserved. The staff viewer shows the 200 runtime records by default and labels the evaluation subset separately. The sidebar row count now reads the live retriever.
- **Terminology**: "Approved Knowledge Base" became "Runtime FAQ Knowledge Base" and "Verified Knowledge Base Answer" became "Direct Knowledge Base Match". No wording implies external university approval.
- **Authoritative outcome counts**: 1 answered, 21 fallback, 18 abstained of 40, from the stored reports (model `linsvc-a473f4580f16+tfidf-5f62a51adde8`, policy `phase4-v1`, KB `kb-df-db0915f97b6b`, train-only 160 rows). A reported 20/14/6 split appears nowhere in the repository or its history and is impossible under this model and policy: 16 queries abstain before retrieval whatever the knowledge base. No evaluation was re-run.

---

## Final documentation consistency review (2026-10-31)
- **Authorization**: RECORDED (the project owner requested this documentation consistency review, including removal of machine-specific links and alignment of B1 status in `CLAUDE.md`. No code, artifact, report, benchmark, notebook or raw data file changed).
- **Final values**: runtime FAQ knowledge base 200 rows; evaluation knowledge base 160 rows; pipeline outcomes 1 answered, 21 fallback, 18 abstained of 40; LinearSVC accuracy 20/40; macro F1 0.334 (present classes).
- **HCL CSV identity**: `dataset.csv`, 500 rows, 6 columns, sha256 `cbf1fbd353623819c462ad76e4277c264b7dcfe41af5dc91ecb44c4d743f7526`. Windows download metadata records `university-lms.career-shaper.com`, course 14873. "HCL-supplied" is the project owner's description; nothing in the file states its publisher.
- **Superseded wording**: the 2026-10-31 viewer entry above says "approved 160-record knowledge base". The later terminology entry replaces it; earlier text stays unchanged because this file is append-only.
- **Notebook**: left unedited. Some prose cells contradict its stored outputs. Correcting them properly needs a deliberate re-execution, which is an open owner decision.
