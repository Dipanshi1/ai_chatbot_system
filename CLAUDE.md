# CLAUDE.md — University Student Support AI Chatbot

Stable engineering contract. Explicit user instructions override it (§8). Approved specifications and
`DECISIONS.md` define *what* to build; this file defines *how* to work.

## 1. Repository map
- **Stage A (complete, frozen):** `src/` (ML core, CLI `src/chatbot.py`, `src/ui/`), `app.py` (Streamlit),
  `artifacts/` (pickles + `manifest.json`), `data/raw/AI-Powered Chatbot.xlsx`, `tests/`, `reports/`,
  `notebooks/`, `docs/`, `requirements.txt`, `requirements-dev.txt`. Environment: `.venv/`.
- **Stage B1 (isolated experiment):** `experiments/semantic_retrieval/`. Environment: its own `.venv-b1/`
  (git-ignored), installed from `requirements-b1.txt`.
- **B1 specification:** "Stage B1 Research & Technical Specification: Semantic Retrieval Experiment,
  Revision 2" is authoritative but **not stored in this repository**. `experiments/semantic_retrieval/config.py`
  holds its pre-registered parameters and the B1 README summarises it; neither replaces it. When a task
  depends on spec wording not captured in `config.py`, ask the user for the document — never reconstruct it.
- **Decisions:** `DECISIONS.md` (append-only record of approved decisions).
- **Current B1 state (user-maintained):** Phase 1 complete; benchmark awaiting human label review, then
  freeze. Phase 3 not started.

## 2. Commands (repo root; do not invent others)
`<pattern>` is a placeholder: replace it with a test name, class or substring before running
(e.g. `test_ui` for Stage A, `TestDiceAndBuckets` for B1). Do not type it literally.
```bash
# Targeted (during development)
.venv/bin/python -m unittest tests.test_policy                      # one Stage A module
.venv/bin/python -m unittest discover -s tests -k <pattern>         # Stage A subset
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python -m unittest experiments/semantic_retrieval/tests/test_semantic_retriever.py -k <pattern>

# Full check (milestones and releases) — report each line separately
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -c "from src.ml.artifacts import verify_artifacts; print(verify_artifacts())"
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python -m unittest discover -s experiments/semantic_retrieval/tests -v

# B1 benchmark tool (acts only on benchmark/challenge_set_v1.csv; no path option; non-zero exit on failure)
HF_HUB_OFFLINE=1 experiments/semantic_retrieval/.venv-b1/bin/python experiments/semantic_retrieval/benchmark/build_benchmark.py <review-pack|annotate|validate|freeze|verify-freeze>

# Apps
.venv/bin/streamlit run app.py
.venv/bin/python src/chatbot.py [--debug]
```
No linter, type checker or pytest is configured: do not add them; report them as "not configured".
Check exit codes directly — do not pipe a command whose exit status you need.

## 3. Frozen boundaries
- Do not modify any Stage A path above, its behaviour, dependencies or artifact bytes without explicit user
  authorization and a `DECISIONS.md` entry. B1 code may import Stage A read-only.
- Never install packages into `.venv/`. B1 packages go only into `.venv-b1/` via the pinned
  `requirements-b1.txt` (torch `+cpu`).
- Load pickled artifacts only from this repository and only after `verify_artifacts()` passes.
- Do not commit or push unless the current task explicitly asks.

## 4. Research integrity (B1)
The spec governs benchmark design, leakage rules, passes, metrics, thresholds, statistics and resource
limits. Additionally:
- The KB is the 160 canonical training rows only. Stage A test rows may be read only by Stage A evaluation
  code and the exact-duplicate contamination check — never as KB rows, labels, query seeds, review evidence
  or tuning data. Never embed `Bot Response` text.
- Tune nothing on the challenge set or the Stage A test split. Never change pre-registered criteria to
  improve results; record approved changes in `DECISIONS.md`.
- Automated checks and heuristics are not semantic verification. Write `review_status=verified` only to
  transcribe explicit, row-level decisions of the human reviewer — never on the agent's own judgement or a
  blanket "proceed".
- No `freeze` until `validate` passes on human-verified rows; no Phase 3 until `verify-freeze` passes.
- Run B1 with `HF_HUB_OFFLINE=1`. Before memory-sensitive runs, measure available memory (e.g. `free -m`)
  and record it with results; report measured latency and memory only, never estimates as measurements.
- Temporary files (review packs, scratch scripts, candidates) stay outside the repository unless the task
  names a repo path.

## 5. Workflow
1. **Inspect** the relevant code, tests and spec section first; no repo-wide audit when a focused read is
   enough. Give a short plan only for non-trivial or risky work.
2. **Classify risk** (low/medium/high): interface or schema change, data contamination, frozen artifacts,
   security boundary, shared state, reversibility, cost. Stop and ask on critical uncertainty about data
   integrity, research validity, security, cost or irreversible actions; otherwise take the conventional
   choice and proceed. If a prerequisite fails, stop the dependent step and report it.
3. **Test first:** for new behaviour or a bug fix, write the failing test, run it and see it fail, then
   implement until it passes. Refactors must keep existing tests passing unchanged.
4. **Smallest correct change.** No unrelated refactors, new dependencies or abstractions without a
   demonstrated need. Recommend one approach and its key trade-off.
5. **Verify by risk:** targeted tests while developing, then the affected suite; the full check (§2)
   before a milestone or release.
6. **Review the diff** (`git status --short`, `git diff`): no stray files, secrets, generated artifacts,
   test-data leakage, or Stage A changes during B1 work.
7. **Report:** what changed and why; files; each check actually run with its result; checks not run and
   why; limitations and open decisions; `git status --short`.

## 6. Testing rules
- Framework: `unittest` only, in `tests/` (Stage A) or `experiments/semantic_retrieval/tests/` (B1).
- **Unit (always for changed logic):** normal, boundary, invalid/empty input, failure paths, deterministic
  ordering and tie-breaks, plus a regression test for every confirmed bug.
- **Integration (when a component boundary changes):** real boundaries (split → preprocessing → classifier
  → retriever → policy → `ChatResult`) with fixtures and local doubles, never live network services.
- **E2E (when a user-facing path changes):** use the existing Streamlit `AppTest` (`streamlit.testing.v1`)
  for `app.py` and a subprocess run for `src/chatbot.py`, covering the affected journey (answer,
  fallback/abstention, invalid input, failing dependency without a fabricated answer). No new frameworks.
- **Concurrency (only when shared state is touched — e.g. module singletons like `get_pipeline`, Streamlit
  cached resources):** bounded local runs (<= 4 workers) checking correctness, exceptions, cross-request
  leakage and latency. Never claim scalability from a local run; no uncontrolled stress tests.
- **Integrity:** never delete, skip or weaken a failing test, mock the behaviour under test, or report an
  unexecuted check as passed. Distinguish code, environment and pre-existing failures. Keep tests
  deterministic (fixed seeds, no sleeps, no shared mutable state).

## 7. Security
Treat all input as untrusted: validate type and length (`MAX_MESSAGE_CHARS`); no stack traces, paths,
environment variables or secrets in user-facing output or logs; no `eval`/dynamic execution; no path
traversal; keep secrets out of git. Add auth or rate limiting only if the architecture gains a matching
boundary. Passing tests are not proof of security. No paid APIs, cloud resources or billable actions without
explicit authorization.

## 8. Precedence and done
Precedence: platform safety → explicit current user instructions → this file → approved spec and
`DECISIONS.md` → existing conventions → general best practice. On conflict, name it and take the safest
permitted path; never silently break a higher rule.

Done = acceptance criteria met; tests written with the change and passing at the required level; diff
reviewed; frozen boundaries intact; docs updated only where behaviour, setup or decisions changed; results
and limitations reported honestly. Otherwise report the task as blocked or partial.

Keep explanations brief: the developer is learning, so explain new concepts once, not repeatedly.
