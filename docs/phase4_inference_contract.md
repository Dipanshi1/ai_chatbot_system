# Phase 4 — Inference Contract & Safety Gating

Policy version: `phase4-v1`

## 1. Why this phase exists

Before Phase 4 the chatbot could return a confident, irrelevant canned answer:

```
"how do i connect to college faculty?"
  → technical_issue (margin 0.353, accepted at 0.20)
  → retrieval similarity 0.629 (driven by "how", "do", "to", "connect")
  → "Install the VPN client recommended on the IT site…"
```

Two facts about the knowledge base explain this:

- **All 200 bot responses are unique.** Each canned answer answers exactly one
  question, so retrieval is a *paraphrase test*, not "find something relevant in
  the intent".
- **The nearest *other* question in the same intent never shares its answer**
  (leave-one-out over the training split: 0%). Returning the nearest match for a
  novel question returns the answer to a different question.

## 2. Canonical inference flow

```
answer(message, policy=DEFAULT_POLICY)                 pipeline.py  (orchestration only)
 │
 ├─ clean_text(message)                                preprocessing.py
 │     └─ "" → no further evidence; decide() returns EMPTY_INPUT
 ├─ analyze_lexical(cleaned, classifier vocabulary)    lexical.py   → LexicalEvidence
 ├─ classifier.predict(cleaned)                        inference.py → ClassifierResult
 ├─ retriever.best_match(cleaned, predicted_intent)    retrieval.py → RetrievalCandidate | None
 ├─ decide(lexical, classifier, candidate, policy)     policy.py    → Decision   ← ONLY decision point
 └─ assemble ChatResult                                pipeline.py
```

| Component | Responsibility | Makes decisions? |
|---|---|---|
| `preprocessing.py` | lowercase, collapse whitespace, strip | no |
| `lexical.py` | tokens, content tokens, in-vocab / OOV counts | no |
| `inference.py` | LinearSVC scores, top-k, top-1 − top-2 margin | no |
| `retrieval.py` | best KB question within the predicted intent; cosine, shared content tokens, Dice | no |
| `policy.py` | answer / fallback / abstain + reason | **yes — the only place** |
| `pipeline.py` | orchestration, user-facing text, versions | no |
| `chatbot.py` | CLI: prints `result.answer` (`--debug` adds diagnostics) | no |

Evidence is always collected in full (except for empty input) so `decide()` is a
pure function over complete evidence and diagnostics are available for every
outcome.

## 3. Outcomes

| Outcome | Meaning | `answer_source` | User sees |
|---|---|---|---|
| `ANSWERED` | Confident intent **and** a paraphrase match in the knowledge base | `knowledge_base` | The matched canned response |
| `INTENT_FALLBACK` | Confident intent, but no reliable specific answer | `intent_fallback` | "I think this is about {area}, but I don't have enough information to give a reliable answer to that specific question yet." |
| `ABSTAINED` | Not enough evidence to name an area | `clarification` | Clarification request (or empty-input prompt) |

Abstention is a first-class outcome, not an error. `abstained` is `True` for both
`INTENT_FALLBACK` and `ABSTAINED` (no specific answer was given).

## 4. Gate order

The first failing gate wins, so exactly one reason is reported.

| # | Condition | Result |
|---|---|---|
| 1 | message empty after `clean_text` | `ABSTAINED` / `EMPTY_INPUT` |
| 2 | `in_vocab_content_count == 0` | `ABSTAINED` / `NO_LEXICAL_EVIDENCE` |
| 3 | `margin < margin_threshold` (equality is accepted) | `ABSTAINED` / `LOW_MARGIN` |
| 4 | paraphrase gate passes | `ANSWERED` |
| 5 | `in_vocab_content_count <= 1` **and** `content_oov_ratio >= 0.5` | `ABSTAINED` / `WEAK_LEXICAL_EVIDENCE` |
| 6 | otherwise | `INTENT_FALLBACK` / `NO_RELIABLE_MATCH` |

**Paraphrase gate** (content tokens = tokens minus sklearn `ENGLISH_STOP_WORDS`):

```
Q = query content tokens        M = matched question content tokens
dice = 2·|Q ∩ M| / (|Q| + |M|)

passes ⇔ cosine > 0  AND  dice >= 0.60  AND  (Q ∩ M) − GENERIC ≠ ∅
GENERIC = {campus, student, students, university, bot, help, request}
```

Design rules:

- Retrieval **cannot** override a low classifier margin.
- Retrieval only searches within the predicted intent.
- Cosine similarity ranks candidates; it is **not** a gate (one rare shared token
  can produce a high cosine).
- OOV ratio alone never causes abstention (27% of genuine training queries have a
  content-OOV ratio above 0.5 under a leave-one-out vocabulary). It only
  downgrades a fallback to an abstention when the decision rests on one known word.

## 5. Contract

`ChatPipeline.answer(message, policy=None) -> ChatResult` (also `answer()` and
`get_pipeline()` at module level). `ChatResult.to_dict()` is JSON-serialisable.

| Field | Purpose |
|---|---|
| `outcome` | Primary three-way result |
| `answer` | Always non-empty display text |
| `answer_source` | `knowledge_base` / `intent_fallback` / `clarification` |
| `abstained` | `outcome != ANSWERED` |
| `abstain_reason` | Non-null exactly when `abstained` |
| `intent` | **Committed** intent — set only for `ANSWERED` / `INTENT_FALLBACK`; safe to show |
| `predicted_intent` | Raw classifier top-1, kept for diagnostics even when abstaining; never shown to users |
| `margin` | Top-1 − top-2 decision score; `None` for empty input |
| `top_predictions` | Top-5 `(intent, score)` pairs |
| `similarity` | Cosine of the best candidate (diagnostic only) |
| `overlap` | Dice score used by the gate |
| `matched_question`, `matched_row_id` | Traceability to the KB row |
| `shared_content_tokens` | Why a paraphrase was accepted / rejected |
| `lexical` | `LexicalEvidence` (tokens, content, in-vocab, OOV, ratio) |
| `policy` | The effective `PolicyConfig` (makes custom-threshold runs self-describing) |
| `model_version` | `linsvc-<sha256[:12]>+tfidf-<sha256[:12]>` of the artifacts |
| `kb_version` | `kb-<sha256[:12]>` of the knowledge-base XLSX |
| `policy_version` | `phase4-v1` |

Custom thresholds: `answer(msg, policy=dataclasses.replace(DEFAULT_POLICY, margin_threshold=0.4))`.

## 6. Policy constants

| Constant | Value | Status |
|---|---|---|
| `MARGIN_THRESHOLD` (alias `ABSTENTION_THRESHOLD`) | 0.20 | prototype policy constant; not fitted; not optimal |
| `PARAPHRASE_MIN_DICE` | 0.60 | prototype policy constant; not fitted; not optimal |
| `WEAK_EVIDENCE_MAX_IN_VOCAB_CONTENT` | 1 | prototype policy constant; not fitted; not optimal |
| `WEAK_EVIDENCE_MIN_OOV_RATIO` | 0.5 | prototype policy constant; not fitted; not optimal |
| `RETRIEVAL_GENERIC_TOKENS` | 7 words above | prototype policy constant; not fitted; not optimal |

### Why these are prototype constants

- **Margin 0.20.** Correction to the Phase 3 report: 0.20 is **a reasonable
  prototype heuristic on this single 40-sample test split**, not an "optimal
  operating point". No optimisation objective was defined; the value predates the
  analysis; with 40 samples one example moves coverage by 2.5 pp and the 95%
  interval for accepted accuracy (18/26) is roughly 50%–84%, overlapping the 50%
  baseline. 10 of 26 intents are absent from the test split.
- **Dice 0.60.** There are no relevance labels to fit against (every response is
  unique). 0.60 reads as "most of the combined content words are shared". 0.5 in
  each direction was rejected because it accepted "how can i apply for financial
  aid?" → "how to request a meeting with financial aid?".
- **Weak-evidence rule and generic tokens** were chosen from training-split
  statistics only. None of the constants were tuned on the 40-example test split.

## 7. Known limitations

- **Far fewer canned answers.** In leave-one-out on the training split, only ~4.5%
  of novel phrasings find a qualifying paraphrase; most become `INTENT_FALLBACK`.
  This is intended: the alternative was the answer to a different question.
- **Wrong intent with a high margin** still yields an `INTENT_FALLBACK` naming the
  wrong area (e.g. the campus-card query → "campus facilities"; the grief query →
  "student services"). No wrong answer is given, but routing is wrong. This cannot
  be fixed lexically.
- **No synonyms or spelling correction.** "my wifi is not working" abstains
  (`wifi`, `working` are OOV).
- **Stopword list** is generic English and removes negations.
- **Retrieval index includes test-split rows.** Fine at runtime; never evaluate
  retrieval quality on the test split.
- Regression tests pin behaviour on near-duplicates of KB rows; they are not
  accuracy evidence.

## 8. Phase 5 direction

1. Build a small hand-labelled paraphrase set (50–100 queries, each labelled
   "answerable by row X" or "not answerable") to calibrate `PARAPHRASE_MIN_DICE`.
2. Margin threshold selection via out-of-fold analysis on the training split
   (repeated CV / leave-one-out), keeping the test split untouched — noting that
   classes with 1–2 examples limit stratification.
3. Collect more data for the 10 intents with ≤ 2 examples.
4. Only then: FastAPI as a thin layer over `answer()` / `ChatResult.to_dict()`.

Deferred until then: FastAPI, RAG, embeddings, LLM generation, PostgreSQL,
frontend, retraining, calibration, retrieval "rescue" of low margins,
cross-intent retrieval.
