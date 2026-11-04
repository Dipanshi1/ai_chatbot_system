# Project Report and Synopsis Inserts

Content to place in the Project Report and Project Synopsis. Every figure traces to the repository. Items in [brackets] must be supplied by the project team.

## Fixed Facts

| Item | Value | Source |
| :--- | :--- | :--- |
| Project title | AI-Powered Chatbot for University Student Support | HCL project list |
| Project type | Application (classical machine learning) | README |
| Canonical workbook | `AI-Powered Chatbot.xlsx`, 200 rows, 10 columns, 26 intents | `reports/evaluation_summary.json` |
| Train / test split | 160 / 40, seed 42, unstratified | `reports/evaluation_summary.json` |
| Model | TF-IDF (unigrams, 360 features) + LinearSVC | `artifacts/manifest.json` |
| LinearSVC accuracy | 20/40 = 50.0%, 95% Wilson CI [0.352, 0.648] | `reports/canonical_model_comparison.csv` |
| Macro F1 | 0.334 (18 present classes); 0.231 (all 26 classes) | `reports/canonical_model_comparison.csv` |
| Weighted F1 | 0.510 | `reports/canonical_model_comparison.csv` |
| Majority baseline accuracy | 7/40 = 17.5% | `reports/canonical_model_comparison.csv` |
| Repeated splits (30 seeds) | Mean 0.406, std 0.064, range 0.250-0.525 | `reports/repeated_split_summary.csv` |
| Pipeline outcomes (40 queries) | 1 answered, 21 intent fallback, 18 abstained | `reports/evaluation_summary.json` |
| Committed intent accuracy | 16/22 = 72.7%, 95% Wilson CI [0.518, 0.868] | `reports/evaluation_summary.json` |
| Wrong-intent answers | 0 of 40 | `reports/pipeline_evaluation.csv` |
| Wrong-area fallbacks | 6 of 40 | `reports/pipeline_evaluation.csv` |
| Evaluation knowledge base | 160 rows (training split) | `reports/evaluation_summary.json` |
| Runtime FAQ knowledge base | 200 rows (whole workbook) | `src/ml/pipeline.py` |
| Separate HCL dataset | 500 rows, 6 columns, no message text; not used by the chatbot | `DECISIONS.md` |
| Stage A test suite | 266 tests passing (263 on 2026-10-31; +3 sample prompt tests on 2026-11-01) | `tests/` automated test suite (README §16) |

## Project Synopsis (by Template Heading)

**Cover page**: [student names, roll numbers, sections, group members, project guide, signatures, remarks, submission date].

**Problem Introduction**: Students repeatedly ask common questions about registration, financial aid, campus services and support. Wrong answers on these topics can cause real harm.

**Motivation**: A small labelled dataset calls for a transparent classical approach that admits uncertainty rather than generating unverified text.

**Project Objective**: Classify questions into 26 intents, return a canned answer only on a strong paraphrase match, name the likely support area when no reliable answer exists, and abstain otherwise.

**Scope of the Project**:
- In scope: single-turn FAQ support, PII redaction, phrase-based crisis guidance, heuristic sentiment, session opt-out, consent-based escalation tickets, optional feedback, and a password-protected staff portal.
- Out of scope: generative answers, multi-turn memory, access to student records, monitored crisis response, and per-user accounts or RBAC.

**Related Previous Work**: [Team literature summary]. The repository's own related experiment is B1: a semantic-retrieval comparison on a frozen 72-query benchmark. It improved retrieval on the pre-registered metrics, but overall acceptance is withheld and it is not deployed.

**1.3 Software and Hardware Requirements**: Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, Streamlit 1.64.0, joblib 1.6.0, openpyxl 3.1.5, SQLite (Python standard library). No paid services or external APIs. Development environment: Linux on WSL2. [Add team hardware.]

**1.4 Proposed Method**: Redact PII, run crisis and request triage, clean text, classify with TF-IDF and LinearSVC, retrieve within the predicted intent, and apply the safety policy to produce answered, intent fallback or abstained.

**1.5 Deliverables**: Streamlit student chat, staff portal, CLI, trained artifacts with SHA-256 manifest, evaluation reports, educational notebook, automated tests and documentation.

**Measures of success**: Report the Fixed Facts above, with confidence intervals alongside accuracy.

**1.6 Stakeholders**: Students, student support staff and system administrators. [Add the guide and institution details.]

**1.7 Gantt Chart**: [Team to prepare planned dates.] Repository milestones: initial baseline 2026-09-30; inference and safety policy 2026-10-01; reproducible evaluation and Streamlit app 2026-10-05; B1 experiment 2026-10-28 to 2026-10-29; support features, response remediation, and sample prompt audit 2026-10-30 to 2026-11-01.

**1.8 References**:
1. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
2. Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, 22(158), 209-212.
3. Mitchell, M. et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT\* 2019*, 220-229.
4. Streamlit documentation.
5. [Team references.]

## Project Report (Required Content)

1. **Data sources**: Describe the workbook as the unified canonical source. It is both the training source and the runtime FAQ source; no separate knowledge-base file exists.
2. **Feature boundary**: Only `User Message` is a model feature. `Bot Response` is excluded from feature extraction.
3. **Two knowledge bases**: Explain that evaluation uses 160 rows to avoid leakage, while the live chatbot searches 200 rows. State that metrics describe the 160-row configuration and that performance of the live index on unseen questions is unmeasured.
4. **Separate datasets**: Keep the 200-row workbook and the 500-row HCL CSV distinct. No metric comes from the CSV.
5. **Canned-response remediation**: All 200 `Bot Response` entries were rewritten to remove offers a single-turn system cannot fulfil. All other columns and the 160 training-row identities are unchanged. No retraining or artifact regeneration occurred. Vocabulary, IDF values, classifier predictions and retrieval rankings are unchanged.
6. **Results tables**: Use the Fixed Facts. Label 0.334 as macro F1 over present classes and show 0.231 beside it.
7. **Interpretation**: Seed 42 is optimistic: only 3 of 30 seeds reached 0.500 or higher. Answer correctness is not measured.
8. **Results provenance**: The stored reports date from 2026-10-05, before remediation. Outcome counts are unaffected, but report KB labels and one stored answer predate it.
9. **Support layer limits**: Redaction and crisis triage are pattern-based and not exhaustive. Sentiment accuracy is unmeasured. The chat is not monitored. Crisis guidance contains no phone numbers. Staff cannot contact students because no contact details are stored. Staff access uses one shared password.
10. **Testing**: Report the test count with its date. Do not present passing tests as proof of security.

## Do Not Claim

- Mentor, guide, HCL or university approval of the system or its answers.
- Verified, official or institutional status for the FAQ answers.
- 20 answered, 14 fallback, 6 abstained: this split is not valid for this system.
- High accuracy, production readiness, or that the bot uses an LLM, RAG or semantic search in deployment.
- Zero incorrect answers: only wrong-intent answers were measured.
- Any file named `knowledge_base.parquet`.
- Emergency contact numbers, unless verified by the institution and supplied by the team.
