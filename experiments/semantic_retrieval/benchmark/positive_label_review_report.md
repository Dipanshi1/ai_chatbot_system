# B1 Positive-Label Review Report

> **Automated pre-review, not verification.** Every flag below is a transparent lexical/structural
> heuristic. None of it establishes semantic correctness, and no row has been verified. Per spec
> Revision 2 §3.2, a **human reviewer other than the implementing agent** must inspect every training
> row of each positive query's target intent and is the only party who may set `review_status=verified`.

- Generated: 2026-10-26 from `challenge_set_v1_assistant_reviewed.csv`
  (sha256 `10e5399a2be94a6bc2e46f08d3fd9f271c113d928f8c855bd3c1479e39cfbf6a`) and the **160 training rows only**. Stage A test rows were not used as labelling
  evidence; they are touched only by the existing exact-duplicate contamination check.
- Positives reviewed by automated checks: **58** · structurally valid: **58** · with any flag: **18** · with HIGH/MEDIUM flags: **16**.

## 1. Verification gate (actual repository behaviour)

- `build_benchmark.py` takes one positional mode: `review-pack | annotate | validate | freeze | verify-freeze`.
  It has **no path option**: every mode reads `config.BENCHMARK_PATH` = `benchmark/challenge_set_v1.csv`.
  This reviewed candidate is therefore **not** what `validate`/`freeze` check until its content is
  promoted into `challenge_set_v1.csv` (currently the 90-row Phase 1 draft).
- `validate` passes only if: schema rules (incl. verified-row rules), stored Dice fields = fresh
  annotation, contamination checks, 72 rows, categories 22/18/18/14, **every row `verified`**, |P*| >= 40.
- `freeze` re-runs validation, refuses if `config.BENCHMARK_SHA256` is already set, then writes the CSV
  sha256 into `config.py` and appends the pre-registration entry to `DECISIONS.md`.
- `verify-freeze` checks the CSV sha256 against `config.py` and `DECISIONS.md`.

## 2. Flag definitions

| Flag | Priority | Meaning |
|---|---|---|
| `STRUCT` | HIGH | Row fails a schema rule (spec §3.3). |
| `TARGET≠SEED` | HIGH | target_intent differs from the intent of the row the query was drafted from. |
| `SEED-NOT-ACCEPTABLE` | HIGH | The seed row is not in acceptable_row_ids (label diverges from drafting intent). |
| `CROSS-INTENT-ACCEPTABLE` | HIGH | An acceptable row is outside target_intent; spec §3.1 requires a rationale in notes. |
| `COMPETING-ROW` | MEDIUM | A non-acceptable row IN the target intent has higher lexical support (max Dice vs question/response) than every acceptable row, and >= 0.20: possible missed acceptable row or mislabel. |
| `MULTI-ACCEPTABLE` | MEDIUM | More than one acceptable row: confirm each response answers the same need, and the preferred choice. |
| `NEAR-DUP-QUERY` | MEDIUM | Another benchmark query has content-token Dice >= 0.60 with this one. |
| `OUTSIDE-LURE` | LOW | A row OUTSIDE the target intent has question-Dice >= 0.50 and more support than every acceptable row: check the intent label. |
| `P*-BOUNDARY` | LOW | 0.50 <= max_dice_to_acceptable < 0.60 (B3a): a label change could move the query out of P*. |
| `NO-RATIONALE` | LOW | notes contain no text beyond 'drafted from row N'. |

Lexical support of a training row = max(content-token Dice with its question, with its response),
using the Stage A tokenizer and stopword list. Paraphrase queries are *designed* to have low overlap,
so low support alone is not flagged.

## 3. Prioritized exceptions

| ID | Top priority | Query | target_intent | acceptable (preferred) | Flags |
|---|---|---|---|---|---|
| CH_001 | MEDIUM | Is it possible to switch my field of study? | academic_advising | 8;160 (8) | `MULTI-ACCEPTABLE` 2 rows |
| CH_005 | MEDIUM | How do I contest the mark I received? | academic_policies | 17;188 (17) | `MULTI-ACCEPTABLE` 2 rows |
| CH_007 | MEDIUM | Where do I flag someone copying another person's work? | academic_policies | 57;130 (130) | `MULTI-ACCEPTABLE` 2 rows |
| CH_014 | MEDIUM | What are the opening hours of the library on Saturday and Sunday? | campus_facilities | 1 (1) | `COMPETING-ROW` 121 (0.40 vs acceptable 0.25) |
| CH_026 | MEDIUM | Is there a place to stay urgently if my apartment becomes unlivable? | housing | 32;196 (32) | `MULTI-ACCEPTABLE` 2 rows |
| CH_027 | MEDIUM | Who can I talk to if exams are stressing me out? | mental_health | 7;23 (23) | `MULTI-ACCEPTABLE` 2 rows |
| CH_031 | MEDIUM | Could the grading of my exam be checked a second time, and if so, how? | academic_policies | 17;188 (188) | `MULTI-ACCEPTABLE` 2 rows |
| CH_032 | MEDIUM | Should a student witness cheating, to whom ought it be reported? | academic_policies | 57;130 (57) | `MULTI-ACCEPTABLE` 2 rows |
| CH_040 | MEDIUM | At what times does the bus that goes around campus run? | campus_facilities | 139 (139) | `COMPETING-ROW` 58 (0.25 vs acceptable 0.22), 36 (0.25 vs acceptable 0.22) |
| CH_049 | MEDIUM | If an instructor or classmates make the classroom feel threatening, where is that reported? | safety | 102;151 (151) | `MULTI-ACCEPTABLE` 2 rows |
| CH_051 | MEDIUM | finals are freaking me out lol | mental_health | 7;23 (7) | `MULTI-ACCEPTABLE` 2 rows |
| CH_052 | MEDIUM | can i skip the prereq for a class?? | academic_advising | 64 (64) | `COMPETING-ROW` 20 (0.33 vs acceptable 0.00) |
| CH_060 | MEDIUM | any plant based food at the canteen? | campus_facilities | 19 (19) | `COMPETING-ROW` 127 (0.22 vs acceptable 0.00) |
| CH_065 | MEDIUM | missed the signup window for classes, am i screwed | course_registration | 47 (47) | `COMPETING-ROW` 156 (0.25 vs acceptable 0.00), 153 (0.25 vs acceptable 0.00), 71 (0.25 vs acceptable 0.00) |
| CH_069 | MEDIUM | can i use the lab stuff on weekends | research_support | 73 (73) | `COMPETING-ROW` 167 (0.29 vs acceptable 0.20), 197 (0.25 vs acceptable 0.20), 147 (0.25 vs acceptable 0.20) |
| CH_070 | MEDIUM | my friend just collapsed on campus what do i do | safety | 163 (163) | `COMPETING-ROW` 29 (0.29 vs acceptable 0.25) |
| CH_029 | LOW | Would it be possible for me to have a conversation with an academic advisor at this moment? | academic_advising | 52 (52) | `P*-BOUNDARY` max_dice_to_acceptable 0.50 |
| CH_062 | LOW | left my water bottle somewhere on campus, where do lost things go | campus_facilities | 58 (58) | `P*-BOUNDARY` max_dice_to_acceptable 0.50 |

Flag counts: `MULTI-ACCEPTABLE` 9, `COMPETING-ROW` 7, `P*-BOUNDARY` 2.
Unflagged positives (40): CH_002, CH_003, CH_008, CH_009, CH_010, CH_012, CH_013, CH_015, CH_016, CH_018, CH_020, CH_021, CH_022, CH_023, CH_025, CH_028, CH_030, CH_033, CH_034, CH_035, CH_036, CH_038, CH_039, CH_042, CH_043, CH_045, CH_047, CH_048, CH_050, CH_053, CH_054, CH_055, CH_057, CH_058, CH_063, CH_066, CH_067, CH_068, CH_071, CH_072.

## 4. Per-query review cards (all 58 positives)

Each card shows the proposed labels with the acceptable rows' question and response inline. The
complete evidence the spec requires (every training row of the target intent) is in Appendix A.

### CH_001 · synonym_substitution · `academic_advising`
- **Query:** Is it possible to switch my field of study?
- **Labels:** acceptable `8;160` · preferred `8` · seed row `8` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** major change forms in advising
  - **8** [academic_advising] ← preferred: _Can I change my major?_ → Yes — you can request a major change. Visit Academic Advising > Change of Major and submit the form. I can open the form or provide the advisor's contact details.
  - **160** [academic_advising]: _How to update my major in the system?_ → Submit Major Change form via Advising or meet with your advisor for approval. I can provide the form link and next steps.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_002 · synonym_substitution · `academic_advising`
- **Query:** How do I add a secondary specialization to my degree?
- **Labels:** acceptable `22` · preferred `22` · seed row `22` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** minor declaration
  - **22** [academic_advising] ← preferred: _How do I declare a minor?_ → To declare a minor, submit the Minor Declaration form through the Registrar. I can provide the form link or list available minors for your program.
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_003 · synonym_substitution · `academic_advising`
- **Query:** What is the total number of units required to finish my degree?
- **Labels:** acceptable `25` · preferred `25` · seed row `25` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** graduation credit requirements
  - **25** [academic_advising] ← preferred: _How many credits do I need to graduate?_ → Graduation credit requirements depend on your program. Check your degree audit in the student portal or tell me your major and I can give the requirement.
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_005 · synonym_substitution · `academic_policies`
- **Query:** How do I contest the mark I received?
- **Labels:** acceptable `17;188` · preferred `17` · seed row `17` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** grade appeal and recount options
  - **17** [academic_policies] ← preferred: _How can I appeal a grade?_ → Grade appeals follow the Academic Appeals Policy. You'll need to submit a written appeal within the stated deadline. I can link the policy and the appeal form.
  - **188** [academic_policies]: _How to request a grade recount?_ → Request recounts per course policy by contacting the instructor or department; include reasons and supporting evidence. I can draft the recount request.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `academic_policies`:** Appendix A → `academic_policies` (7 rows).

### CH_007 · synonym_substitution · `academic_policies`
- **Query:** Where do I flag someone copying another person's work?
- **Labels:** acceptable `57;130` · preferred `130` · seed row `130` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** academic integrity/plagiarism reporting
  - **57** [academic_policies]: _How to report academic dishonesty?_ → Report allegations to Academic Integrity Office with evidence. They will investigate according to the policy. Want the report form?
  - **130** [academic_policies] ← preferred: _How do I report plagiarism?_ → Report plagiarism to the Academic Integrity Office with supporting evidence. I can guide you through the submission process.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `academic_policies`:** Appendix A → `academic_policies` (7 rows).

### CH_008 · synonym_substitution · `accessibility`
- **Query:** What adjustments can I receive for tests because of a disability?
- **Labels:** acceptable `110` · preferred `110` · seed row `110` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** exam disability accommodation
  - **110** [accessibility] ← preferred: _Are there disability accommodations for exams?_ → Yes — accommodations like extended time are available with approval. Submit an Exam Accommodation Request through Accessibility Services.
- **Other rows of `accessibility`:** Appendix A → `accessibility` (5 rows).

### CH_009 · synonym_substitution · `accessibility`
- **Query:** Can I bring my guide dog to class?
- **Labels:** acceptable `194` · preferred `194` · seed row `194` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** service animal accommodation
  - **194** [accessibility] ← preferred: _Can I request service animal accommodation?_ → Submit a Service Animal Request to Accessibility Services with documentation. I can provide the necessary forms and policy info.
- **Other rows of `accessibility`:** Appendix A → `accessibility` (5 rows).

### CH_010 · synonym_substitution · `administration`
- **Query:** How do I order a copy of my academic record?
- **Labels:** acceptable `11` · preferred `11` · seed row `11` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** official transcript request
  - **11** [administration] ← preferred: _How do I apply for a transcript?_ → Official transcripts can be requested through the Registrar's Office. You can request electronic or mailed copies. Should I provide the transcript request link?
- **Other rows of `administration`:** Appendix A → `administration` (15 rows).

### CH_012 · synonym_substitution · `technical_issue`
- **Query:** I can't remember the login for my student account.
- **Labels:** acceptable `2` · preferred `2` · seed row `2` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** portal password reset
  - **2** [technical_issue] ← preferred: _I forgot my student portal password._ → I can help reset your password. Please visit the portal's Password Reset page or confirm you'd like me to send password reset instructions to your registered email.
- **Other rows of `technical_issue`:** Appendix A → `technical_issue` (6 rows).

### CH_013 · synonym_substitution · `administration`
- **Query:** Where do I collect my campus identity card?
- **Labels:** acceptable `181` · preferred `181` · seed row `181` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** student ID pickup at Card Services
  - **181** [administration] ← preferred: _Where do I pick up my student ID?_ → Pick up IDs at Card Services during office hours; bring photo ID. I can confirm pickup times.
- **Other rows of `administration`:** Appendix A → `administration` (15 rows).

### CH_014 · synonym_substitution · `campus_facilities`
- **Query:** What are the opening hours of the library on Saturday and Sunday?
- **Labels:** acceptable `1` · preferred `1` · seed row `1` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** library weekend hours
  - **1** [campus_facilities] ← preferred: _When is the library open on weekends?_ → The library hours are 9:00–18:00 on Saturdays and 10:00–16:00 on Sundays. Would you like directions or the study room booking link?
- **Flags:** `COMPETING-ROW` 121 (0.40 vs acceptable 0.25)
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_015 · synonym_substitution · `campus_facilities`
- **Query:** How do I fight a fine for my car?
- **Labels:** acceptable `51` · preferred `51` · seed row `51` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** parking ticket dispute portal
  - **51** [campus_facilities] ← preferred: _How can I dispute a parking ticket?_ → You can appeal a parking ticket via the Parking Appeals Portal. Collect evidence and submit within the appeal window. Want the appeal link?
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_016 · synonym_substitution · `student_services`
- **Query:** Is there a free attorney service for learners?
- **Labels:** acceptable `70` · preferred `70` · seed row `70` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** student legal clinic consultations
  - **70** [student_services] ← preferred: _Do we have free legal advice for students?_ → The Student Legal Clinic offers free consultations on certain matters. I can provide their schedule and contact info.
- **Other rows of `student_services`:** Appendix A → `student_services` (16 rows).

### CH_018 · synonym_substitution · `career_services`
- **Query:** How can I get feedback on my resume?
- **Labels:** acceptable `90` · preferred `90` · seed row `90` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** CV review by Career Services
  - **90** [career_services] ← preferred: _Can I get help with my CV?_ → Career Services offers CV reviews and workshops. I can book a review session or provide CV tips now if you'd like.
- **Other rows of `career_services`:** Appendix A → `career_services` (9 rows).

### CH_020 · synonym_substitution · `course_registration`
- **Query:** How do I sign up for courses during the summer break?
- **Labels:** acceptable `71` · preferred `71` · seed row `71` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** summer class enrollment
  - **71** [course_registration] ← preferred: _How do I enroll in summer classes?_ → Summer enrollment opens on the Registration page. Search for summer term courses and add them to your cart; let me know which course codes you want.
- **Other rows of `course_registration`:** Appendix A → `course_registration` (9 rows).

### CH_021 · synonym_substitution · `course_registration`
- **Query:** Can I change to a different lab group?
- **Labels:** acceptable `80` · preferred `80` · seed row `80` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** swap lab sections
  - **80** [course_registration] ← preferred: _Can I swap lab sections?_ → Swapping lab sections depends on seat availability. Use the Swap option in Registration or request instructor permission; shall I check availability?
- **Other rows of `course_registration`:** Appendix A → `course_registration` (9 rows).

### CH_022 · synonym_substitution · `financial_aid`
- **Query:** Is financial help offered to overseas students?
- **Labels:** acceptable `3` · preferred `3` · seed row `3` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** international student scholarships
  - **3** [financial_aid] ← preferred: _Are there scholarships for international students?_ → Yes — we have several scholarships for international students. Check the Financial Aid page and filter by 'International'. Would you like me to list currently open scholarships?
- **Other rows of `financial_aid`:** Appendix A → `financial_aid` (11 rows).

### CH_023 · synonym_substitution · `financial_aid`
- **Query:** Will I get my money back for a course I withdrew from?
- **Labels:** acceptable `33` · preferred `33` · seed row `33` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** tuition refund for dropped class
  - **33** [financial_aid] ← preferred: _Can I get a refund for a dropped class?_ → Refund eligibility depends on the drop date and tuition policy. The Refunds page has the schedule. Want me to calculate a likely refund date based on drop date?
- **Other rows of `financial_aid`:** Appendix A → `financial_aid` (11 rows).

### CH_025 · synonym_substitution · `safety`
- **Query:** Where can I report being bullied by a classmate?
- **Labels:** acceptable `102` · preferred `102` · seed row `102` · max_dice_to_acceptable `0.400000` · bucket `B2` · structural: valid
- **Notes rationale:** harassment/bullying reporting
  - **102** [safety] ← preferred: _Where do I report harassment?_ → Report harassment to Title IX Office or Campus Safety immediately. I can provide contact details and help you file a confidential report.
- **Other rows of `safety`:** Appendix A → `safety` (6 rows).

### CH_026 · synonym_substitution · `housing`
- **Query:** Is there a place to stay urgently if my apartment becomes unlivable?
- **Labels:** acceptable `32;196` · preferred `32` · seed row `32` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** emergency displacement housing
  - **32** [housing] ← preferred: _Is there emergency housing in case of displacement?_ → Emergency housing is arranged by Student Affairs depending on availability. Contact Student Affairs immediately; I can provide contact info.
  - **196** [housing]: _How do I apply for emergency housing after a natural disaster?_ → Contact Student Affairs and Housing immediately; emergency protocols and temporary housing are activated as needed. I can provide emergency contacts.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `housing`:** Appendix A → `housing` (5 rows).

### CH_027 · synonym_substitution · `mental_health`
- **Query:** Who can I talk to if exams are stressing me out?
- **Labels:** acceptable `7;23` · preferred `23` · seed row `23` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** finals stress counseling
  - **7** [mental_health]: _I'm feeling anxious about finals._ → I'm sorry you're feeling anxious. The counseling center offers drop-in hours and appointments. Would you like to book a session or see immediate coping resources?
  - **23** [mental_health] ← preferred: _Is there counseling available for stress?_ → Yes — counseling services offer individual therapy and workshops. I can schedule an appointment or share immediate coping resources.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `mental_health`:** Appendix A → `mental_health` (4 rows).

### CH_028 · synonym_substitution · `research_support`
- **Query:** How do I obtain grants for my research project?
- **Labels:** acceptable `145` · preferred `145` · seed row `145` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** research funding application
  - **145** [research_support] ← preferred: _How do I apply for research funding?_ → Research funding opportunities are posted by departments and the Office of Research. I can compile active funding calls for your field.
- **Other rows of `research_support`:** Appendix A → `research_support` (6 rows).

### CH_029 · syntactic_inversion · `academic_advising`
- **Query:** Would it be possible for me to have a conversation with an academic advisor at this moment?
- **Labels:** acceptable `52` · preferred `52` · seed row `52` · max_dice_to_acceptable `0.500000` · bucket `B3` · structural: valid
- **Notes rationale:** speak to academic advisor
  - **52** [academic_advising] ← preferred: _Can I speak to an academic advisor now?_ → Advisors have scheduled hours; I can show next available slots or submit a request for an urgent appointment.
- **Flags:** `P*-BOUNDARY` max_dice_to_acceptable 0.50
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_030 · syntactic_inversion · `academic_advising`
- **Query:** Regarding credits earned at another institution, what is the procedure for getting them recognised?
- **Labels:** acceptable `103` · preferred `103` · seed row `103` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** transfer credit petition
  - **103** [academic_advising] ← preferred: _How do I petition for credit transfer?_ → Submit a Transfer Credit Petition with course syllabi for evaluation. I can provide the petition form and guidelines.
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_031 · syntactic_inversion · `academic_policies`
- **Query:** Could the grading of my exam be checked a second time, and if so, how?
- **Labels:** acceptable `17;188` · preferred `188` · seed row `188` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** exam grade recount and appeal
  - **17** [academic_policies]: _How can I appeal a grade?_ → Grade appeals follow the Academic Appeals Policy. You'll need to submit a written appeal within the stated deadline. I can link the policy and the appeal form.
  - **188** [academic_policies] ← preferred: _How to request a grade recount?_ → Request recounts per course policy by contacting the instructor or department; include reasons and supporting evidence. I can draft the recount request.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `academic_policies`:** Appendix A → `academic_policies` (7 rows).

### CH_032 · syntactic_inversion · `academic_policies`
- **Query:** Should a student witness cheating, to whom ought it be reported?
- **Labels:** acceptable `57;130` · preferred `57` · seed row `57` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** academic dishonesty reporting
  - **57** [academic_policies] ← preferred: _How to report academic dishonesty?_ → Report allegations to Academic Integrity Office with evidence. They will investigate according to the policy. Want the report form?
  - **130** [academic_policies]: _How do I report plagiarism?_ → Report plagiarism to the Academic Integrity Office with supporting evidence. I can guide you through the submission process.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `academic_policies`:** Appendix A → `academic_policies` (7 rows).

### CH_033 · syntactic_inversion · `academic_support`
- **Query:** From whom, and by what process, might a letter of reference be obtained?
- **Labels:** acceptable `143` · preferred `143` · seed row `143` · max_dice_to_acceptable `0.333333` · bucket `B2` · structural: valid
- **Notes rationale:** recommendation letter request
  - **143** [academic_support] ← preferred: _How to get a recommendation letter?_ → Ask professors early and provide your CV and context. I can draft a polite request email template for you to customize.
- **Other rows of `academic_support`:** Appendix A → `academic_support` (7 rows).

### CH_034 · syntactic_inversion · `academic_support`
- **Query:** Is assistance offered to students who are writing research papers in the sciences?
- **Labels:** acceptable `159` · preferred `159` · seed row `159` · max_dice_to_acceptable `0.200000` · bucket `B1` · structural: valid
- **Notes rationale:** writing center scientific writing
  - **159** [academic_support] ← preferred: _Is there help for scientific writing?_ → The Writing Center and Research Support offer help with scientific writing. I can schedule a consultation or provide style guides.
- **Other rows of `academic_support`:** Appendix A → `academic_support` (7 rows).

### CH_035 · syntactic_inversion · `accessibility`
- **Query:** Could you kindly explain the steps required to arrange support for a disability?
- **Labels:** acceptable `40` · preferred `40` · seed row `40` · max_dice_to_acceptable `0.200000` · bucket `B1` · structural: valid
- **Notes rationale:** accommodation request form
  - **40** [accessibility] ← preferred: _How to request disability accommodations?_ → Submit the Accommodation Request form to Accessibility Services and provide documentation. I can provide the form and list required documents.
- **Other rows of `accessibility`:** Appendix A → `accessibility` (5 rows).

### CH_036 · syntactic_inversion · `administration`
- **Query:** My personal details have changed; where should they be updated?
- **Labels:** acceptable `62` · preferred `62` · seed row `62` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** update personal profile
  - **62** [administration] ← preferred: _How do I update my personal information?_ → Update personal info in the Student Portal under 'Profile'. I can guide you to the exact fields to update.
- **Other rows of `administration`:** Appendix A → `administration` (15 rows).

### CH_038 · syntactic_inversion · `administration`
- **Query:** Before I finish my final year, what must be done so that I can graduate?
- **Labels:** acceptable `107` · preferred `107` · seed row `107` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** graduation application checklist
  - **107** [administration] ← preferred: _What are the steps to apply for graduation?_ → Apply for graduation via the Registrar by the posted deadline. I can check your eligibility and submit the application link.
- **Other rows of `administration`:** Appendix A → `administration` (15 rows).

### CH_039 · syntactic_inversion · `campus_facilities`
- **Query:** With regard to parking, what rules apply to students who drive to campus?
- **Labels:** acceptable `12` · preferred `12` · seed row `12` · max_dice_to_acceptable `0.400000` · bucket `B2` · structural: valid
- **Notes rationale:** student parking permit policy
  - **12** [campus_facilities] ← preferred: _What's the parking policy for students?_ → Student parking permits are required Monday–Friday. Permits can be purchased online via Campus Parking. Want the parking permit rates?
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_040 · syntactic_inversion · `campus_facilities`
- **Query:** At what times does the bus that goes around campus run?
- **Labels:** acceptable `139` · preferred `139` · seed row `139` · max_dice_to_acceptable `0.222222` · bucket `B1` · structural: valid
- **Notes rationale:** campus shuttle bus schedule
  - **139** [campus_facilities] ← preferred: _Is there a campus shuttle schedule?_ → Yes — the shuttle schedule and routes are on the Transport Services page. I can show next shuttle times for your stop.
- **Flags:** `COMPETING-ROW` 58 (0.25 vs acceptable 0.22), 36 (0.25 vs acceptable 0.22)
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_042 · syntactic_inversion · `career_services`
- **Query:** If I wanted a part-time position at the university itself, how would I go about it?
- **Labels:** acceptable `117` · preferred `117` · seed row `117` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** on-campus student jobs
  - **117** [career_services] ← preferred: _How to apply for on-campus jobs?_ → Search on-campus job listings via the Career Portal and submit your application online. Would you like help building your application?
- **Other rows of `career_services`:** Appendix A → `career_services` (9 rows).

### CH_043 · syntactic_inversion · `course_registration`
- **Query:** When a class has no seats left, is there some way of being placed on a list?
- **Labels:** acceptable `173` · preferred `173` · seed row `173` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** course waitlist procedure
  - **173** [course_registration] ← preferred: _How to get into a full class waitlist?_ → Join the waitlist in Registration; some departments allow instructor overrides. I can check waitlist position if you're already queued.
- **Other rows of `course_registration`:** Appendix A → `course_registration` (9 rows).

### CH_045 · syntactic_inversion · `financial_aid`
- **Query:** If a student is facing a sudden financial crisis, is any money available to help?
- **Labels:** acceptable `72` · preferred `72` · seed row `72` · max_dice_to_acceptable `0.333333` · bucket `B2` · structural: valid
- **Notes rationale:** student emergency assistance fund
  - **72** [financial_aid] ← preferred: _Are there student emergency funds available?_ → Yes, the Student Emergency Fund provides short-term assistance. Apply via Student Affairs; I can link the application.
- **Other rows of `financial_aid`:** Appendix A → `financial_aid` (11 rows).

### CH_047 · syntactic_inversion · `housing`
- **Query:** By which date must a request to move to a different room be submitted?
- **Labels:** acceptable `89` · preferred `89` · seed row `89` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** room reassignment deadline
  - **89** [housing] ← preferred: _What's the deadline to change housing assignments?_ → Housing reassignment deadlines are posted on the Housing Portal. Early applications have priority. Would you like the current deadlines?
- **Other rows of `housing`:** Appendix A → `housing` (5 rows).

### CH_048 · syntactic_inversion · `mental_health`
- **Query:** Will what I share with a therapist be kept private?
- **Labels:** acceptable `85` · preferred `85` · seed row `85` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** therapy confidentiality
  - **85** [mental_health] ← preferred: _Is mental health counseling confidential?_ → Yes, counseling services maintain confidentiality within legal limits. I can share privacy policies or help schedule an appointment.
- **Other rows of `mental_health`:** Appendix A → `mental_health` (4 rows).

### CH_049 · syntactic_inversion · `safety`
- **Query:** If an instructor or classmates make the classroom feel threatening, where is that reported?
- **Labels:** acceptable `102;151` · preferred `151` · seed row `151` · max_dice_to_acceptable `0.181818` · bucket `B1` · structural: valid
- **Notes rationale:** hostile classroom environment reporting
  - **102** [safety]: _Where do I report harassment?_ → Report harassment to Title IX Office or Campus Safety immediately. I can provide contact details and help you file a confidential report.
  - **151** [safety] ← preferred: _How do I report a hostile classroom environment?_ → Report incidents to the Office of Student Conduct and Title IX as appropriate. I can help gather details and submit a formal report.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `safety`:** Appendix A → `safety` (6 rows).

### CH_050 · syntactic_inversion · `technical_issue`
- **Query:** Should my two-step login stop working, how is it reset?
- **Labels:** acceptable `46` · preferred `46` · seed row `46` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** 2FA reset ticket
  - **46** [technical_issue] ← preferred: _How do I reset two-factor authentication?_ → Resetting 2FA requires contacting IT or using backup codes. I can guide you through the reset steps or create a support request.
- **Other rows of `technical_issue`:** Appendix A → `technical_issue` (6 rows).

### CH_051 · slang_informal · `mental_health`
- **Query:** finals are freaking me out lol
- **Labels:** acceptable `7;23` · preferred `7` · seed row `7` · max_dice_to_acceptable `0.333333` · bucket `B2` · structural: valid
- **Notes rationale:** anxiety about finals
  - **7** [mental_health] ← preferred: _I'm feeling anxious about finals._ → I'm sorry you're feeling anxious. The counseling center offers drop-in hours and appointments. Would you like to book a session or see immediate coping resources?
  - **23** [mental_health]: _Is there counseling available for stress?_ → Yes — counseling services offer individual therapy and workshops. I can schedule an appointment or share immediate coping resources.
- **Flags:** `MULTI-ACCEPTABLE` 2 rows
- **Other rows of `mental_health`:** Appendix A → `mental_health` (4 rows).

### CH_052 · slang_informal · `academic_advising`
- **Query:** can i skip the prereq for a class??
- **Labels:** acceptable `64` · preferred `64` · seed row `64` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** prerequisite waiver request
  - **64** [academic_advising] ← preferred: _Can I get a waiver for course prerequisites?_ → Prerequisite waivers require instructor approval and justification. I can prepare a waiver request template for you to submit.
- **Flags:** `COMPETING-ROW` 20 (0.33 vs acceptable 0.00)
- **Other rows of `academic_advising`:** Appendix A → `academic_advising` (9 rows).

### CH_053 · slang_informal · `academic_policies`
- **Query:** wanna ditch a class without it messing up my record
- **Labels:** acceptable `114` · preferred `114` · seed row `114` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** drop course without penalty
  - **114** [academic_policies] ← preferred: _How do I drop a course without penalty?_ → Dropping without penalty requires action before the tuition refund deadline. Check the academic calendar for exact dates or tell me the course and date.
- **Other rows of `academic_policies`:** Appendix A → `academic_policies` (7 rows).

### CH_054 · slang_informal · `academic_support`
- **Query:** how do i cite stuff in my thesis, apa or whatever
- **Labels:** acceptable `27` · preferred `27` · seed row `27` · max_dice_to_acceptable `0.222222` · bucket `B1` · structural: valid
- **Notes rationale:** library citation and style guides
  - **27** [academic_support] ← preferred: _I need help with referencing styles for my thesis._ → I can guide you on APA, MLA, or Chicago styles and provide citation examples. Which style are you using?
- **Other rows of `academic_support`:** Appendix A → `academic_support` (7 rows).

### CH_055 · slang_informal · `academic_support`
- **Query:** need a tutor asap
- **Labels:** acceptable `162` · preferred `162` · seed row `162` · max_dice_to_acceptable `0.333333` · bucket `B2` · structural: valid
- **Notes rationale:** book tutoring session
  - **162** [academic_support] ← preferred: _Can the bot connect me to a tutor now?_ → I can check tutor availability and connect you. Which subject do you need help with?
- **Other rows of `academic_support`:** Appendix A → `academic_support` (7 rows).

### CH_057 · slang_informal · `accessibility`
- **Query:** pregnant and worried about keeping up w classes, any support?
- **Labels:** acceptable `140` · preferred `140` · seed row `140` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** academic accommodations for pregnancy
  - **140** [accessibility] ← preferred: _Can I get academic accommodations for pregnancy?_ → Accommodations are available; contact Accessibility Services for documentation and support. I can help initiate the request.
- **Other rows of `accessibility`:** Appendix A → `accessibility` (5 rows).

### CH_058 · slang_informal · `administration`
- **Query:** lost my uni card, how do i get a new one
- **Labels:** acceptable `88` · preferred `88` · seed row `88` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** replacement ID card
  - **88** [administration] ← preferred: _Can I get an official ID replacement?_ → ID replacements are issued at Card Services for a fee. Bring photo ID or request a replacement online. Want the replacement fee details?
- **Other rows of `administration`:** Appendix A → `administration` (15 rows).

### CH_060 · slang_informal · `campus_facilities`
- **Query:** any plant based food at the canteen?
- **Labels:** acceptable `19` · preferred `19` · seed row `19` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** dining hall vegan options
  - **19** [campus_facilities] ← preferred: _Do we have a vegan option in the cafeteria?_ → Yes — the cafeteria offers multiple vegan options daily. I can show today's menu if you'd like.
- **Flags:** `COMPETING-ROW` 127 (0.22 vs acceptable 0.00)
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_062 · slang_informal · `campus_facilities`
- **Query:** left my water bottle somewhere on campus, where do lost things go
- **Labels:** acceptable `58` · preferred `58` · seed row `58` · max_dice_to_acceptable `0.500000` · bucket `B3` · structural: valid
- **Notes rationale:** lost and found at security
  - **58** [campus_facilities] ← preferred: _Is there a lost and found on campus?_ → Yes — Lost & Found is at the Campus Police office. You can check items online or visit their office during business hours.
- **Flags:** `P*-BOUNDARY` max_dice_to_acceptable 0.50
- **Other rows of `campus_facilities`:** Appendix A → `campus_facilities` (16 rows).

### CH_063 · slang_informal · `career_services`
- **Query:** wanna chat w a career person this week
- **Labels:** acceptable `166` · preferred `166` · seed row `166` · max_dice_to_acceptable `0.444444` · bucket `B2` · structural: valid
- **Notes rationale:** career counselor appointment
  - **166** [career_services] ← preferred: _Can I get an appointment with the career counselor this week?_ → Career Counselors have limited slots; I can show available times and book an appointment this week if there's availability.
- **Other rows of `career_services`:** Appendix A → `career_services` (9 rows).

### CH_065 · slang_informal · `course_registration`
- **Query:** missed the signup window for classes, am i screwed
- **Labels:** acceptable `47` · preferred `47` · seed row `47` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** late registration policy
  - **47** [course_registration] ← preferred: _Is late registration allowed?_ → Late registration policies depend on the semester and course. There may be late fees; check the Registration Deadlines page. Want the deadline for your course?
- **Flags:** `COMPETING-ROW` 156 (0.25 vs acceptable 0.00), 153 (0.25 vs acceptable 0.00), 71 (0.25 vs acceptable 0.00)
- **Other rows of `course_registration`:** Appendix A → `course_registration` (9 rows).

### CH_066 · slang_informal · `financial_aid`
- **Query:** my aid amount looks totally off
- **Labels:** acceptable `35` · preferred `35` · seed row `35` · max_dice_to_acceptable `0.285714` · bucket `B2` · structural: valid
- **Notes rationale:** financial aid package review
  - **35** [financial_aid] ← preferred: _My financial aid package seems incorrect._ → I can help review it. Please provide the specific discrepancy, or I can create a support ticket to Financial Aid on your behalf.
- **Other rows of `financial_aid`:** Appendix A → `financial_aid` (11 rows).

### CH_067 · slang_informal · `housing`
- **Query:** my roomie keeps breaking the dorm rules, who do i tell
- **Labels:** acceptable `49` · preferred `49` · seed row `49` · max_dice_to_acceptable `0.181818` · bucket `B1` · structural: valid
- **Notes rationale:** roommate rule violations
  - **49** [housing] ← preferred: _I think my roommate is violating housing rules._ → Report housing violations to Residential Life with details and dates. I can provide the reporting form or connect you to a housing coordinator.
- **Other rows of `housing`:** Appendix A → `housing` (5 rows).

### CH_068 · slang_informal · `mental_health`
- **Query:** need someone to talk to at 2am, is anyone there
- **Labels:** acceptable `61` · preferred `61` · seed row `61` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** 24/7 crisis counseling line
  - **61** [mental_health] ← preferred: _Do we have emergency counseling after hours?_ → After-hours crisis support is available via the 24/7 helpline. If you are in immediate danger, call emergency services. Want the helpline number?
- **Other rows of `mental_health`:** Appendix A → `mental_health` (4 rows).

### CH_069 · slang_informal · `research_support`
- **Query:** can i use the lab stuff on weekends
- **Labels:** acceptable `73` · preferred `73` · seed row `73` · max_dice_to_acceptable `0.200000` · bucket `B1` · structural: valid
- **Notes rationale:** after-hours lab access
  - **73** [research_support] ← preferred: _Can I get access to lab equipment outside class hours?_ → Access outside hours requires supervisor approval and training. I can help request extended access if you provide supervisor details.
- **Flags:** `COMPETING-ROW` 167 (0.29 vs acceptable 0.20), 197 (0.25 vs acceptable 0.20), 147 (0.25 vs acceptable 0.20)
- **Other rows of `research_support`:** Appendix A → `research_support` (6 rows).

### CH_070 · slang_informal · `safety`
- **Query:** my friend just collapsed on campus what do i do
- **Labels:** acceptable `163` · preferred `163` · seed row `163` · max_dice_to_acceptable `0.250000` · bucket `B2` · structural: valid
- **Notes rationale:** campus emergency medical response
  - **163** [safety] ← preferred: _How to get emergency medical help on campus?_ → For medical emergencies call campus police or local emergency services immediately. For non-emergency health issues contact Student Health Services.
- **Flags:** `COMPETING-ROW` 29 (0.29 vs acceptable 0.25)
- **Other rows of `safety`:** Appendix A → `safety` (6 rows).

### CH_071 · slang_informal · `technical_issue`
- **Query:** my uni inbox isnt getting any emails
- **Labels:** acceptable `42` · preferred `42` · seed row `42` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** student email delivery failure
  - **42** [technical_issue] ← preferred: _My student email is not receiving messages._ → Check your storage quota and spam folder. If the problem continues, reset your mail settings or submit a ticket; I can initiate that for you.
- **Other rows of `technical_issue`:** Appendix A → `technical_issue` (6 rows).

### CH_072 · slang_informal · `student_services`
- **Query:** kinda broke and cant afford groceries, any help?
- **Labels:** acceptable `168` · preferred `168` · seed row `168` · max_dice_to_acceptable `0.000000` · bucket `B1` · structural: valid
- **Notes rationale:** campus food pantry and vouchers
  - **168** [student_services] ← preferred: _Is there support for food insecurity?_ → The campus food pantry and emergency meal vouchers assist students facing food insecurity. I can provide hours and application details.
- **Other rows of `student_services`:** Appendix A → `student_services` (16 rows).

## 5. Negative probes (Category 4)

All 14 negatives have empty `acceptable_row_ids` and `preferred_row_id`: **True**. Answerability
evidence is the prior all-160-response audit; target intents follow the query's own topic, not the lure.

| ID | Query | target_intent | Lure row (intent) | max_dice_to_any_kb_row | Notes |
|---|---|---|---|---|---|
| CH_073 | Is late check-in allowed when moving into the dorms? | housing | 47 (course_registration) | 0.500000 | drafted from row 47; shares late/allowed, asks about dorm move-in check-in; unanswerable |
| CH_075 | Is the computer lab open late during exam week? | campus_facilities | 41 (campus_facilities) | 0.400000 | drafted from row 41; shares open-late, asks about computer lab; unanswerable |
| CH_078 | Is there a safety escort for students walking to parking at night? | safety | 12 (campus_facilities) | 0.444444 | drafted from row 12; shares parking/students, asks about a night safety escort; unanswerable |
| CH_079 | Is there a policy on using AI tools in assignments? | academic_policies | 190 (student_life) | 0.250000 | drafted from row 190; shares policy, asks about AI use in coursework; unanswerable |
| CH_080 | Is Microsoft Office free to use for students? | technical_issue | 81 (campus_facilities) | 0.500000 | drafted from row 81; shares free to use, asks about software licences; unanswerable |
| CH_081 | Do scholarships cover the cost of on-campus housing? | financial_aid | 10 (housing) | 0.444444 | drafted from row 10; shares on-campus housing, asks what scholarships cover; unanswerable |
| CH_082 | Are there housing options for visiting faculty next semester? | housing | 10 (housing) | 0.444444 | drafted from row 10; shares housing next semester, asks about visiting faculty; unanswerable |
| CH_083 | Can I apply research methods credits toward my minor? | academic_advising | 145 (research_support) | 0.500000 | drafted from row 145; shares research/minor, asks about credit applicability; unanswerable |
| CH_084 | How many free counseling sessions do students get per semester? | mental_health | 179 (student_services) | 0.444444 | drafted from row 179; shares free/students, asks about counseling session limits; unanswerable |
| CH_085 | Do I need approval to survey other students for my research project? | research_support | 154 (student_services) | 0.222222 | drafted from row 145; shares research, asks about research ethics approval; unanswerable |
| CH_086 | Where do I pick up my graduation gown? | administration | 181 (administration) | 0.333333 | drafted from row 181; shares pick up, asks about graduation gown; unanswerable |
| CH_088 | Is there a course on campus about running a daycare? | course_information | 36 (campus_facilities) | 0.666667 | drafted from row 36; shares daycare/campus, asks about a course on running a daycare; unanswerable |
| CH_089 | Will a dropped class show up on my transcript? | academic_policies | 33 (financial_aid) | 0.666667 | drafted from row 33; shares dropped class, asks about transcript notation; unanswerable |
| CH_090 | How do I report a bug infestation in my dorm room? | housing | 191 (technical_issue) | 0.444444 | drafted from row 191; shares report bug, asks about dorm pest infestation; unanswerable |

Open items for the human reviewer: confirm row 114 does not answer CH_089 (it covers penalty-free
dropping, not transcript notation) and row 49 does not answer CH_090 (rule violations, not
maintenance). Non-blocking: CH_079 and CH_085 have weak lures (0.250 / 0.222); CH_075, CH_082 and
CH_086 have same-intent lures, so `WrongIntent@1` cannot register lure capture for them.

## 6. Composition and validation

- Rows 72; categories {'distractor_negative': 14, 'slang_informal': 18, 'synonym_substitution': 22, 'syntactic_inversion': 18}; P 58; N 14.
- |P*| = **58** (>= 40 required); slices {'B1': 36, 'B2': 20, 'B3a': 2, 'NEG': 14}.
- Schema: PASS. Dice fields vs fresh annotation: PASS. Contamination: PASS.
- Composition: ["72 rows are not verified: ['CH_001', 'CH_002', 'CH_003', 'CH_005', 'CH_007', 'CH_008', 'CH_009', 'CH_010', 'CH_012', 'CH_013']"]  ← the only expected issue (rows are `draft`).
- Unique challenge ids: True.

## Appendix A. Every training row of each positive target intent

### `academic_advising` (9 rows) · used by CH_001, CH_002, CH_003, CH_029, CH_030, CH_052
- **8** · _Can I change my major?_ → Yes — you can request a major change. Visit Academic Advising > Change of Major and submit the form. I can open the form or provide the advisor's contact details.
- **20** · _Can I audit a class this semester?_ → Auditing is allowed for certain courses with instructor permission. Check the course policy or request permission from the instructor. Want me to find the instructor's email?
- **22** · _How do I declare a minor?_ → To declare a minor, submit the Minor Declaration form through the Registrar. I can provide the form link or list available minors for your program.
- **25** · _How many credits do I need to graduate?_ → Graduation credit requirements depend on your program. Check your degree audit in the student portal or tell me your major and I can give the requirement.
- **52** · _Can I speak to an academic advisor now?_ → Advisors have scheduled hours; I can show next available slots or submit a request for an urgent appointment.
- **64** · _Can I get a waiver for course prerequisites?_ → Prerequisite waivers require instructor approval and justification. I can prepare a waiver request template for you to submit.
- **103** · _How do I petition for credit transfer?_ → Submit a Transfer Credit Petition with course syllabi for evaluation. I can provide the petition form and guidelines.
- **120** · _How to find my advisor's office hours?_ → Advisor office hours are listed on the Advising page and in the faculty directory. Tell me your advisor's name and I will fetch their hours.
- **160** · _How to update my major in the system?_ → Submit Major Change form via Advising or meet with your advisor for approval. I can provide the form link and next steps.

### `academic_policies` (7 rows) · used by CH_005, CH_007, CH_031, CH_032, CH_053
- **14** · _Can I get an extension on my assignment?_ → Extensions depend on the instructor and course policy. I can provide the extension request form and guidance on what details to include.
- **17** · _How can I appeal a grade?_ → Grade appeals follow the Academic Appeals Policy. You'll need to submit a written appeal within the stated deadline. I can link the policy and the appeal form.
- **57** · _How to report academic dishonesty?_ → Report allegations to Academic Integrity Office with evidence. They will investigate according to the policy. Want the report form?
- **114** · _How do I drop a course without penalty?_ → Dropping without penalty requires action before the tuition refund deadline. Check the academic calendar for exact dates or tell me the course and date.
- **130** · _How do I report plagiarism?_ → Report plagiarism to the Academic Integrity Office with supporting evidence. I can guide you through the submission process.
- **169** · _How do I request academic probation removal?_ → Academic probation removal usually requires an appeal and evidence of improved performance. Contact your academic advisor to prepare an appeal.
- **188** · _How to request a grade recount?_ → Request recounts per course policy by contacting the instructor or department; include reasons and supporting evidence. I can draft the recount request.

### `academic_support` (7 rows) · used by CH_033, CH_034, CH_054, CH_055
- **27** · _I need help with referencing styles for my thesis._ → I can guide you on APA, MLA, or Chicago styles and provide citation examples. Which style are you using?
- **54** · _Can you help me draft an email to a professor?_ → Sure — tell me the purpose and key points and I will draft a polite, concise email you can review and send.
- **143** · _How to get a recommendation letter?_ → Ask professors early and provide your CV and context. I can draft a polite request email template for you to customize.
- **155** · _How to submit a capstone project proposal?_ → Submit your proposal to your department's Capstone Committee per the guidelines. I can provide the template and checklist.
- **159** · _Is there help for scientific writing?_ → The Writing Center and Research Support offer help with scientific writing. I can schedule a consultation or provide style guides.
- **162** · _Can the bot connect me to a tutor now?_ → I can check tutor availability and connect you. Which subject do you need help with?
- **189** · _Can I get help with time management?_ → Academic coaching offers time management workshops and one-on-one coaching. I can schedule a coaching session for you.

### `accessibility` (5 rows) · used by CH_008, CH_009, CH_035, CH_057
- **40** · _How to request disability accommodations?_ → Submit the Accommodation Request form to Accessibility Services and provide documentation. I can provide the form and list required documents.
- **110** · _Are there disability accommodations for exams?_ → Yes — accommodations like extended time are available with approval. Submit an Exam Accommodation Request through Accessibility Services.
- **140** · _Can I get academic accommodations for pregnancy?_ → Accommodations are available; contact Accessibility Services for documentation and support. I can help initiate the request.
- **142** · _Are course materials accessible for visually impaired students?_ → Yes — Accessibility Services and the Library provide accessible formats. I can request materials conversion for you.
- **194** · _Can I request service animal accommodation?_ → Submit a Service Animal Request to Accessibility Services with documentation. I can provide the necessary forms and policy info.

### `administration` (15 rows) · used by CH_010, CH_013, CH_036, CH_038, CH_058
- **11** · _How do I apply for a transcript?_ → Official transcripts can be requested through the Registrar's Office. You can request electronic or mailed copies. Should I provide the transcript request link?
- **26** · _Can the bot submit a leave of absence?_ → Leaves of absence require an official form submitted to Student Records. I can provide the form and steps but cannot submit personal documents for you.
- **44** · _What's the procedure for medical leave?_ → Medical leave requires documentation and submission to Student Records. I can provide the Medical Leave form and steps if you want.
- **62** · _How do I update my personal information?_ → Update personal info in the Student Portal under 'Profile'. I can guide you to the exact fields to update.
- **74** · _How to contact the registrar?_ → Registrar can be reached at registrar@university.edu or (555) 010-2000. I can draft an email or find office hours.
- **88** · _Can I get an official ID replacement?_ → ID replacements are issued at Card Services for a fee. Bring photo ID or request a replacement online. Want the replacement fee details?
- **92** · _How to add an emergency contact?_ → Add emergency contacts in the Student Portal under 'Emergency Contacts'. Provide name, relation, and contact number.
- **94** · _Can I request a meeting with the dean?_ → Dean meetings require an appointment request via the Dean's Office. I can draft a request and check available dates for you.
- **97** · _How do I withdraw from the university?_ → Withdrawing requires submitting a Withdrawal Form and meeting with Student Records. I can provide the steps and potential financial implications.
- **107** · _What are the steps to apply for graduation?_ → Apply for graduation via the Registrar by the posted deadline. I can check your eligibility and submit the application link.
- **111** · _Can I defer my enrollment?_ → Deferral policies depend on program; notify Admissions to defer enrollment and check deadlines. Shall I provide the Admissions contact?
- **146** · _Can I change my billing address?_ → Update billing address in the Bursar portal or contact Bursar's Office. I can guide you through the update form.
- **171** · _Can I get a waiver for health insurance?_ → Health insurance waivers require proof of comparable coverage. Check Student Health Insurance page and submit waiver documentation online.
- **181** · _Where do I pick up my student ID?_ → Pick up IDs at Card Services during office hours; bring photo ID. I can confirm pickup times.
- **187** · _What are acceptable ID forms for exams?_ → Acceptable ID usually includes government-issued photo ID or student ID. Confirm specifics with your instructor or testing center.

### `campus_facilities` (16 rows) · used by CH_014, CH_015, CH_039, CH_040, CH_060, CH_062
- **1** · _When is the library open on weekends?_ → The library hours are 9:00–18:00 on Saturdays and 10:00–16:00 on Sundays. Would you like directions or the study room booking link?
- **12** · _What's the parking policy for students?_ → Student parking permits are required Monday–Friday. Permits can be purchased online via Campus Parking. Want the parking permit rates?
- **19** · _Do we have a vegan option in the cafeteria?_ → Yes — the cafeteria offers multiple vegan options daily. I can show today's menu if you'd like.
- **28** · _Do we have a student discount for public transport?_ → There is a reduced fare student pass available each semester. Visit Student Services for purchase instructions. Would you like the link?
- **36** · _Is there daycare on campus?_ → Yes, the Campus Childcare Center provides services; apply early as spots are limited. Want the application link?
- **41** · _Is the gym open late tonight?_ → The gym is open until 22:00 on weekdays. Would you like today's class schedule for the fitness center?
- **51** · _How can I dispute a parking ticket?_ → You can appeal a parking ticket via the Parking Appeals Portal. Collect evidence and submit within the appeal window. Want the appeal link?
- **58** · _Is there a lost and found on campus?_ → Yes — Lost & Found is at the Campus Police office. You can check items online or visit their office during business hours.
- **81** · _Are printers free to use?_ → Printing may require prepaid credits or charges per page. Check the Printing Services page for current rates and payment methods.
- **83** · _How to get a parking permit for the semester?_ → Purchase permits on the Parking Portal with your student ID. Choose the permit zone and payment method. Want the purchase link?
- **113** · _Is there an app for campus alerts?_ → Yes — download the Campus Safety app for alerts and emergency contacts. Want the app store link?
- **121** · _Is group study allowed in the library after hours?_ → Some library rooms are available for 24/7 group study with reservation. I can help you book a room.
- **127** · _How to report a food allergy to dining services?_ → Notify Dining Services via their allergy form and speak to the chef at meal times. I can provide the allergy accommodation form link.
- **136** · _How to reserve a conference room on campus?_ → Reserve conference rooms via the Room Booking portal. Choose date, time, and setup options. Would you like me to reserve one now?
- **139** · _Is there a campus shuttle schedule?_ → Yes — the shuttle schedule and routes are on the Transport Services page. I can show next shuttle times for your stop.
- **161** · _Where are the graduate student lounges?_ → Graduate lounges are in the Graduate Center and Science Building. I can show locations and access rules.

### `career_services` (9 rows) · used by CH_018, CH_042, CH_063
- **4** · _I need to book an appointment with career services._ → You can book an appointment via the Career Services portal. I can open the booking page or suggest available time slots if you share preferred days.
- **37** · _How do I check my internship credit approval?_ → Internship credit approvals are handled by your department. Submit the internship form and supervisor details to your department advisor. Shall I fetch the form?
- **48** · _How to apply for research assistant positions?_ → Research positions are posted on department boards and the Career Portal. I can search openings by department and send application tips.
- **90** · _Can I get help with my CV?_ → Career Services offers CV reviews and workshops. I can book a review session or provide CV tips now if you'd like.
- **96** · _Can the bot help me prepare for interviews?_ → I can provide common interview questions, role-specific tips, and mock interview practice. Which role are you interviewing for?
- **117** · _How to apply for on-campus jobs?_ → Search on-campus job listings via the Career Portal and submit your application online. Would you like help building your application?
- **149** · _How to apply for graduate assistantships?_ → Search graduate assistantship postings via department pages and the Career Portal. I can help prepare your application materials.
- **166** · _Can I get an appointment with the career counselor this week?_ → Career Counselors have limited slots; I can show available times and book an appointment this week if there's availability.
- **192** · _Are there campus job fairs coming up?_ → Yes — job fairs are scheduled this semester. I can list dates and employers if you tell me your field of interest.

### `course_registration` (9 rows) · used by CH_020, CH_021, CH_043, CH_065
- **0** · _How do I add a course to my schedule?_ → To add a course, go to the student portal > Registration > Add/Drop Courses and follow the prompts. If you tell me the course code I can check seat availability.
- **47** · _Is late registration allowed?_ → Late registration policies depend on the semester and course. There may be late fees; check the Registration Deadlines page. Want the deadline for your course?
- **71** · _How do I enroll in summer classes?_ → Summer enrollment opens on the Registration page. Search for summer term courses and add them to your cart; let me know which course codes you want.
- **80** · _Can I swap lab sections?_ → Swapping lab sections depends on seat availability. Use the Swap option in Registration or request instructor permission; shall I check availability?
- **99** · _Can I enroll in a class after it starts?_ → Late enrollments depend on instructor permission and seat availability. I can request instructor approval if you'd like.
- **108** · _Can I request late add for a course?_ → Late adds require instructor approval and may incur fees. I can submit a late add request if you provide the course code.
- **153** · _How to register for language classes?_ → Register via Registration; language classes have placement tests for some levels. Would you like the placement test schedule?
- **156** · _Can I register for evening classes only?_ → You can select primarily evening classes but check program requirements for availability. I can filter classes by time for you.
- **173** · _How to get into a full class waitlist?_ → Join the waitlist in Registration; some departments allow instructor overrides. I can check waitlist position if you're already queued.

### `financial_aid` (11 rows) · used by CH_022, CH_023, CH_045, CH_066
- **3** · _Are there scholarships for international students?_ → Yes — we have several scholarships for international students. Check the Financial Aid page and filter by 'International'. Would you like me to list currently open scholarships?
- **33** · _Can I get a refund for a dropped class?_ → Refund eligibility depends on the drop date and tuition policy. The Refunds page has the schedule. Want me to calculate a likely refund date based on drop date?
- **35** · _My financial aid package seems incorrect._ → I can help review it. Please provide the specific discrepancy, or I can create a support ticket to Financial Aid on your behalf.
- **72** · _Are there student emergency funds available?_ → Yes, the Student Emergency Fund provides short-term assistance. Apply via Student Affairs; I can link the application.
- **87** · _Where are study abroad scholarships listed?_ → Study Abroad scholarships are listed on both Financial Aid and Study Abroad pages. I can compile a list of scholarships with upcoming deadlines.
- **112** · _How do I update payment method for tuition?_ → Update payment methods in the Bursar portal under 'Payment Options'. I can guide you through the steps if needed.
- **116** · _Is there financial counseling for budgeting?_ → Yes — Financial Aid offers budgeting workshops and one-on-one counseling. I can book a session or provide budgeting templates.
- **175** · _Are there scholarships for student parents?_ → Some scholarships and emergency funds target student parents. I can compile applicable opportunities and deadlines.
- **180** · _Can I apply for a medical scholarship?_ → Medical scholarships exist for specific criteria. I can search scholarships relevant to your field and eligibility.
- **183** · _Can I defer payment for tuition?_ → Payment plans may be available through Bursar. Contact them to discuss options; I can provide the payment plan application link.
- **198** · _How to request a meeting with financial aid?_ → Book an appointment via the Financial Aid portal or call their office. I can schedule a meeting and list required documents.

### `housing` (5 rows) · used by CH_026, CH_047, CH_067
- **10** · _Is there on-campus housing available next semester?_ → Housing availability varies. Check the Housing Portal for application deadlines and current openings. Would you like me to show available room types?
- **32** · _Is there emergency housing in case of displacement?_ → Emergency housing is arranged by Student Affairs depending on availability. Contact Student Affairs immediately; I can provide contact info.
- **49** · _I think my roommate is violating housing rules._ → Report housing violations to Residential Life with details and dates. I can provide the reporting form or connect you to a housing coordinator.
- **89** · _What's the deadline to change housing assignments?_ → Housing reassignment deadlines are posted on the Housing Portal. Early applications have priority. Would you like the current deadlines?
- **196** · _How do I apply for emergency housing after a natural disaster?_ → Contact Student Affairs and Housing immediately; emergency protocols and temporary housing are activated as needed. I can provide emergency contacts.

### `mental_health` (4 rows) · used by CH_027, CH_048, CH_051, CH_068
- **7** · _I'm feeling anxious about finals._ → I'm sorry you're feeling anxious. The counseling center offers drop-in hours and appointments. Would you like to book a session or see immediate coping resources?
- **23** · _Is there counseling available for stress?_ → Yes — counseling services offer individual therapy and workshops. I can schedule an appointment or share immediate coping resources.
- **61** · _Do we have emergency counseling after hours?_ → After-hours crisis support is available via the 24/7 helpline. If you are in immediate danger, call emergency services. Want the helpline number?
- **85** · _Is mental health counseling confidential?_ → Yes, counseling services maintain confidentiality within legal limits. I can share privacy policies or help schedule an appointment.

### `research_support` (6 rows) · used by CH_028, CH_069
- **73** · _Can I get access to lab equipment outside class hours?_ → Access outside hours requires supervisor approval and training. I can help request extended access if you provide supervisor details.
- **145** · _How do I apply for research funding?_ → Research funding opportunities are posted by departments and the Office of Research. I can compile active funding calls for your field.
- **147** · _What's the protocol for lab equipment checkout?_ → Checkout requires training, supervisor approval, and logging equipment use. I can provide the checkout policy and request form.
- **167** · _How do I access lab protocols?_ → Lab protocols are usually in the department intranet or provided by your supervisor. I can search the protocol repository for your lab.
- **185** · _How do I get funding for conference travel?_ → Apply for travel grants through your department or the Office of Research. I can list current grant opportunities and deadlines.
- **197** · _Can I access lab data remotely?_ → Remote data access depends on permissions and VPN access. Coordinate with your PI or IT to set up secure access.

### `safety` (6 rows) · used by CH_025, CH_049, CH_070
- **29** · _How to report a crime on campus?_ → If it's an emergency, call campus police immediately. For non-emergency reports, use the Campus Safety portal or visit their office. Need the contact number?
- **53** · _What's the protocol for lab safety training?_ → Complete the online Lab Safety module through Environmental Health & Safety. Certificates are required before lab access. Need the module link?
- **102** · _Where do I report harassment?_ → Report harassment to Title IX Office or Campus Safety immediately. I can provide contact details and help you file a confidential report.
- **118** · _Are there emergency protocols for weather closures?_ → Follow Campus Alerts and the Emergency Procedures page for weather closures. I can subscribe you to alert notifications.
- **151** · _How do I report a hostile classroom environment?_ → Report incidents to the Office of Student Conduct and Title IX as appropriate. I can help gather details and submit a formal report.
- **163** · _How to get emergency medical help on campus?_ → For medical emergencies call campus police or local emergency services immediately. For non-emergency health issues contact Student Health Services.

### `student_services` (16 rows) · used by CH_016, CH_072
- **13** · _Who do I contact about accessibility services?_ → Contact the Accessibility Services Office at accessibility@university.edu or call (555) 010-1000. I can draft an email or give office hours if you'd like.
- **38** · _Can I access alumni resources as a current student?_ → Some alumni resources are available to current students, while others are restricted. I can list resources accessible to you now.
- **50** · _Are there opportunities for study abroad next year?_ → Yes — study abroad applications open in the fall. I can show program options and application deadlines for next year.
- **63** · _Is there on-campus childcare subsidies?_ → Some subsidies may be available through Student Services depending on eligibility. I can share application details if you'd like.
- **70** · _Do we have free legal advice for students?_ → The Student Legal Clinic offers free consultations on certain matters. I can provide their schedule and contact info.
- **98** · _Is free tutoring available for first-year students?_ → Yes — free tutoring programs prioritize first-year students. I can show available subjects and schedules.
- **106** · _Is there childcare for parenting students?_ → Campus Childcare offers programs; limited spots are available. I can share application details and financial assistance options.
- **123** · _Can I get childcare emergency support?_ → Student Affairs may provide emergency childcare assistance on a case-by-case basis. I can connect you to the office to discuss options.
- **126** · _Is there a mentorship program for freshmen?_ → Yes — the Freshman Mentorship Program pairs new students with upperclass mentors. Would you like to sign up?
- **129** · _Is there a veterans' office on campus?_ → Yes — Veterans Services assists with benefits and transition support. I can provide contact info and resources.
- **133** · _Is there assistance for undocumented students?_ → Student Support Services offers resources for undocumented students. I can share advisors and legal resources available on campus.
- **154** · _Is there support for first-generation students?_ → Yes — First-Generation Programs provide mentoring and resources. I can connect you with program coordinators.
- **168** · _Is there support for food insecurity?_ → The campus food pantry and emergency meal vouchers assist students facing food insecurity. I can provide hours and application details.
- **179** · _Is tutoring free for engineering students?_ → Engineering tutoring is free for enrolled engineering students. I can show the tutoring schedule and book a session.
- **184** · _Are there translation services for documents?_ → Translation services are available through Student Services or external partners. I can provide contacts and pricing.
- **199** · _Is there support for veteran spouses?_ → Veterans Services may provide resources and referrals for veteran spouses. I can provide contact info and available services.

### `technical_issue` (6 rows) · used by CH_012, CH_050, CH_071
- **2** · _I forgot my student portal password._ → I can help reset your password. Please visit the portal's Password Reset page or confirm you'd like me to send password reset instructions to your registered email.
- **21** · _My laptop won't connect to the projector._ → Try restarting your laptop and checking the HDMI/adapter connection. If it persists, campus AV support can assist. Shall I create a support ticket?
- **42** · _My student email is not receiving messages._ → Check your storage quota and spam folder. If the problem continues, reset your mail settings or submit a ticket; I can initiate that for you.
- **46** · _How do I reset two-factor authentication?_ → Resetting 2FA requires contacting IT or using backup codes. I can guide you through the reset steps or create a support request.
- **86** · _How do I connect my phone to campus VPN?_ → Install the VPN client recommended on the IT site and use your student credentials. I can provide step-by-step instructions for your device.
- **191** · _How to report a bug in the student app?_ → Report bugs via the IT Helpdesk with steps to reproduce and screenshots. I can file a bug report for you.

