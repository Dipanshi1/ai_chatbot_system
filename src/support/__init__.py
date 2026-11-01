"""
Student support layer around the frozen ML core: PII redaction, crisis triage,
heuristic sentiment, session privacy, escalation tickets and feedback.

Nothing in this package imports src.ml, so it cannot affect intent
classification, retrieval or the safety policy.
"""
