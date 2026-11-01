"""
Regex-based PII redaction, applied before inference, display and persistence.

Each placeholder is a single token outside the classifier and knowledge-base
vocabularies, so it adds no signal to the frozen model. "[EMAIL]" would not be
safe: it tokenizes to "email", which is in the classifier vocabulary.

This is pattern matching, not a guarantee. Free-text names, addresses and
spelled-out details ("jane dot doe at gmail dot com") are not detected, so
redacted text must still be treated as sensitive.
"""
from collections import Counter
from dataclasses import dataclass
import re
from typing import Dict

PLACEHOLDERS = {
    "EMAIL": "[REDACTED_EMAIL]",
    "PHONE": "[REDACTED_PHONE]",
    "CARD": "[REDACTED_CARD]",
    "ID": "[REDACTED_ID]",
    "NAME": "[REDACTED_NAME]",
}

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_LABELLED_ID = re.compile(
    r"(?i)\b(?P<label>"
    r"student\s*(?:id|no\.?|number)|roll\s*(?:no\.?|number)|enrol{1,2}ment\s*(?:id|no\.?|number)"
    r"|registration\s*(?:no\.?|number)|reg\s*no\.?|sid|ssn|aadha+r"
    r"|passport(?:\s*(?:no\.?|number))?|id(?:\s*(?:no\.?|number))?"
    r")(?P<sep>\s*(?:is|:|#|=)?\s*)"
    r"(?=[a-z0-9-]*\d)[a-z0-9][a-z0-9-]{3,}\b"
)
# Digit-first, so course codes such as MATH201 and CS101 never match.
_BARE_ID = re.compile(r"\b\d{2,4}[A-Za-z]{2,4}\d{3,6}\b")
_PHONE = re.compile(r"(?<![\w/.:-])\+?\(?\d[\d ().-]{5,19}\d(?![\w/:-])")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_STATED_NAME = re.compile(
    r"(?i)\b(?P<lead>my name is|my name's|i am called|i'm called)\s+"
    r"(?P<words>[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*)?)"
)
_NOT_A_NAME = frozenset(
    {
        "a", "also", "an", "and", "at", "because", "but", "from", "i", "in", "incorrect",
        "is", "listed", "misspelled", "missing", "my", "not", "on", "please", "so",
        "spelled", "still", "the", "wrong",
    }
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: Dict[str, int]

    @property
    def found(self) -> bool:
        return bool(self.counts)


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def redact(text: object) -> RedactionResult:
    """Replace detected PII with typed placeholders. Non-string input yields empty text."""
    if not isinstance(text, str):
        return RedactionResult("", {})

    counts: Counter = Counter()

    def mask(kind: str) -> str:
        counts[kind] += 1
        return PLACEHOLDERS[kind]

    def replace_card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return mask("CARD") if 13 <= len(digits) <= 19 and _luhn_valid(digits) else match.group(0)

    def replace_labelled_id(match: re.Match) -> str:
        return f"{match.group('label')}{match.group('sep')}{mask('ID')}"

    def replace_phone(match: re.Match) -> str:
        found = match.group(0)
        if _DATE.search(found):
            return found
        digit_count = sum(char.isdigit() for char in found)
        is_phone = (
            10 <= digit_count <= 15
            or (found.startswith("+") and digit_count >= 8)
            or re.fullmatch(r"\d{3}-\d{4}", found) is not None
        )
        return mask("PHONE") if is_phone else found

    def replace_name(match: re.Match) -> str:
        words = match.group("words").split()
        if words[0].lower() in _NOT_A_NAME:
            return match.group(0)
        trailing = f" {words[1]}" if len(words) > 1 and words[1].lower() in _NOT_A_NAME else ""
        return f"{match.group('lead')} {mask('NAME')}{trailing}"

    # Order matters: specific shapes run first so digit runs are not claimed as phones.
    text = _EMAIL.sub(lambda _: mask("EMAIL"), text)
    text = _SSN.sub(lambda _: mask("ID"), text)
    text = _CARD.sub(replace_card, text)
    text = _LABELLED_ID.sub(replace_labelled_id, text)
    text = _BARE_ID.sub(lambda _: mask("ID"), text)
    text = _PHONE.sub(replace_phone, text)
    text = _STATED_NAME.sub(replace_name, text)
    return RedactionResult(text, dict(counts))
