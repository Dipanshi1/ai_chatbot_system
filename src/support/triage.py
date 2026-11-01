"""
Phrase-based triage, run on redacted text before any ML processing.

Crisis patterns deliberately over-trigger: a false alarm costs a supportive
message, a miss costs far more. They are not exhaustive and do not handle
misspellings, so they are a safety net, not a safety guarantee.
"""
from enum import Enum
import re
from typing import Optional, Sequence, Tuple

from .config import PHYSICAL_DANGER_MESSAGE, SELF_HARM_MESSAGE


class Signal(str, Enum):
    SELF_HARM = "self_harm"
    PHYSICAL_DANGER = "physical_danger"
    HUMAN_REQUEST = "human_request"
    OPT_OUT = "opt_out"


_NOT = r"(?:do not|don't|dont)"
_SELF = r"(?:i am|i'm|im|we are|we're)"

_SELF_HARM = re.compile(
    r"\bkill(?:ing)? myself\b"
    r"|\bend(?:ing)? my (?:own )?life\b|\bend(?:ing)? it all\b|\btak(?:e|ing) my (?:own )?life\b"
    r"|\bsuicid(?:e|al)\b"
    r"|\b(?:want|going|wish|ready) to die\b|\bwish i (?:was|were) dead\b"
    rf"|\b{_NOT} want to (?:live|be alive|be here|exist|wake up)\b"
    r"|\b(?:hurt|hurting|harm|harming|cut|cutting) myself\b|\bself[- ]harm"
    r"|\bno reason to live\b|\bbetter off (?:dead|without me)\b"
)
_PHYSICAL_DANGER = re.compile(
    r"\b(?:following|stalking|chasing|attacking|assaulting|hitting|beating) me\b"
    r"|\bthreaten\w* (?:me\b|to (?:hurt|kill|harm|attack)\b)"
    r"|\b(?:going to|gonna|trying to|wants? to|will) (?:kill|hurt|attack|stab|shoot|rape) me\b"
    rf"|\b{_SELF} being (?:attacked|assaulted|followed|stalked|threatened|chased)\b"
    r"|\b(?:has|have|had|with|holding|carrying|there is|there's) (?:an? )?(?:gun|knife|weapon|bomb)\b"
    r"|\bactive shooter\b"
    rf"|\b{_SELF} in danger\b|\b{_NOT} feel safe\b|\bnot safe (?:right now|here)\b|\bfeel unsafe\b"
    rf"|\b{_SELF} trapped\b"
    r"|\bnot breathing\b|\bunconscious\b|\boverdosed\b|\bheart attack\b"
    r"|\b(?:there is|there's) a fire\b|\bis on fire\b"
)
_HUMAN_REQUEST = re.compile(
    r"\b(?:talk|speak|chat|connect me|transfer me|put me through)\b (?:to|with) "
    r"(?:an? |the |some )?(?:real |actual |live )?"
    r"(?:human|person|agent|representative|staff|someone|somebody)\b"
    r"|\b(?:human agent|real person|live agent)\b"
)
_OPT_OUT = re.compile(
    r"^opt[- ]?out\W*$"
    r"|\bopt(?:ing)?[- ]?out of (?:data|storage|tracking|this chat|being stored)\b"
    r"|\bdelete my (?:data|information|details|messages|chat)\b"
    rf"|\b{_NOT} (?:want (?:you to )?)?(?:my (?:data|information|details|messages|chat) )?"
    r"(?:stored?|saved?|kept|keep)\b"
)

_ORDERED_PATTERNS = (
    (Signal.SELF_HARM, _SELF_HARM),
    (Signal.PHYSICAL_DANGER, _PHYSICAL_DANGER),
    (Signal.HUMAN_REQUEST, _HUMAN_REQUEST),
    (Signal.OPT_OUT, _OPT_OUT),
)
_CRISIS_SIGNALS = frozenset({Signal.SELF_HARM, Signal.PHYSICAL_DANGER})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").lower()).strip()


def detect_signals(text: object) -> Tuple[Signal, ...]:
    """Return every matched signal, most urgent first."""
    if not isinstance(text, str):
        return ()
    normalized = _normalize(text)
    return tuple(signal for signal, pattern in _ORDERED_PATTERNS if pattern.search(normalized))


def is_crisis(signals: Sequence[Signal]) -> bool:
    return any(signal in _CRISIS_SIGNALS for signal in signals)


def crisis_guidance(signals: Sequence[Signal]) -> Optional[str]:
    """Fixed guidance for the crises present. Immediate danger is shown first."""
    parts = []
    if Signal.PHYSICAL_DANGER in signals:
        parts.append(PHYSICAL_DANGER_MESSAGE)
    if Signal.SELF_HARM in signals:
        parts.append(SELF_HARM_MESSAGE)
    return "\n\n".join(parts) or None


def should_offer_escalation(
    abstained: bool, sentiment_label: str, signals: Sequence[Signal]
) -> bool:
    """
    Offer a support request after a human request, or when the assistant gave no
    reliable answer (ChatResult.abstained) to a negative-sounding message. Crisis
    messages never get an offer: the unmonitored queue must not look like help.
    """
    if is_crisis(signals):
        return False
    if Signal.HUMAN_REQUEST in signals:
        return True
    return abstained and sentiment_label == "negative"
