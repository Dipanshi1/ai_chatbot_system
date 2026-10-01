import re


def clean_text(text: str) -> str:
    """
    Standardize text input:
    - lowercase
    - normalize repeated whitespace
    - strip surrounding whitespace

    Preserves the exact preprocessing behavior established during baseline training.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()
