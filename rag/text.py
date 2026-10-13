import re


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase terms for BM25 and keyword search."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())
