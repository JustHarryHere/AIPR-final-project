from __future__ import annotations

import math
from collections import Counter
import numpy as np

try:
    from rag.text import tokenize
except ImportError:
    try:
        from text import tokenize
    except ImportError:
        import re
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\w+", text.lower()) if text else []


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = float(self.lens.mean()) if len(texts) else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(texts)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                    out[i] += idf * tf * (self.k1 + 1) / denom
        return out


def saturate(bm25_scores: np.ndarray, s0: float = 5.0) -> np.ndarray:
    """Map raw BM25 into [0, 1) WITHOUT per-query max-normalising.

    Dividing by the max would give the best chunk 1.0 for any query sharing a
    single token, which destroys the no_answer threshold. 1 - exp(-s/s0)
    keeps weak matches weak; s0 is a tunable constant (record it in the config).
    """
    return 1.0 - np.exp(-bm25_scores / s0)


def fuse(dense: np.ndarray, bm25_sat: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """alpha * dense + (1 - alpha) * bm25. Both inputs live in [0, 1]."""
    return alpha * np.clip(dense, 0.0, 1.0) + (1 - alpha) * bm25_sat
