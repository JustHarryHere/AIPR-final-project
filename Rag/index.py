from __future__ import annotations
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from DocumentChunking import (Document, DocumentChunk, DocumentStore, chunk,
                          load_documents)
from rag.text import tokenize

JINA_MODEL = "jina-embeddings-v3"
JINA_BATCH = 64


class JinaEmbedder:
    """Class approach: Jina v3 with task=retrieval.passage / retrieval.query."""

    def __init__(self, model: str = JINA_MODEL, api_key: str | None = None) -> None:
        self.model, self.api_key = model, api_key
        self.name = model

    def embed(self, texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
        from jina_embeddings import jina_embeddings
        out: list[list[float]] = []
        for i in range(0, len(texts), JINA_BATCH):
            resp = jina_embeddings(list(texts[i:i + JINA_BATCH]), model=self.model,
                                   task=task, api_key=self.api_key)
            out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return out


def make_embedder(kind: str | None = None):
    kind = kind or os.getenv("EMBEDDER", "jina")
    if kind == "jina":
        return JinaEmbedder()

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a_arr, b_arr) / (norm_a * norm_b))))

