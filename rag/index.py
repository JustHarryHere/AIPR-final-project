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

@dataclass
class VectorStore(DocumentStore):
    embedder: object = field(default_factory=JinaEmbedder, repr=False)

    def add_chunk(self, chunk_: DocumentChunk) -> str:
        return self.add_chunks([chunk_])[0]

    def add_chunks(self, chunks: list[DocumentChunk]) -> list[str]:
        ids = [DocumentStore.add_chunk(self, c) for c in chunks]
        self._embed()
        return ids

    def _embed(self) -> None:
        """Embed what needs embedding. A corpus-fitted embedder (offline IDF)
        is refitted and everything re-embedded; Jina only embeds missing ones."""
        allc = list(self.chunks.values())
        if hasattr(self.embedder, "fit"):
            self.embedder.fit([c.content for c in allc])
            targets = allc
        else:
            targets = [c for c in allc if c.embedding is None]
        for c, v in zip(targets, self.embedder.embed([c.content for c in targets])):
            c.embedding = v

    def score_all(self, query: str) -> list[tuple[DocumentChunk, float]]:
        """Cosine of the query against EVERY chunk, in store order (for hybrid)."""
        qv = self.embedder.embed([query], task="retrieval.query")[0]
        return [(c, cosine_similarity(qv, c.embedding))
                for c in self.chunks.values() if c.embedding is not None]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0,
               where: Callable[[DocumentChunk], bool] | None = None
               ) -> list[tuple[DocumentChunk, float]]:
        if not self.chunks:
            return []
        results = [(c, s) for c, s in self.score_all(query)
                   if s >= min_score and (where is None or where(c))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path: str, include_embeddings: bool = True, meta: dict | None = None) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        docs_data = {}
        for doc_id, doc in self.documents.items():
            d = asdict(doc)
            d["created_at"] = doc.created_at.isoformat()
            docs_data[doc_id] = d
        chunks_data = {}
        for chunk_id, c in self.chunks.items():
            cd = asdict(c)
            if not include_embeddings:
                cd["embedding"] = None
            chunks_data[chunk_id] = cd
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta or {}, "documents": docs_data, "chunks": chunks_data},
                      f, ensure_ascii=False)

    def load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.documents = {}
        for doc_id, d in payload.get("documents", {}).items():
            ca = d.get("created_at")
            self.documents[doc_id] = Document(
                content=d["content"], doc_id=d["doc_id"], title=d["title"], source=d["source"],
                metadata=d.get("metadata", {}),
                created_at=datetime.fromisoformat(ca) if ca else datetime.now())
        self.chunks = {}
        for chunk_id, c in payload.get("chunks", {}).items():
            self.chunks[chunk_id] = DocumentChunk(
                content=c["content"], chunk_id=c["chunk_id"], doc_id=c["doc_id"],
                chunk_index=c.get("chunk_index", 0), metadata=c.get("metadata", {}),
                embedding=c.get("embedding"))
        self._embed()                     
        return payload.get("meta", {})

def corpus_fingerprint(docs_dir, metadata_path, strategy: str, chunk_kw: dict) -> str:
    h = hashlib.sha256()
    h.update(Path(metadata_path).read_bytes())
    for p in sorted(Path(docs_dir).glob("*")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    h.update(json.dumps([strategy, chunk_kw], sort_keys=True).encode())
    return h.hexdigest()[:16]


def build_store(docs_dir, metadata_path, embedder=None, strategy: str = "heading",
                cache_path: str | Path | None = None, **chunk_kw) -> VectorStore:
    """Load docs -> chunk -> embed -> VectorStore. Rebuildable from the corpus
    alone (`make setup`); the cache only saves Jina API calls."""
    embedder = embedder or make_embedder()
    fp = corpus_fingerprint(docs_dir, metadata_path, strategy, chunk_kw)
    store = VectorStore(embedder=embedder)

    if cache_path and Path(cache_path).exists():
        try:
            meta = json.loads(Path(cache_path).read_text(encoding="utf-8")).get("meta", {})
        except (OSError, ValueError):
            meta = {}
        if meta.get("fingerprint") == fp and meta.get("embedder") == embedder.name:
            store.load(str(cache_path))
            return store

    for doc in load_documents(docs_dir, metadata_path):
        store.documents[doc.doc_id] = doc
    store.add_chunks([c for d in store.documents.values() for c in chunk(d, strategy, **chunk_kw)])
    if cache_path:
        store.save(str(cache_path), meta={"fingerprint": fp, "embedder": embedder.name,
                                          "strategy": strategy})
    return store
