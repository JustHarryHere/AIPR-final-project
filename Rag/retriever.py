from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hybrid_search import BM25, fuse, saturate
from index import VectorStore, build_store, make_embedder
from schemas import Chunk, SearchResult

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.6
DEFAULT_S0 = 5.0
FALLBACK_THRESHOLDS = {"dense": 0.20, "hybrid": 0.15}


def resolve_versions(metadata: list[dict]) -> tuple[set[str], dict[str, str]]:
    """Return (active_doc_ids, warnings_by_doc_id). Deterministic, never guesses.

    Group by (course_id, doc_type). Within a group:
      1. If any doc has status == "official", only those are candidates.
      2. One candidate left -> it wins.
      3. Several: any missing date -> ambiguous (keep all, warn). Else the
         newest date wins; a tie on the newest date -> conflict (keep tied, warn).
    File names and index order are never used.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for m in metadata:
        groups[(m.get("course_id"), m.get("doc_type"))].append(m)

    active: set[str] = set()
    warnings: dict[str, str] = {}
    for key, docs in groups.items():
        if len(docs) == 1:
            active.add(docs[0]["doc_id"])
            continue
        cands = [d for d in docs if d.get("status") == "official"] or docs
        if len(cands) == 1:
            active.add(cands[0]["doc_id"])
            continue
        if any(not d.get("date") for d in cands):
            msg = f"version ambiguous for {key}: missing date/version metadata; cannot tell which is newer"
        else:
            newest = max(d["date"] for d in cands)
            top = [d for d in cands if d["date"] == newest]
            if len(top) == 1:
                active.add(top[0]["doc_id"])
                continue
            cands = top
            msg = f"version conflict for {key}: same date {newest}, unresolved"
        for d in cands:
            active.add(d["doc_id"])
            warnings[d["doc_id"]] = msg
    return active, warnings

class Retriever:
    def __init__(self, store: VectorStore, thresholds_path=None, top_k: int = DEFAULT_TOP_K,
                 alpha: float = DEFAULT_ALPHA, s0: float = DEFAULT_S0) -> None:
        self.store = store
        self.embedder_name = store.embedder.name
        self.top_k, self.alpha, self.s0 = top_k, alpha, s0
        self._chunks = list(store.chunks.values())           # fixed order for BM25
        self.bm25 = BM25([c.content for c in self._chunks])
        self.metadata = [{"doc_id": d.doc_id, **d.metadata} for d in store.documents.values()]
        self.active_docs, self.version_warnings = resolve_versions(self.metadata)

        self.thresholds = dict(FALLBACK_THRESHOLDS)
        tp = Path(thresholds_path or ROOT / "data" / "thresholds.json")
        if tp.exists():
            saved = json.loads(tp.read_text(encoding="utf-8"))
            if saved.get("embedder") == self.embedder_name:   # only trust same-embedder values
                self.thresholds.update(saved.get("thresholds", {}))

    @classmethod
    def from_corpus(cls, embedder: str | None = None, strategy: str = "heading",
                    docs_dir=None, metadata_path=None, cache: bool = True,
                    thresholds_path=None, **kw) -> "Retriever":
        docs_dir = docs_dir or ROOT / "data" / "docs"
        metadata_path = metadata_path or ROOT / "data" / "metadata.json"
        emb = make_embedder(embedder)
        cache_path = ROOT / "data" / "index" / f"{strategy}-{emb.name.replace(':', '_')}.json" if cache else None
        store = build_store(docs_dir, metadata_path, emb, strategy, cache_path)
        return cls(store, thresholds_path=thresholds_path, **kw)

    def _to_chunk(self, c, score: float) -> Chunk:
        md = c.metadata
        return Chunk(text=c.content, doc_id=c.doc_id, title=md.get("title", ""),
                     chunk_id=c.chunk_id, version=md.get("version"), date=md.get("date"),
                     course_id=md.get("course_id"), section=md.get("section"), score=float(score),
                     metadata={k: md.get(k) for k in ("source", "doc_type", "status", "owner", "start")})

    def search(self, query: str, mode: str = "hybrid", filters: dict | None = None,
               top_k: int | None = None) -> SearchResult:
        """filters: course_id, version, all_versions (bool).

        version given -> only that exact version (the user asked for it);
        all_versions  -> skip version resolution ("compare old vs new");
        neither       -> only the active (official/latest) docs.
        """
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"mode must be 'dense' or 'hybrid', got {mode!r}")
        filters = dict(filters or {})
        k = top_k or self.top_k
        threshold = self.thresholds[mode]
        result = SearchResult(query=query, mode=mode, filters=filters, retrieval_config={
            "top_k": k, "threshold": threshold, "embedder": self.embedder_name,
            "alpha": self.alpha if mode == "hybrid" else None,
            "s0": self.s0 if mode == "hybrid" else None})
        if not query or not query.strip():
            result.no_answer = True
            result.warnings.append("empty query")
            return result

        dense = np.array([s for _, s in self.store.score_all(query)])   # same order as _chunks
        scores = dense if mode == "dense" else fuse(dense, saturate(self.bm25.scores(query), self.s0), self.alpha)

        allowed = []
        for i, c in enumerate(self._chunks):
            md = c.metadata
            if filters.get("course_id") and md.get("course_id") != filters["course_id"]:
                continue
            if filters.get("version"):
                if md.get("version") != filters["version"]:
                    continue
            elif not filters.get("all_versions") and c.doc_id not in self.active_docs:
                continue
            allowed.append(i)
        if not allowed:
            result.no_answer = True
            result.warnings.append("no chunks match the filters")
            return result

        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)[:k]
        result.chunks = [self._to_chunk(self._chunks[i], scores[i]) for i in ranked]
        result.top_score = float(scores[ranked[0]])
        result.no_answer = result.top_score < threshold
        for doc_id in sorted({c.doc_id for c in result.chunks} & self.version_warnings.keys()):
            result.warnings.append(f"{doc_id}: {self.version_warnings[doc_id]}")
        return result


_default: Retriever | None = None


def search(query: str, mode: str = "hybrid", filters: dict | None = None) -> SearchResult:
    """Module-level interface from the brief (section 10)."""
    global _default
    if _default is None:
        _default = Retriever.from_corpus()
    return _default.search(query, mode, filters)

