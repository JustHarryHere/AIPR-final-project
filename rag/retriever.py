from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
RAG_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

import numpy as np

try:
    from rag.hybrid_search import BM25, fuse, saturate
except ImportError:
    from hybrid_search import BM25, fuse, saturate

from schemas import Chunk, SearchResult
DEFAULT_TOP_K = 4
DEFAULT_ALPHA = 0.6
DEFAULT_S0 = 5.0
FALLBACK_THRESHOLDS = {"dense": 0.25, "hybrid": 0.20}


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
    def __init__(
        self,
        docs_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        self.docs_dir = Path(docs_dir or ROOT / "data" / "docs")
        self.metadata_path = Path(metadata_path or self.docs_dir / "metadata.json")
        self.chunks: list[Chunk] = []
        self.metadata: list[dict] = []
        self.active_doc_ids: set[str] = set()
        self.bm25: BM25 | None = None
        self._load()

    def _load(self):
        if not self.metadata_path.exists():
            return
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.active_doc_ids, _ = resolve_versions(self.metadata)

        raw_chunks = []
        for meta in self.metadata:
            fpath = self.docs_dir / meta["file"]
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8")
            # Chunking theo heading ##
            sections = text.split("\n## ")
            for idx, sec in enumerate(sections):
                content = sec if idx == 0 else "## " + sec
                content = content.strip()
                if not content:
                    continue
                cid = f"{meta['doc_id']}_c{idx}"
                raw_chunks.append(Chunk(
                    content=content,
                    chunk_id=cid,
                    doc_id=meta["doc_id"],
                    chunk_index=idx,
                    metadata=dict(meta),
                ))

        self.chunks = raw_chunks
        if self.chunks:
            texts = [c.content for c in self.chunks]
            self.bm25 = BM25(texts)

    def search(self, query: str, mode: str = "hybrid", top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        """Search lesson plan chunks with version resolve and no_answer threshold."""
        if not query.strip() or not self.chunks or self.bm25 is None:
            return []

        # 1. BM25 scores
        bm25_raw = self.bm25.scores(query)
        bm25_sat = saturate(bm25_raw, s0=DEFAULT_S0)

        # 2. Dense semantic score (term overlap + length cosine proxy)
        dense_scores = np.zeros(len(self.chunks))
        q_tokens = set(query.lower().split())
        for i, c in enumerate(self.chunks):
            c_tokens = set(c.content.lower().split())
            if q_tokens and c_tokens:
                dense_scores[i] = len(q_tokens & c_tokens) / (len(q_tokens) ** 0.5 * len(c_tokens) ** 0.5 + 1e-6)

        # 3. Fuse scores based on mode
        if mode == "dense":
            final_scores = dense_scores
            thresh = FALLBACK_THRESHOLDS["dense"]
        else:
            final_scores = fuse(dense_scores, bm25_sat, alpha=DEFAULT_ALPHA)
            thresh = FALLBACK_THRESHOLDS["hybrid"]

        # 4. Filter by active version and threshold
        scored_candidates = []
        for idx, score in enumerate(final_scores):
            chunk = self.chunks[idx]
            if chunk.doc_id not in self.active_doc_ids:
                continue
            if score >= thresh and (bm25_raw[idx] > 0 or dense_scores[idx] >= FALLBACK_THRESHOLDS["dense"]):
                scored_candidates.append((score, chunk))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top = scored_candidates[:top_k]

        return [
            SearchResult(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                text=c.content,
                score=round(float(s), 4),
                version=c.metadata.get("version", "1.0"),
                source=c.metadata.get("file", ""),
            )
            for s, c in top
        ]


_default_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever


def search_lesson_plan(query: str, mode: str = "hybrid", top_k: int = 4) -> list[SearchResult]:
    """Contract for SV3 to import directly into MCP server."""
    retriever = get_retriever()
    return retriever.search(query, mode=mode, top_k=top_k)


