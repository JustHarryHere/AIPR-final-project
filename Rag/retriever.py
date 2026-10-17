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


