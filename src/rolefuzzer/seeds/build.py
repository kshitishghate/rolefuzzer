"""Build a role's WildChat seed pool by semantic retrieval.

Streams WildChat-1M from Hugging Face, embeds each conversation's opening
user messages with ``jinaai/jina-embeddings-v3``, scores them by maximum
cosine similarity to the role's query strings (``seeds/queries/<role>.yaml``)
and keeps those above the threshold. Records keep only the fields the seed
loader reads; WildChat's IP hashes, locations and request headers are
dropped. The campaign then audits the pool and draws its seed set from it.

Requires the ``seeds`` extra and access to the gated ``allenai/WildChat-1M``
dataset on Hugging Face.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import yaml

from rolefuzzer.paths import SEED_QUERIES_DIR, wildchat_seed_path

logger = logging.getLogger(__name__)

DATASET = "allenai/WildChat-1M"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_THRESHOLD = 0.55


def load_queries(role_id: str) -> tuple[list[str], float]:
    path = SEED_QUERIES_DIR / f"{role_id}.yaml"
    if not path.exists():
        raise SystemExit(f"No retrieval queries for role {role_id!r}: create {path}")
    data = yaml.safe_load(path.read_text())
    return data["queries"], float(data.get("threshold", DEFAULT_THRESHOLD))


def opening_user_text(conversation: list[dict], n: int) -> str:
    texts = [t.get("content", "").strip() for t in conversation if t.get("role") == "user"]
    return " ".join([t for t in texts if t][:n])


def _clean_turns(conversation: list[dict]) -> list[dict]:
    return [{"role": t.get("role"), "content": t.get("content", "")} for t in conversation]


def stream_records(language: str, user_messages: int, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    records = []
    for row in load_dataset(DATASET, split="train", streaming=True):
        if language and row.get("language") != language:
            continue
        conversation = row.get("conversation") or []
        text = opening_user_text(conversation, user_messages)
        if not text:
            continue
        records.append({
            "conversation_hash": row.get("conversation_hash", ""),
            "model": row.get("model", ""),
            "language": row.get("language", ""),
            "turn": row.get("turn", 0),
            "toxic": bool(row.get("toxic", False)),
            "first_user_text": text,
            "conversation": _clean_turns(conversation),
        })
        if limit and len(records) >= limit:
            break
        if len(records) % 50_000 == 0:
            logger.info("streamed %d conversations", len(records))
    return records


def score(records: list[dict], queries: list[str], batch_size: int, max_chars: int) -> list[float]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    q = model.encode(queries, batch_size=batch_size, show_progress_bar=False)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    texts = [r["first_user_text"][:max_chars] for r in records]
    c = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    return (c @ q.T).max(axis=1).astype(float).tolist()


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super().default(obj)


def build_seed_pool(
    role_id: str,
    *,
    threshold: float | None = None,
    language: str = "English",
    user_messages: int = 4,
    batch_size: int = 64,
    max_chars: int = 2048,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    out = wildchat_seed_path(role_id)
    if out.exists() and not force:
        raise SystemExit(f"{out} exists; pass --force to replace it")
    queries, default_threshold = load_queries(role_id)
    threshold = default_threshold if threshold is None else threshold
    records = stream_records(language, user_messages, limit)
    scores = score(records, queries, batch_size, max_chars)
    kept = []
    for record, s in zip(records, scores):
        if s >= threshold:
            kept.append({**record, "similarity_score": s})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, cls=_Encoder) + "\n")
    logger.info("Kept %d of %d conversations (similarity >= %.2f) in %s",
                len(kept), len(records), threshold, out)
    return out
