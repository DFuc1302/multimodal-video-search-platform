"""
kis_conversation.py — Conversational KIS (KISC)
=================================================
Implements the AIC2026 "Conversational KIS" task: instead of one shot
query -> results, the assistant runs a multi-turn dialogue that narrows a
vague first query down with clarifying questions (indoor/outdoor, who's in
frame, color/object, ...) until the candidate pool is small and confident.

Design goals, given the existing pipeline in shared.py / main.py:
  - Turn 1 pays the one expensive FAISS + CLIP-text-embed cost, exactly like
    /search today.
  - Every later turn re-uses the *already narrowed* in-memory pool and only
    reconstructs CLIP vectors straight out of the FAISS index (see
    FaissCollection.reconstruct_by_files in shared.py) — no re-embedding,
    no image I/O, no model calls after turn 1. This keeps each clarification
    round near-instant even though full search over 700k frames is not.
  - Slot filling for attributes YOLO/OCR can't give us for free (indoor vs
    outdoor, gender, day vs night) uses CLIP zero-shot prompts scored
    against the same reconstructed vectors, so it's "free" once we have the
    vectors for the pool.
  - Object/category/color slots reuse the exact same vocab already used by
    /search (CATEGORY_MAP, COLOR_ALIASES) so behavior stays consistent
    between the single-shot and conversational search modes.

Sessions live in memory (dict), keyed by a session_id the client generates
(e.g. a uuid4 per conversation). Fine for a single-machine competition rig;
swap for redis if this ever needs to run distributed.
"""

import threading
from redis_store import ENABLED as REDIS_ENABLED, store as redis_store
import time
import uuid
from typing import Optional

import numpy as np

from shared import (
    get_collection, get_meta_by_files,
    embed_texts,
    CATEGORY_MAP, COLOR_ALIASES, ATTRIBUTE_SLOTS, TOP_K_MAX,
)
from query_hints import keyword_hints

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

FIRST_TURN_FETCH   = 3000   # how many candidates FAISS returns on turn 1
CLARIFY_ABOVE      = 12     # keep asking while pool is bigger than this
FINAL_TOP_K        = 1000   # default number of results shown in the UI
MAX_TURNS          = 6      # hard stop so a confused dialogue still returns something
SESSION_TTL_SEC    = 60 * 60  # sweep sessions idle for over an hour
SLOT_ORDER         = ["location", "gender", "time_of_day"]  # ask priority
ZERO_SHOT_MARGIN   = 0.02   # min score gap to commit a frame to one side of a slot

_ALL_OBJECT_NAMES = sorted({o for members in CATEGORY_MAP.values() for o in members})

# ══════════════════════════════════════════════════════════════════════════════
# Zero-shot prompt embeddings — computed once, reused across every session
# ══════════════════════════════════════════════════════════════════════════════

_slot_prompt_vecs: dict[str, dict[str, np.ndarray]] = {}
_slot_prompt_lock = threading.Lock()

def _get_slot_prompt_vecs(slot: str) -> dict[str, np.ndarray]:
    """Lazily CLIP-embed each option's prompts for a slot, cached forever."""
    with _slot_prompt_lock:
        if slot not in _slot_prompt_vecs:
            options = ATTRIBUTE_SLOTS[slot]["zero_shot"]
            _slot_prompt_vecs[slot] = {
                opt: embed_texts(prompts) for opt, prompts in options.items()
            }
        return _slot_prompt_vecs[slot]

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

class KISSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self.turns: list[str] = []
        self.text_queries: list[str] = []
        self.filters: dict = {           # accumulated hard filters
            "objects": set(),
            "categories": set(),
            "colors": set(),
            "video": None,
        }
        self.slot_values: dict = {}      # slot -> chosen option ("indoor"/"outdoor"/...)
        self.asked_slots: set = set()
        self.pool: Optional[list[dict]] = None   # narrowed candidate metadata + _score
        self.vectors: dict[str, np.ndarray] = {} # file -> reconstructed CLIP vector, cached
        self.created_at = time.time()
        self.updated_at = time.time()

_sessions: dict[str, KISSession] = {}
_sessions_lock = threading.Lock()

def _sweep_stale_sessions():
    cutoff = time.time() - SESSION_TTL_SEC
    with _sessions_lock:
        stale = [sid for sid, s in _sessions.items() if s.updated_at < cutoff]
        for sid in stale:
            del _sessions[sid]

def start_session() -> str:
    if REDIS_ENABLED:
        sid = uuid.uuid4().hex
        save_session(KISSession(sid))
        return sid
    _sweep_stale_sessions()
    sid = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[sid] = KISSession(sid)
    return sid

def get_session(session_id: str) -> Optional[KISSession]:
    if REDIS_ENABLED:
        data = redis_store().json('kis:' + session_id)
        if data is None:
            return None
        session = KISSession(session_id)
        session.__dict__.update(data)
        session.asked_slots = set(session.asked_slots)
        for key in ('objects', 'categories', 'colors'):
            session.filters[key] = set(session.filters[key])
        session.vectors = {}
        return session
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is not None and time.time() - session.updated_at >= SESSION_TTL_SEC:
            del _sessions[session_id]
            return None
        return session

def end_session(session_id: str):
    if REDIS_ENABLED:
        redis_store().r.delete(redis_store().prefix + 'kis:' + session_id)
        return
    with _sessions_lock:
        _sessions.pop(session_id, None)

# ══════════════════════════════════════════════════════════════════════════════
# Free-text parsing — auto-fills slots/filters from the user's own wording so
# the dialogue only asks about what genuinely wasn't said yet
# ══════════════════════════════════════════════════════════════════════════════

def _extract_slot_hints(text: str) -> dict:
    """Scan free text for slot keywords (bilingual). Returns {slot: option}."""
    hits = {}
    for slot, cfg in ATTRIBUTE_SLOTS.items():
        matched = keyword_hints(text, {kw: option for option, kws in cfg['keywords'].items() for kw in kws})
        for option, kws in cfg["keywords"].items():
            if option in matched:
                hits[slot] = option
                break
    return hits

def _extract_color_hints(text: str) -> set:
    return keyword_hints(text, COLOR_ALIASES)

def _extract_object_hints(text: str) -> set:
    return keyword_hints(text, {word: word for word in _ALL_OBJECT_NAMES})

# ══════════════════════════════════════════════════════════════════════════════
# Embedding helpers
# ══════════════════════════════════════════════════════════════════════════════

def _combined_query_vector(text_queries: list[str]) -> np.ndarray:
    """
    Weighted CLIP text embedding across every turn so far — the most recent
    utterance (the clarification the user just gave) counts more than the
    original vague first query, without discarding earlier context.
    """
    if not text_queries:
        raise ValueError("No text queries to embed")
    weighted = list(text_queries)
    weighted += [text_queries[-1]] * min(2, len(text_queries))  # ~3x weight on latest
    return embed_texts(weighted)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both already L2-normalized

# ══════════════════════════════════════════════════════════════════════════════
# Core turn processing
# ══════════════════════════════════════════════════════════════════════════════

def process_turn(session_id: str, message: str, top_k: int = FINAL_TOP_K,
                 *, collection=None, metadata_loader=None) -> dict:
    top_k = max(1, min(int(top_k), TOP_K_MAX))
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Unknown KIS session: {session_id}")

    session.turns.append(message)
    session.text_queries.append(message)
    session.updated_at = time.time()

    # ── auto-fill slots/filters from this turn's free text ────────────────────
    for slot, option in _extract_slot_hints(message).items():
        session.slot_values[slot] = option
    session.filters["colors"]  |= _extract_color_hints(message)
    session.filters["objects"] |= _extract_object_hints(message)

    combined_vec = _combined_query_vector(session.text_queries)
    col = collection if collection is not None else get_collection()
    load_metadata = metadata_loader if metadata_loader is not None else get_meta_by_files

    # ── turn 1: pay the real FAISS cost, build the working pool ──────────────
    if session.pool is None:
        n_results = max(1, col.count())
        cr = col.query(query_embeddings=[combined_vec.tolist()], n_results=n_results)
        cand_files = [m["file"] for m in cr["metadatas"][0]]
        raw_scores = {m["file"]: 1 - d for m, d in zip(cr["metadatas"][0], cr["distances"][0])}
        pool = []
        for r in load_metadata(cand_files):
            item = dict(r)
            item["_score"] = raw_scores.get(item["file"], 0.0)
            pool.append(item)
        session.pool = pool
    else:
        pool = session.pool

    # ── hard filters accumulated so far (objects/categories/colors) ──────────
    if session.filters["objects"]:
        wanted = session.filters["objects"]
        pool = [e for e in pool if wanted & {o.lower() for o in e["objects"]}]
    if session.filters["categories"]:
        wanted = session.filters["categories"]
        pool = [e for e in pool if wanted & set(e["categories"])]
    if session.filters["colors"]:
        wanted = session.filters["colors"]
        pool = [e for e in pool if wanted & set(e["colors"])]

    # ── zero-shot slot filters (location / gender / time_of_day) ─────────────
    # Reconstruct vectors once per file, cached on the session so repeat
    # turns over the same shrinking pool never touch FAISS/CLIP again.
    need_vecs = [e["file"] for e in pool if e["file"] not in session.vectors]
    if need_vecs:
        session.vectors.update(col.reconstruct_by_files(need_vecs))

    for slot, option in session.slot_values.items():
        cfg = ATTRIBUTE_SLOTS[slot]
        requires = cfg.get("requires_object")
        if requires:
            pool = [e for e in pool if requires in {o.lower() for o in e["objects"]}]
        prompt_vecs = _get_slot_prompt_vecs(slot)
        chosen_vec = prompt_vecs[option]
        rival_opts = [o for o in prompt_vecs if o != option]

        kept = []
        for e in pool:
            vec = session.vectors.get(e["file"])
            if vec is None:
                kept.append(e)  # no vector (shouldn't normally happen) — don't drop it
                continue
            score_chosen = _cos(vec, chosen_vec)
            score_rivals = max(_cos(vec, prompt_vecs[o]) for o in rival_opts) if rival_opts else -1.0
            if score_chosen - score_rivals >= -ZERO_SHOT_MARGIN:
                # frame at least ties on this slot's chosen option
                kept.append(e)
        pool = kept

    # ── re-rank by similarity to the combined (recency-weighted) query ───────
    for e in pool:
        vec = session.vectors.get(e["file"])
        e["_score"] = _cos(vec, combined_vec) if vec is not None else e["_score"]
    pool.sort(key=lambda e: (-e["_score"], e["file"]))

    session.pool = pool

    # ── decide: ask another clarifying question, or finish? ──────────────────
    next_slot = None
    if len(pool) > CLARIFY_ABOVE and len(session.turns) < MAX_TURNS:
        for slot in SLOT_ORDER:
            if slot in session.slot_values or slot in session.asked_slots:
                continue
            requires = ATTRIBUTE_SLOTS[slot].get("requires_object")
            if requires and not any(requires in {o.lower() for o in e["objects"]} for e in pool):
                continue  # e.g. don't ask about gender if nobody's in the pool
            next_slot = slot
            break

    preview = pool[:top_k]
    results = [{
        "file":       e["file"],
        "objects":    e["objects"],
        "categories": e["categories"],
        "colors":     e["colors"],
        "score":      round(float(e["_score"]), 4),
    } for e in preview]

    response = {
        "session_id":   session.id,
        "turn":         len(session.turns),
        "pool_size":    len(pool),
        "top_k":        top_k,
        "results":      results,
        "clarifying":   next_slot is not None,
        "question":     ATTRIBUTE_SLOTS[next_slot]["question"] if next_slot else None,
        "slot":         next_slot,
        "slots_filled": dict(session.slot_values),
        "filters": {
            "objects":    sorted(session.filters["objects"]),
            "categories": sorted(session.filters["categories"]),
            "colors":     sorted(session.filters["colors"]),
        },
    }
    if next_slot:
        session.asked_slots.add(next_slot)
    if REDIS_ENABLED:
        save_session(session)
    return response

def save_session(session):
    data = {k: v for k, v in session.__dict__.items() if k != 'vectors'}
    data['asked_slots'] = sorted(data['asked_slots'])
    data['filters'] = {k: sorted(v) if isinstance(v, set) else v for k, v in data['filters'].items()}
    redis_store().put_json('kis:' + session.id, data, ttl=SESSION_TTL_SEC)
