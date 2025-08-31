"""
Active learning + lightweight MF helpers.

Design goals:
- If item factors V exist (n_items x K), use them for predictions + influence preview.
- If not available yet, gracefully fall back to popularity/random so UI still works.
- Ridge closed-form for new-user vector U given their ad-hoc ratings.
"""
from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")  # create project4/data/
V_PATH = os.path.join(DATA_DIR, "item_factors.npy")         # shape: [n_items, K]
MAP_PATH = os.path.join(DATA_DIR, "mlid_to_index.json")     # {ml_id: row_index_in_V}
POPULAR_PATH = os.path.join(DATA_DIR, "movie_popularity.json")  # {ml_id: count}

# ---------------- Core loaders ----------------

def _maybe_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_item_factors() -> Tuple[Optional[np.ndarray], Optional[Dict[str, int]]]:
    try:
        V = np.load(V_PATH)
        mlid2idx = _maybe_load_json(MAP_PATH)
        if not isinstance(mlid2idx, dict):
            return None, None
        # keys as str in json; convert to int keys for convenience
        mlid2idx = {int(k): int(v) for k, v in mlid2idx.items()}
        return V, mlid2idx
    except Exception:
        return None, None

def load_popularity() -> Dict[int, int]:
    pop = _maybe_load_json(POPULAR_PATH) or {}
    return {int(k): int(v) for k, v in pop.items()}

# ---------------- New-user vector solve ----------------

def solve_user_vector(
    ratings: Dict[int, float],  # {ml_id: rating}
    V: np.ndarray,
    mlid2idx: Dict[int, int],
    lam: float = 1.0
) -> Optional[np.ndarray]:
    if not ratings:
        return np.zeros((V.shape[1],), dtype=np.float32)
    rows, y = [], []
    for mlid, r in ratings.items():
        idx = mlid2idx.get(mlid)
        if idx is None: 
            continue
        rows.append(V[idx])
        y.append(r)
    if not rows:
        return np.zeros((V.shape[1],), dtype=np.float32)
    A = np.vstack(rows)                # m x K
    y = np.array(y, dtype=np.float32)  # m
    # ridge closed-form: u = (A^T A + lam I)^(-1) A^T y
    ATA = A.T @ A
    K = ATA.shape[0]
    u = np.linalg.solve(ATA + lam * np.eye(K), A.T @ y)
    return u

# ---------------- Predictions & selection ----------------

def predict_scores(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V @ u  # n_items

def select_next_movie(
    rated_ids: set[int],
    candidate_ids: List[int],
    V: Optional[np.ndarray],
    mlid2idx: Optional[Dict[int, int]],
    u: Optional[np.ndarray],
    pop: Dict[int, int],
    strategy: str = "uncertainty"
) -> Optional[int]:
    """
    - 'uncertainty': prefer items whose predicted score ~ 3.0 (middle), tie-break by diversity and popularity.
    - fallback: popularity among unseen.
    """
    unseen = [m for m in candidate_ids if m not in rated_ids]
    if not unseen:
        return None

    if V is None or mlid2idx is None or u is None:
        # popularity fallback
        unseen.sort(key=lambda mid: pop.get(mid, 0), reverse=True)
        return unseen[0]

    preds = []
    for mid in unseen:
        idx = mlid2idx.get(mid)
        if idx is None: 
            continue
        score = float(V[idx] @ u)
        # uncertainty = closeness to mid-point (3.0); smaller is better
        unc = abs(score - 3.0)
        # pop bonus (higher pop -> slightly lower "cost")
        bonus = -0.05 * np.log1p(pop.get(mid, 0))
        preds.append((unc + bonus, mid))
    if not preds:
        return None
    preds.sort(key=lambda x: x[0])
    return preds[0][1]

# ---------------- Influence preview ----------------

def influence_preview(
    current_ratings: Dict[int, float],
    probe_movie: int,
    probe_rating: float,
    V: Optional[np.ndarray],
    mlid2idx: Optional[Dict[int, int]],
    all_movie_ids: List[int],
    topn: int = 5,
    lam: float = 1.0
):
    """
    Returns the 'before' and 'after' top-N recommendations and a diff list.
    """
    if V is None or mlid2idx is None:
        # No MF yet -> can't compute true influence; return empty diff.
        return [], [], []

    # compute u_before
    u_before = solve_user_vector(current_ratings, V, mlid2idx, lam=lam)
    s_before = predict_scores(u_before, V)

    # add probe and compute u_after
    tmp = dict(current_ratings)
    tmp[probe_movie] = probe_rating
    u_after = solve_user_vector(tmp, V, mlid2idx, lam=lam)
    s_after = predict_scores(u_after, V)

    # build candidate list excluding already rated + the probe
    rated = set(current_ratings.keys()) | {probe_movie}
    cand = [m for m in all_movie_ids if m not in rated and mlid2idx.get(m) is not None]
    # rank
    idx_map = mlid2idx
    top_before = sorted(cand, key=lambda mid: s_before[idx_map[mid]], reverse=True)[:topn]
    top_after  = sorted(cand, key=lambda mid: s_after[idx_map[mid]], reverse=True)[:topn]

    diff = []
    pos_after = {mid: i for i, mid in enumerate(top_after)}
    for i, mid in enumerate(top_before):
        j = pos_after.get(mid, None)
        if j is None:
            diff.append((mid, i, None))  # dropped from topN
        else:
            diff.append((mid, i, j))     # moved from i -> j
    return top_before, top_after, diff
