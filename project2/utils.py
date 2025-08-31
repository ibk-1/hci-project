"""
Utility helpers for Project-2:
- Task-1: full-data supervised baseline (IMDB 50k sentiment)
- Task-2: pool-based Active Learning with uncertainty sampling
"""

from __future__ import annotations

import json
import pathlib
import pickle
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils import Bunch

# ---------------------- Representation back-ends ---------------------- #
class BaseEncoder:
    """Every encoder returns a 2-D numpy / sparse matrix via .fit_transform / .transform"""

    def fit_transform(self, texts):
        raise NotImplementedError()

    def transform(self, texts):
        raise NotImplementedError()


class TfidfEncoder(BaseEncoder):
    def __init__(self, max_features=20_000):  # a bit smaller → faster
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            dtype=np.float32,  # lighter
        )

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts)

    def transform(self, texts):
        return self.vec.transform(texts)


class DistilBERTEncoder(BaseEncoder):
    """
    Lightweight sentence-level embeddings via sentence-transformers.
    NOTE: Heavier than TF-IDF; keep for demos, default to TF-IDF for speed.
    """

    def __init__(self, model_name="distilbert-base-nli-mean-tokens"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        return np.asarray(self.model.encode(list(texts), show_progress_bar=False))


# Map short names → encoder classes
ENCODERS = {
    "tfidf": TfidfEncoder,
    "distilbert": DistilBERTEncoder,
}


# ------------------------- Classifier heads --------------------------- #
def build_classifier(name: str):
    if name == "logreg":
        # Sparse-friendly + fast incremental refits during AL
        return LogisticRegression(max_iter=2000, solver="liblinear", warm_start=True)
    if name == "svm":
        # decision_function only; we convert to probs in AL
        return LinearSVC()
    raise ValueError(f"Unknown classifier: {name}")


# ----------------------------- Dataset I/O ---------------------------- #
def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    """
    Load any CSV with sentiment columns and normalise to ['text', 'label'].
    - text column ∈ {text, review, sentence, comment}
    - label column ∈ {label, sentiment, class, target}
    - strings 'positive'/'negative' → 1/0
    """
    df = pd.read_csv(csv_path)

    text_col_candidates = ["text", "review", "sentence", "comment"]
    label_col_candidates = ["label", "sentiment", "class", "target"]

    try:
        text_col = next(c for c in text_col_candidates if c in df.columns)
        label_col = next(c for c in label_col_candidates if c in df.columns)
    except StopIteration:
        raise ValueError(
            "CSV must contain a text column (e.g. 'review') and a label column (e.g. 'sentiment')."
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    if df.label.dtype == object:  # strings
        mapping = {"positive": 1, "negative": 0}
        df["label"] = (
            df.label.str.lower().map(mapping).astype("Int64", errors="ignore")
        )

    if df.label.isnull().any():
        raise ValueError(
            "Some label values could not be interpreted as positive/negative."
        )

    return df


def dataset_head(csv_path: pathlib.Path | None, n=5):
    """Return a list[{label, snippet}] or [] if path is None."""
    if not csv_path or not csv_path.exists():
        return []
    df = load_dataset(csv_path).iloc[:n]
    return (
        df.assign(snippet=lambda d: d.text.str.slice(0, 120) + "…")
        .loc[:, ["label", "snippet"]]
        .to_dict("records")
    )


# ------------------------- Task-1: Supervised ------------------------- #
def train_full_pipeline(
    cfg: Dict[str, Any],
    job_id: uuid.UUID,
    media_root: pathlib.Path,
    csv_path: pathlib.Path,
) -> Tuple[float, str, pathlib.Path]:
    """
    Train/test split, encode, fit classifier, report accuracy.
    Returns (test_accuracy, classification_report, model_dir).
    """
    df = load_dataset(csv_path=csv_path)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Encode
    enc_cls = ENCODERS[cfg.get("representation", "tfidf")]
    encoder: BaseEncoder = enc_cls()
    X_train = encoder.fit_transform(train_df.text)
    X_test = encoder.transform(test_df.text)

    # Train
    clf = build_classifier(cfg.get("classifier", "logreg"))
    clf.fit(X_train, train_df.label)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(test_df.label, y_pred)
    report = classification_report(
        test_df.label, y_pred, target_names=["negative", "positive"]
    )

    # Persist artefacts
    out_dir = media_root / "models" / f"{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "encoder.pkl", "wb") as fp:
        pickle.dump(encoder, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_dir / "classifier.pkl", "wb") as fp:
        pickle.dump(clf, fp, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "meta.json").write_text(
        json.dumps({"accuracy": acc, "cfg": cfg}, indent=2)
    )
    (out_dir / "report.txt").write_text(report)

    return acc, report, out_dir


# ────────────────────────────────────────────────────────────────────────
# ACTIVE LEARNING CORE  (Task-2 / Task-3)
# ────────────────────────────────────────────────────────────────────────
def _entropy(probs):
    # returns a vector: higher = more uncertain
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def _margin(probs):
    # higher = more uncertain (1 - (p1 - p2))
    top2 = np.sort(probs, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return 1.0 - margin


def _least_confident(probs):
    return 1 - probs.max(axis=1)


UTILITY_FUNCS = {
    "entropy": _entropy,
    "margin": _margin,
    "least_confident": _least_confident,
    "random": lambda probs: np.random.rand(len(probs)),
}


@dataclass
class ActiveSession:
    # raw pool (possibly subsetted for speed)
    X_pool: list[str]
    y_pool: np.ndarray  # gold labels (for simulation)
    encoder: BaseEncoder
    clf: Any  # e.g., LogisticRegression/LinearSVC
    strategy: str
    budget: int
    batch_size: int = 1

    # performance / speed
    X_pool_vec: Any | None = None  # vectorised pool
    candidate_size: int = 4096  # subsample per query for speed

    labelled_idx: list[int] = field(default_factory=list)
    history_acc: list[float] = field(default_factory=list)
    _label_dict: dict[int, int] = field(default_factory=dict)

    def _get_probs(self, X_vec):
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X_vec)
        if hasattr(self.clf, "decision_function"):
            scores = self.clf.decision_function(X_vec)
            if scores.ndim == 1:  # binary
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p1, p1])
            # multi-class softmax
            z = scores - scores.max(axis=1, keepdims=True)
            ez = np.exp(z)
            return ez / ez.sum(axis=1, keepdims=True)
        raise NotFittedError("Model has no confidence interface.")

    def query_indices(self, k=None):
        if k is None:
            k = self.batch_size

        pool_ids = [i for i in range(len(self.X_pool)) if i not in self.labelled_idx]
        if self.strategy == "random":
            return random.sample(pool_ids, k)

        # Subsample candidates to avoid full-pool scoring every round
        if len(pool_ids) > self.candidate_size:
            pool_ids = random.sample(pool_ids, self.candidate_size)

        try:
            X_vec = self.X_pool_vec[pool_ids]
            probs = self._get_probs(X_vec)
            scores = UTILITY_FUNCS[self.strategy](probs)
            ranked = [pid for _, pid in sorted(zip(scores, pool_ids), reverse=True)][:k]
            return ranked
        except NotFittedError:
            # First rounds (not enough labels) → random
            return random.sample(pool_ids, k)

    def step(self, indices, labels):
        # add newly labeled examples
        for i, lab in zip(indices, labels):
            self._label_dict[i] = lab
            if i not in self.labelled_idx:
                self.labelled_idx.append(i)

        # train only if ≥2 classes present
        y_lab_list = list(self._label_dict.values())
        if len(set(y_lab_list)) >= 2:
            X_lab = self.X_pool_vec[list(self._label_dict.keys())]
            self.clf.fit(X_lab, y_lab_list)

        # validate on remaining pool (proxy score)
        remaining = [i for i in range(len(self.X_pool)) if i not in self.labelled_idx]
        if remaining:
            try:
                X_val = self.X_pool_vec[remaining]
                y_val = self.y_pool[remaining]
                acc = self.clf.score(X_val, y_val)
                self.history_acc.append(acc)
                return acc
            except NotFittedError:
                return None
        return None


def bootstrap_active_learning(
    df: pd.DataFrame, encoder: BaseEncoder, clf: Any, settings: Bunch
) -> ActiveSession:
    """
    df must have columns ['text','label']; encoder+clf are *fresh* (untrained).
    Speed defaults:
      - pool_limit: cap pool size (default 10k)
      - candidate_size: subsample evaluated per round (default 4096)
    """
    seed = int(getattr(settings, "seed", 42))
    rng = np.random.default_rng(seed)

    # Optional pool cap for speed
    pool_limit = int(getattr(settings, "pool_limit", 10_000))
    idx = np.arange(len(df))
    if len(idx) > pool_limit:
        idx = rng.choice(idx, size=pool_limit, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    pool_texts = df.text.tolist()
    pool_labels = df.label.to_numpy()

    # Fit representation on the pool once
    X_pool_vec = encoder.fit_transform(pool_texts)

    sess = ActiveSession(
        X_pool=pool_texts,
        y_pool=pool_labels,
        encoder=encoder,
        clf=clf,
        strategy=getattr(settings, "strategy", "entropy"),
        budget=int(getattr(settings, "budget", 20)),
        batch_size=int(getattr(settings, "batch_size", 1)),
        X_pool_vec=X_pool_vec,
        candidate_size=int(getattr(settings, "candidate_size", 4096)),
    )
    return sess
