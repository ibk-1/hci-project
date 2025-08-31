"""
Helpers for Project 3 – Explainable models on Palmer Penguins.
Requires: scikit-learn, pandas, graphviz (optional at runtime)
"""

from __future__ import annotations

import io
import pathlib
from typing import Dict, Any

import numpy as np
import pandas as pd
from django.conf import settings
from palmerpenguins import load_penguins

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# ----------------------------- Data ----------------------------- #
def load_penguins(cache: str = ".cache") -> pd.DataFrame:
    p = pathlib.Path(cache) / "penguins.csv"
    if p.exists():
        return pd.read_csv(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = load_penguins()
    df.to_csv(p, index=False)
    return df


def penguins_head(n: int = 5):
    """Return (rows, columns) for template preview."""
    df = load_penguins().dropna().head(n)
    rows = df.values.tolist()
    cols = df.columns.tolist()
    return rows, cols


# ---------------------- Interpretable models --------------------- #
def train_tree(
    *,
    lam: float,
    max_depth: int = 3,
    media_dir: pathlib.Path = pathlib.Path("media/project3"),
) -> Dict[str, Any]:
    """
    Decision Tree with cost-complexity parameter ccp_alpha = λ.
    Returns accuracy, #leaves, image URL, and the fitted pipeline.
    """
    df = load_penguins().dropna()
    X = df.drop("species", axis=1)
    y = df["species"]

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    clf = DecisionTreeClassifier(max_depth=int(max_depth), ccp_alpha=float(lam), random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))

    # Use the FITTED estimator from the pipeline
    fitted_clf: DecisionTreeClassifier = pipe.named_steps["clf"]
    n_leaves = fitted_clf.get_n_leaves()

    # ---------- Export tree image (cache-busted filename) ----------
    media_dir.mkdir(parents=True, exist_ok=True)
    img_name = f"tree_d{int(max_depth)}_lam{float(lam):.6g}.png"
    img_path = media_dir / img_name
    feature_names = pipe[:-1].get_feature_names_out()
    class_names = [str(c) for c in fitted_clf.classes_]

    try:
        import graphviz  # requires Graphviz installed on system

        dot = io.StringIO()
        export_graphviz(
            fitted_clf,
            out_file=dot,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            impurity=False,
        )
        png_bytes = graphviz.Source(dot.getvalue()).pipe(format="png")
        img_path.write_bytes(png_bytes)
    except Exception:
        # Fallback to Matplotlib
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree

        plt.figure(figsize=(14, 8))
        plot_tree(
            fitted_clf,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            impurity=False,
        )
        plt.tight_layout()
        plt.savefig(img_path, dpi=120)
        plt.close()

    img_url = settings.MEDIA_URL + f"project3/{img_name}"

    return {
        "accuracy": acc,
        "leaves": n_leaves,
        "img_url": img_url,
        "depth": int(max_depth),
        "lam": float(lam),
        "pipeline": pipe,
        "model": "tree",
    }


def train_logreg(*, lam: float, media_dir: pathlib.Path):
    """
    Multinomial logistic regression with sparsity via L1 (λ → C).
    lam = 0 ⇒ effectively L2.
    """
    df = load_penguins().dropna()
    X = df.drop("species", axis=1)
    y = df["species"]

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if lam == 0:
        C = 1e6  # practically unregularized
        penalty = "l2"
    else:
        C = 1.0 / float(lam)
        penalty = "l1"

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="saga",
        multi_class="multinomial",
        max_iter=1000,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))

    # number of non-zero coefficients (sum over classes) with small threshold
    coef = clf.coef_.ravel()
    n_nonzero = int((np.abs(coef) > 1e-8).sum())

    return {
        "accuracy": acc,
        "nonzero": n_nonzero,
        "lam": float(lam),
        "model": "logreg",
        "pipeline": pipe,
    }


# ----------------------- Counterfactual search ---------------------- #
def counterfactual_search(
    *,
    pipeline,
    df,
    source_idx: int,
    target_label: str,
    k: int = 3,
    n_samples: int = 2000,
    random_state=42,
):
    """
    Counterfactuals via local sampling:
      1) sample around x using per-feature MAD (median absolute deviation)
      2) keep samples predicted as target
      3) rank by MAD-weighted L1 distance; return top-k
    """
    rng = np.random.default_rng(random_state)

    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes("object").columns

    source_row = df.loc[source_idx:source_idx]  # keep DataFrame

    # --- compute MAD (median abs deviation) per numeric feature ---
    med = df[num_cols].median()
    mad = (df[num_cols] - med).abs().median().values + 1e-6

    # --- generate synthetic numeric perturbations, keep cats fixed ---
    numeric_src = source_row[num_cols].values.astype(float)[0]
    candidates = []
    while len(candidates) < n_samples:
        noise = rng.normal(0, mad, size=numeric_src.shape)
        numeric_new = numeric_src + noise
        synth_row = source_row.copy()
        synth_row[num_cols] = numeric_new
        candidates.append(synth_row)

    synth_df = pd.concat(candidates, ignore_index=True)

    # --- keep those classified as target ---
    preds = pipeline.predict(synth_df)
    keep_mask = preds == target_label
    if keep_mask.sum() == 0:
        return pd.DataFrame()

    kept = synth_df[keep_mask].copy()

    # --- rank by MAD-weighted L1 distance ---
    distances = (
        pairwise_distances(kept[num_cols], source_row[num_cols], metric="manhattan").ravel()
        / (np.abs(mad).sum())
    )
    kept["dist"] = distances
    kept = kept.sort_values("dist").head(k).drop(columns=["dist"])

    kept.index = [f"CF{i+1}" for i in range(len(kept))]
    src_labeled = source_row.copy()
    src_labeled.index = ["Original"]
    return pd.concat([src_labeled, kept])
