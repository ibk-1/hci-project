from __future__ import annotations

import pathlib
import pickle
from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from .forms import TrainTreeForm, CounterfactualForm
from . import utils
from .utils import counterfactual_search

MEDIA_DIR = pathlib.Path(settings.MEDIA_ROOT) / "project3"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def index(request):
    form = TrainTreeForm()
    head_rows, head_cols = utils.penguins_head()
    return render(
        request,
        "project3/index.html",
        {"form": form, "head_rows": head_rows, "head_cols": head_cols},
    )


def train_tree(request):
    if request.method != "POST":
        raise Http404()
    form = TrainTreeForm(request.POST)
    if not form.is_valid():
        return render(request, "project3/_metrics.html", {"error": "Invalid form"})

    cfg = form.cleaned_data
    lam_val = float(cfg["lam"])
    model = cfg["model"]

    if model == "tree":
        res = utils.train_tree(
            max_depth=int(cfg["max_depth"]),
            lam=lam_val,
            media_dir=MEDIA_DIR,
        )
        res["model"] = "tree"
        _store_model_in_session(request, pipe=res["pipeline"], cfg=cfg)
        del res["pipeline"]
    else:  # sparse logistic regression
        res = utils.train_logreg(lam=lam_val, media_dir=MEDIA_DIR)
        _store_model_in_session(request, pipe=res["pipeline"], cfg=cfg)
        del res["pipeline"]

    return render(request, "project3/_metrics.html", res)


def _store_model_in_session(request, pipe, cfg):
    request.session["p3_cache"] = pickle.dumps({"pipe": pipe, "cfg": cfg}).hex()


def _load_model_from_session(request):
    blob = request.session.get("p3_cache")
    if not blob:
        return None
    return pickle.loads(bytes.fromhex(blob))


def counterfactual(request):
    model_blob = _load_model_from_session(request)
    if not model_blob:
        return render(
            request, "project3/_cf_error.html", {"msg": "Train a model first."}
        )

    df = utils.get_penguins_df().dropna()

    if request.method == "GET":
        form = CounterfactualForm(df=df)
        return render(request, "project3/_cf_form.html", {"form": form})

    # POST
    form = CounterfactualForm(request.POST, df=df)
    if not form.is_valid():
        return render(request, "project3/_cf_error.html", {"msg": "Invalid input."})

    src_idx = int(form.cleaned_data["source_id"])
    target = form.cleaned_data["target"]
    pipe = model_blob["pipe"]

    cf_df = counterfactual_search(
        pipeline=pipe, df=df, source_idx=src_idx, target_label=target, k=3
    )
    if cf_df.empty:
        return render(
            request,
            "project3/_cf_error.html",
            {"msg": "No counterfactual found in the search budget."},
        )

    cf_df.insert(1, "prediction", pipe.predict(cf_df))
    orig = cf_df.iloc[0]

    header_cells = "".join(f"<th>{col}</th>" for col in cf_df.columns)
    rows_html = []
    for idx, row in cf_df.iterrows():
        cell_html = []
        for col, val in row.items():
            if idx == cf_df.index[0]:  # Original row
                cell_html.append(f"<td>{val}</td>")
                continue
            base = orig[col]
            if val == base:
                cell_html.append(f"<td>{val}</td>")
            else:
                if isinstance(val, (int, float)) and isinstance(base, (int, float)):
                    up = val > base
                    cls = "table-success" if up else "table-danger"
                    arrow = "↑" if up else "↓"
                    cell_html.append(f"<td class='{cls}'>{val} {arrow}</td>")
                else:  # categorical change
                    cell_html.append(f"<td class='table-warning'>{val}</td>")
        rows_html.append(f"<tr><th>{idx}</th>{''.join(cell_html)}</tr>")

    cf_html = (
        "<table class='table table-sm table-bordered'>"
        f"<thead><tr><th>Row</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )
    return render(request, "project3/_cf_table.html", {"cf_html": cf_html})
