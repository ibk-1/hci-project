from __future__ import annotations

import json
import pathlib
import pickle
import uuid
import zlib

from django.conf import settings
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from . import utils
from .forms import ALSettingsForm, TrainFullForm, UploadDatasetForm
from .utils import ActiveSession, ENCODERS, bootstrap_active_learning, load_dataset

MEDIA_DIR = pathlib.Path(settings.MEDIA_ROOT) / "project2"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = pathlib.Path(settings.BASE_DIR) / "data"
DEFAULT_DATASET = DATA_DIR / "imdb.csv"

def _ensure_default_dataset(request) -> pathlib.Path:
    """
    Ensure we have a dataset in session. If none uploaded yet,
    fall back to the built-in IMDB dataset under data/imdb.csv.
    """
    ds_path = _get_dataset_path(request)
    if ds_path and ds_path.exists():
        return ds_path
    if DEFAULT_DATASET.exists():
        print("Using default dataset")
        request.session["dataset_relpath"] = f"../data/{DEFAULT_DATASET.name}"
        return DEFAULT_DATASET
    return None


def _get_dataset_path(request):
    rel = request.session.get("dataset_relpath")
    if rel:
        p = MEDIA_DIR / rel
        if p.exists():
            return p
        # allow "../data/imdb.csv"
        p2 = pathlib.Path(settings.BASE_DIR) / rel
        if p2.exists():
            return p2
    return None



def _dump_session(obj) -> str:
    """Compressed hex to reduce session size."""
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).hex()


def _load_session(hexstr):
    return pickle.loads(zlib.decompress(bytes.fromhex(hexstr)))


def _new_encoder(rep="tfidf"):
    return ENCODERS[rep]()


def _new_clf(name="logreg"):


    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            solver="liblinear",  # supports sparse, binary OvR
            warm_start=True,
        )
    if name == "svm":
        return LinearSVC()
    raise ValueError(f"Unknown classifier: {name}")


# ------------------------------- views ----------------------------- #
def index(request):
    run = request.session.get("last_run")
    form = TrainFullForm()
    upform = UploadDatasetForm()
    ds_path = _ensure_default_dataset(request)
    head = utils.dataset_head(ds_path) if ds_path else None
    report = run.get("report") if run else None
    return render(
        request,
        "project2/index.html",
        {
            "form": form,
            "upform": upform,
            "run": run,
            "report": report,
            "head": head,
            "ds_path": ds_path,
        },
    )


def train_full(request):
    if request.method != "POST":
        raise Http404()

    form = TrainFullForm(request.POST)
    if not form.is_valid():
        ds_path = _ensure_default_dataset(request)
        head = utils.dataset_head(ds_path) if ds_path else []
        return render(
            request,
            "project2/index.html",
            {"form": form, "upform": UploadDatasetForm(), "head": head},
        )

    cfg = form.cleaned_data
    cache_dir = MEDIA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = f'{cfg["representation"]}_{cfg["classifier"]}.json'
    manifest = cache_dir / cache_key

    ds_path = _ensure_default_dataset(request)
    if not ds_path:
        return render(
            request,
            "project2/index.html",
            {
                "form": form,
                "upform": UploadDatasetForm(),
                "error": "Please upload a dataset first.",
            },
        )

    # -- load from cache or train fresh --------------------
    report = None
    if cfg.get("use_cached") and manifest.exists():
        cached = json.loads(manifest.read_text())
        acc = cached.get("accuracy")
        report = cached.get("report")
    else:
        job_id = uuid.uuid4()
        acc, report, _ = utils.train_full_pipeline(
            cfg, job_id, MEDIA_DIR, csv_path=ds_path
        )
        manifest.write_text(json.dumps({"accuracy": acc, "report": report}))

    request.session["last_run"] = {"accuracy": acc, "report": report}

    # -- HTMX: return partial ------------------------------
    if request.headers.get("HX-Request") == "true":
        return render(
            request, "project2/_train_result.html", {"accuracy": acc, "report": report}
        )

    # classic fallback
    return redirect("project2:index")


def train_status(request, job_id):
    # kept for future async jobs
    return JsonResponse({"state": "finished", "progress": 100})


def upload_dataset(request):
    """POST target for the file-upload form (HTMX or classic)."""
    if request.method != "POST":
        raise Http404()
    form = UploadDatasetForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(
            request,
            "project2/_dataset_form.html",
            {"form": form, "error": "Invalid file."},
        )

    f = form.cleaned_data["csv_file"]
    dest = MEDIA_DIR / "datasets"
    dest.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4()}.csv"
    with open(dest / filename, "wb+") as out:
        for chunk in f.chunks():
            out.write(chunk)

    # remember in the session so future requests see it
    request.session["dataset_relpath"] = f"datasets/{filename}"

    # If HTMX, return updated preview table; otherwise redirect
    if request.headers.get("HX-Request") == "true":
        head = utils.dataset_head(dest / filename)
        return render(request, "project2/_dataset_preview.html", {"head": head})
    return redirect("project2:index")


def active_learning(request):
    # ───── 1) GET: show settings ─────
    if request.method == "GET":
        upform = UploadDatasetForm()
        ds_path = _ensure_default_dataset(request)
        if not ds_path:
            return render(
                request,
                "project2/active_learning.html",
                {"error": "Upload a dataset first.", "upform": upform},
            )
        settings_form = ALSettingsForm()
        return render(
            request, "project2/active_learning.html", {"settings_form": settings_form}
        )

    # ───── 2) POST: start new session ─────
    if "start_session" in request.POST:
        form = ALSettingsForm(request.POST)
        if not form.is_valid():
            return render(
                request, "project2/_al_error.html", {"msg": "Invalid settings."}
            )

        params = form.cleaned_data
        df = load_dataset(_ensure_default_dataset(request))

        # Use same family as Task-1, but fresh models (per brief)
        enc = _new_encoder(params.get("representation", "tfidf"))
        clf = _new_clf(params.get("classifier", "logreg"))

        # bootstrap session; utils sets sane defaults for speed (candidate_size, pool_limit)
        sess = bootstrap_active_learning(df, enc, clf, utils.Bunch(**params))
        request.session["al_state"] = _dump_session(sess)

        q_idx = sess.query_indices()[0]
        return render(
            request,
            "project2/_al_query.html",
            {"query": sess.X_pool[q_idx], "idx": q_idx, "labelled": 0, "acc": "—"},
        )

    # ───── 3) POST: label a queried item ─────
    if "label" in request.POST:
        try:
            sess: ActiveSession = _load_session(request.session["al_state"])
        except Exception:
            return render(
                request, "project2/_al_error.html", {"msg": "Session expired."}
            )

        idx = int(request.POST["idx"])
        label = int(request.POST["label"])
        acc = sess.step([idx], [label])
        request.session["al_state"] = _dump_session(sess)

        acc_display = "—" if acc is None else f"{acc:.3f}"

        if len(sess.labelled_idx) >= sess.budget:
            return render(
                request,
                "project2/_al_done.html",
                {"labelled": len(sess.labelled_idx), "acc": acc_display},
            )

        next_idx = sess.query_indices()[0]
        return render(
            request,
            "project2/_al_query.html",
            {
                "query": sess.X_pool[next_idx],
                "idx": next_idx,
                "labelled": len(sess.labelled_idx),
                "acc": acc_display,
            },
        )

    raise Http404()
