import json
from typing import Dict, List

from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count

from .models import Movie, Interaction, UserSession
from .active_learning import (
    load_item_factors, load_popularity, solve_user_vector, select_next_movie, influence_preview
)

# cache globals (simple; restart server to refresh factors)
V, MLID2IDX = load_item_factors()
POP = load_popularity()# views.py (top-level)
STOP_AFTER = 10  # progress target

def _get_or_create_session(request):
    if not request.session.session_key:
        request.session.save()
    sk = request.session.session_key
    user, _ = UserSession.objects.get_or_create(session_key=sk)
    return user

def landing(request):
    return render(request, "project4/landing.html")

def study(request):
    user = _get_or_create_session(request)
    return render(request, "project4/study.html", {"session_key": user.session_key})

def _collect_user_ratings(user) -> Dict[int, float]:
    return {it.movie.ml_id: float(it.rating) for it in user.interactions.select_related("movie")}

def _all_movie_ids() -> List[int]:
    return list(Movie.objects.values_list("ml_id", flat=True))

@csrf_exempt
def api_next(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    user = _get_or_create_session(request)
    user_ratings = _collect_user_ratings(user)

    # progress gate
    if len(user_ratings) >= STOP_AFTER:
        return JsonResponse({"done": True, "progress": {"answered": len(user_ratings), "total": STOP_AFTER}})

    all_ids = _all_movie_ids()
    u = None
    if V is not None and MLID2IDX is not None:
        u = solve_user_vector(user_ratings, V, MLID2IDX, lam=1.0)

    mid = select_next_movie(set(user_ratings.keys()), all_ids, V, MLID2IDX, u, POP, strategy="uncertainty")
    if mid is None:
        return JsonResponse({"done": True, "progress": {"answered": len(user_ratings), "total": STOP_AFTER}})

    movie = Movie.objects.get(ml_id=mid)

    before_low, after_low, diff_low = influence_preview(user_ratings, mid, 1.0, V, MLID2IDX, all_ids, topn=5)
    before_high, after_high, diff_high = influence_preview(user_ratings, mid, 5.0, V, MLID2IDX, all_ids, topn=5)

    def _pack_list(ids):
        meta = Movie.objects.filter(ml_id__in=ids).values("ml_id", "title")
        d = {m["ml_id"]: m["title"] for m in meta}
        return [{"ml_id": i, "title": d.get(i, str(i))} for i in ids]

    # NEW: pack diffs with movement (delta <0 means moved up, >0 down)
    def _pack_diff(diff, top_before, top_after):
        titles = Movie.objects.filter(ml_id__in=set(top_before+top_after)).values("ml_id", "title")
        tmap = {m["ml_id"]: m["title"] for m in titles}
        packed = []
        for mid_, i, j in diff:
            if j is None:
                packed.append({
                    "ml_id": mid_, "title": tmap.get(mid_, str(mid_)),
                    "from": i, "to": None, "delta": None, "status": "dropped"
                })
            else:
                delta = j - i  # negative is up
                packed.append({
                    "ml_id": mid_, "title": tmap.get(mid_, str(mid_)),
                    "from": i, "to": j, "delta": delta, "status": "moved"
                })
        # mark any NEW items that appeared in after but not before
        before_set = set(top_before)
        for pos, mid_new in enumerate(top_after):
            if mid_new not in before_set:
                packed.append({
                    "ml_id": mid_new, "title": tmap.get(mid_new, str(mid_new)),
                    "from": None, "to": pos, "delta": None, "status": "new"
                })
        return packed

    low_diff_packed  = _pack_diff(diff_low,  before_low,  after_low)
    high_diff_packed = _pack_diff(diff_high, before_high, after_high)

    return JsonResponse({
        "done": False,
        "progress": {"answered": len(user_ratings), "total": STOP_AFTER},
        "movie": {"ml_id": movie.ml_id, "title": movie.title, "genres": movie.genres, "year": movie.year},
        "influence": {
            "low": {
                "before": _pack_list(before_low),
                "after":  _pack_list(after_low),
                "diff":   low_diff_packed
            },
            "high": {
                "before": _pack_list(before_high),
                "after":  _pack_list(after_high),
                "diff":   high_diff_packed
            }
        }
    })


@csrf_exempt
def api_rate(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    try:
        payload = json.loads(request.body.decode("utf-8"))
        ml_id = int(payload["ml_id"])
        rating = float(payload["rating"])
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    user = _get_or_create_session(request)
    try:
        movie = Movie.objects.get(ml_id=ml_id)
    except Movie.DoesNotExist:
        return HttpResponseBadRequest("Unknown movie")

    Interaction.objects.update_or_create(
        user=user, movie=movie, defaults={"rating": rating}
    )
    return JsonResponse({"ok": True})

def api_summary(request):
    """Return current top-10 recs for this user."""
    user = _get_or_create_session(request)
    ratings = _collect_user_ratings(user)
    if V is None or MLID2IDX is None or not ratings:
        # popularity fallback
        ids = _all_movie_ids()
        ids.sort(key=lambda mid: POP.get(mid, 0), reverse=True)
        ids = [m for m in ids if m not in ratings][:10]
        movies = list(Movie.objects.filter(ml_id__in=ids).values("ml_id", "title"))
        return JsonResponse({"method": "popularity", "top10": movies})

    u = solve_user_vector(ratings, V, MLID2IDX, lam=1.0)
    # rank unseen
    idx = [MLID2IDX.get(m) for m in _all_movie_ids()]
    valid = [(m, i) for m, i in zip(_all_movie_ids(), idx) if i is not None and m not in ratings]
    scores = [(m, float(V[i] @ u)) for (m, i) in valid]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [m for m, _ in scores[:10]]
    movies = list(Movie.objects.filter(ml_id__in=top).values("ml_id", "title"))
    return JsonResponse({"method": "mf", "top10": movies})
