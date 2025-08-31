import pathlib, tempfile, zipfile, pandas as pd, requests, numpy as np
from surprise import Reader, Dataset, SVD

_DATA_DIR = pathlib.Path("data/project4")
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_MOVIES = _DATA_DIR / "movies.csv"
_RATINGS = _DATA_DIR / "ratings.csv"

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# -------------------------------

def _download_if_needed():
    if _MOVIES.exists():
        return
    z = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    z.write(requests.get(URL, timeout=60).content)
    z.close()
    with zipfile.ZipFile(z.name) as arc:
        for name in ["movies.csv", "ratings.csv"]:
            target = _DATA_DIR / name
            with arc.open(f"ml-latest-small/{name}") as src, target.open("wb") as dst:
                dst.write(src.read())

_download_if_needed()
MOVIES_DF = pd.read_csv(_MOVIES)
RATINGS_DF = pd.read_csv(_RATINGS)

# keep 2 000 random movies for snappy demos
MOVIES_DF = MOVIES_DF.sample(2000, random_state=42).reset_index(drop=True)

# -------------------------------  Recommender class
class Recommender:
    """Matrix‑factorisation (SVD) plus popularity fallback."""

    def __init__(self):
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(RATINGS_DF[["userId", "movieId", "rating"]], reader)
        trainset = data.build_full_trainset()
        self.algo = SVD(n_factors=30, n_epochs=20, random_state=42)
        self.algo.fit(trainset)
        # item popularity mean
        self.pop_mean = RATINGS_DF.groupby("movieId").rating.mean()

    def predict(self, user_ratings, movie_id):
        """Return predicted rating for (cold-start) user on movie_id."""
        uid = "newuser"                          # synthetic user ID
        iid = str(movie_id)

        # surprise's public API → .predict() returns a Prediction object
        est = self.algo.predict(uid, iid).est    # ★ FIX: no r_ui / verbose ★

        # cold-start override: if we actually have a rating the user just gave,
        # return it directly so the algorithm respects it.
        if movie_id in user_ratings:
            est = user_ratings[movie_id]

        # popularity fallback in case algo produces NaN (rare)
        if np.isnan(est):
            est = self.pop_mean.get(movie_id, 3.0)

        return est


    def top_n(self, user_ratings, n=5):
        unseen = set(MOVIES_DF.movieId) - user_ratings.keys()
        preds = [(mid, self.predict(user_ratings, mid)) for mid in unseen]
        preds.sort(key=lambda t: -t[1])
        return preds[:n]

REC = Recommender()

def random_query(asked):
    pool = MOVIES_DF[~MOVIES_DF.movieId.isin(asked)]
    return pool.sample(1, random_state=np.random.randint(0, 1_000_000)).iloc[0]

import numpy as np
def influence_delta(rec, user_ratings, movie_id, sample_ids):
    """
    Estimate how much *relative* the user-vector would move if this movie
    were rated 5 vs 0.5.  We sample 50 candidate items instead of evaluating
    the full catalogue.
    """
    high = {**user_ratings, movie_id: 5.0}
    low  = {**user_ratings, movie_id: 0.5}

    diffs = [
        abs(rec.predict(high, mid) - rec.predict(low, mid))
        for mid in sample_ids
    ]
    return float(np.mean(diffs))