import csv
import json
import os
import tempfile
import zipfile
import urllib.request
import numpy as np
from django.core.management.base import BaseCommand

DEFAULT_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

class Command(BaseCommand):
    help = "Train quick item factors V using truncated SVD on MovieLens ratings and save to project4/data/."

    def add_arguments(self, parser):
        parser.add_argument("--k", type=int, default=32, help="Latent dim (default 32)")
        parser.add_argument("--url", default=DEFAULT_URL,
                            help="ZIP URL (ignored if --path is given)")
        parser.add_argument("--path", default=None,
                            help="Path to unzipped ml-latest-small (with ratings.csv). If given, zip is not downloaded.")

    def handle(self, *args, **opts):
        K = int(opts["k"])
        path = opts["path"]
        url = opts["url"]

        # ---- Load ratings.csv (download if needed) ----
        if path:
            ratings_csv = os.path.join(path, "ratings.csv")
            if not os.path.exists(ratings_csv):
                self.stderr.write("ratings.csv not found in --path")
                return
        else:
            self.stdout.write(f"Downloading MovieLens from: {url}")
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "ml.zip")
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)
                ratings_csv = None
                for root, _, files in os.walk(tmpdir):
                    if "ratings.csv" in files:
                        ratings_csv = os.path.join(root, "ratings.csv")
                        break
                if ratings_csv is None:
                    self.stderr.write("ratings.csv not found in ZIP")
                    return
                # We will copy data we need from the CSV, so no need to persist extracted tree outside ctx
                # Instead, we will parse immediately below.

                # But we can't return here because file will disappear after with-block.
                # So read rows now:
                rows = []
                with open(ratings_csv, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rows.append((int(row["userId"]), int(row["movieId"]), float(row["rating"])))
        # If we had a local path, read rows now:
        if path:
            rows = []
            with open(ratings_csv, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append((int(row["userId"]), int(row["movieId"]), float(row["rating"])))

        # ---- Index users and movies ----
        self.stdout.write("Indexing users/movies...")
        users = sorted({u for (u, m, r) in rows})
        movies = sorted({m for (u, m, r) in rows})
        uid2i = {u:i for i, u in enumerate(users)}
        mid2j = {m:j for j, m in enumerate(movies)}

        n_users = len(users)
        n_items = len(movies)
        self.stdout.write(f"Users: {n_users}  Items: {n_items}")

        # ---- Build dense ratings (small dataset -> fine) ----
        self.stdout.write("Building dense matrix...")
        R = np.zeros((n_users, n_items), dtype=np.float32)
        for (u, m, r) in rows:
            R[uid2i[u], mid2j[m]] = r

        # subtract user means (simple centering improves SVD)
        user_means = np.divide(R.sum(1), (R != 0).sum(1, keepdims=False), out=np.zeros(n_users), where=(R != 0).sum(1)>0)
        R_centered = R - user_means[:, None] * (R != 0)

        # ---- Truncated SVD (numpy) ----
        # Full SVD then truncate (OK for this small dataset).
        self.stdout.write("Computing SVD...")
        U, S, VT = np.linalg.svd(R_centered, full_matrices=False)
        k = min(K, len(S))
        Uk = U[:, :k]
        Sk = S[:k]
        VTk = VT[:k, :]   # shape: k x n_items

        # Represent item factors V as (n_items x k).
        # One common choice: V = (VTk.T) * sqrt(Sk)
        V = (VTk.T * np.sqrt(Sk))

        # ---- Save outputs ----
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        os.makedirs(data_dir, exist_ok=True)

        v_path = os.path.join(data_dir, "item_factors.npy")
        map_path = os.path.join(data_dir, "mlid_to_index.json")

        np.save(v_path, V.astype(np.float32))

        # Map MovieLens movieId -> column index j
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump({int(mid): int(j) for mid, j in mid2j.items()}, f)

        self.stdout.write(self.style.SUCCESS(
            f"Saved V to {v_path} (shape {V.shape}) and map to {map_path}"
        ))
