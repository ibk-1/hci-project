import csv
import io
import os
import tempfile
import zipfile
import urllib.request
from django.core.management.base import BaseCommand
from django.db import transaction
from project4.models import Movie

DEFAULT_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

class Command(BaseCommand):
    help = "Download MovieLens (small) ZIP, extract, import movies, and build popularity counts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            default=DEFAULT_URL,
            help="URL to the MovieLens ZIP (default: ml-latest-small.zip from GroupLens).",
        )
        parser.add_argument(
            "--path",
            default=None,
            help="Optional local folder (already unzipped) containing movies.csv and ratings.csv. If provided, --path is used and --url is ignored.",
        )

    def handle(self, *args, **opts):
        path = opts["path"]
        url = opts["url"]

        if path:
            self.stdout.write(self.style.WARNING("Using local --path, ignoring --url"))
            movies_csv = os.path.join(path, "movies.csv")
            ratings_csv = os.path.join(path, "ratings.csv")
            if not (os.path.exists(movies_csv) and os.path.exists(ratings_csv)):
                self.stderr.write("movies.csv/ratings.csv not found in --path")
                return
            self._import_from_csvs(movies_csv, ratings_csv)
            return

        # Download + extract from URL
        self.stdout.write(f"Downloading ZIP from: {url}")
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "ml.zip")
            urllib.request.urlretrieve(url, zip_path)  # stdlib; no extra deps

            self.stdout.write("Extracting ZIP...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            # Typical structure: {tmpdir}/ml-latest-small/movies.csv
            # Find the first movies.csv & ratings.csv in the extracted tree
            movies_csv, ratings_csv = None, None
            for root, _, files in os.walk(tmpdir):
                if "movies.csv" in files and "ratings.csv" in files:
                    movies_csv = os.path.join(root, "movies.csv")
                    ratings_csv = os.path.join(root, "ratings.csv")
                    break

            if not (movies_csv and ratings_csv):
                self.stderr.write("Could not locate movies.csv/ratings.csv in the ZIP")
                return

            self._import_from_csvs(movies_csv, ratings_csv)

    # ---------------- internal helpers ----------------

    def _import_from_csvs(self, movies_csv: str, ratings_csv: str):
        self.stdout.write("Importing movies...")
        with open(movies_csv, "r", encoding="utf-8") as f, transaction.atomic():
            r = csv.DictReader(f)
            Movie.objects.all().delete()
            for row in r:
                movieId = int(row["movieId"])
                title = row["title"]
                genres = row.get("genres", "")
                # crude year extraction if title like "Toy Story (1995)"
                year = None
                if "(" in title and ")" in title:
                    tail = title.rsplit("(", 1)[-1].strip(")")
                    try:
                        year = int(tail)
                    except ValueError:
                        year = None
                Movie.objects.create(ml_id=movieId, title=title, year=year, genres=genres)

        # Popularity counts (how many ratings each movie received)
        from collections import Counter
        pop = Counter()
        self.stdout.write("Computing popularity...")
        with open(ratings_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    pop[int(row["movieId"])] += 1
                except Exception:
                    continue

        # Save popularity JSON in project4/data/
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        import json
        with open(os.path.join(data_dir, "movie_popularity.json"), "w", encoding="utf-8") as f:
            json.dump(pop, f)

        self.stdout.write(self.style.SUCCESS("Movies imported and popularity saved."))
