import csv
import json
import re
from array import array
from dataclasses import replace
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import sparse

from .model import Movie, Recommender, item_neighbors

SEED = 42
NEIGHBORS = 100
SHRINKAGE = 10.0
PROMPTS = 5


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        from hashlib import file_digest

        return file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def stable_order(user: int, seed: int, purpose: str) -> bytes:
    return sha256(f"{seed}:{purpose}:{user}".encode()).digest()


def load_data(
    archive: Path, sample_size: int, seed: int
) -> tuple[list[Movie], np.ndarray]:
    with ZipFile(archive) as source:
        members = {Path(name).name: name for name in source.namelist()}
        with TextIOWrapper(
            source.open(members["movies.csv"]), encoding="utf-8-sig"
        ) as stream:
            movies = []
            for row in csv.DictReader(stream):
                title = row["title"]
                year = re.search(r"\s\((\d{4})\)\s*$", title)
                genres = tuple(
                    genre
                    for genre in row["genres"].split("|")
                    if genre != "(no genres listed)"
                )
                movies.append(
                    Movie(
                        int(row["movieId"]),
                        title[: year.start()] if year else title,
                        int(year.group(1)) if year else None,
                        genres,
                    )
                )
        ratings_member = members["ratings.csv"]
        with TextIOWrapper(source.open(ratings_member), encoding="utf-8-sig") as stream:
            rows = csv.reader(stream)
            next(rows)
            users = {int(row[0]) for row in rows}
        sampled = set(
            sorted(users, key=lambda user: stable_order(user, seed, "sample"))[
                :sample_size
            ]
        )
        values = array("q")
        with TextIOWrapper(source.open(ratings_member), encoding="utf-8-sig") as stream:
            rows = csv.reader(stream)
            next(rows)
            for user, movie, rating, timestamp in rows:
                if int(user) in sampled:
                    values.extend(
                        (int(user), int(movie), int(float(rating) * 2), int(timestamp))
                    )
    ratings = np.frombuffer(values, dtype=np.int64).reshape(-1, 4)
    return sorted(movies, key=lambda movie: movie.id), ratings


def freeze(ratings: np.ndarray, data_hash: str, size: int, seed: int) -> dict:
    users = sorted(
        map(int, np.unique(ratings[:, 0])),
        key=lambda user: stable_order(user, seed, "cohort"),
    )
    if len(users) < 10:
        raise ValueError("The benchmark requires at least ten users.")
    fit_end = int(len(users) * 0.8)
    validation_end = int(len(users) * 0.9)
    timestamps = np.sort(ratings[:, 3])
    cutoffs = [
        int(timestamps[int(len(timestamps) * fraction)]) for fraction in (0.8, 0.9)
    ]
    if cutoffs[0] == cutoffs[1]:
        raise ValueError("Validation and test windows overlap at the same timestamp.")
    return {
        "archive_sha256": data_hash,
        "seed": seed,
        "sample_limit": size,
        "cutoffs": cutoffs,
        "fit_users": sorted(users[:fit_end]),
        "validation_users": sorted(users[fit_end:validation_end]),
        "test_users": sorted(users[validation_end:]),
    }


def train(movies: list[Movie], ratings: np.ndarray, split: dict) -> Recommender:
    fit = ratings[
        (ratings[:, 3] < split["cutoffs"][0])
        & np.isin(ratings[:, 0], split["fit_users"])
    ]
    if not len(fit):
        raise ValueError("No eligible fit interactions before the training cutoff.")
    index = {movie.id: pos for pos, movie in enumerate(movies)}
    columns = np.fromiter(
        (index[int(movie)] for movie in fit[:, 1]), dtype=np.int32, count=len(fit)
    )
    counts = np.bincount(columns, minlength=len(movies))
    movies = [
        replace(movie, rating_count=int(count))
        for movie, count in zip(movies, counts, strict=True)
    ]
    _, users = np.unique(fit[:, 0], return_inverse=True)
    liked = fit[:, 2] >= 8
    likes = sparse.csr_matrix(
        (np.ones(liked.sum(), dtype=np.float32), (users[liked], columns[liked])),
        shape=(int(users.max()) + 1, len(movies)),
    )
    likes.data.fill(1)
    return Recommender(movies, item_neighbors(likes, NEIGHBORS, SHRINKAGE))


def ndcg(ids: list[int], relevant: set[int]) -> float:
    ideal = sum(1 / np.log2(i + 2) for i in range(min(10, len(relevant))))
    return float(
        sum(1 / np.log2(i + 2) for i, movie in enumerate(ids[:10]) if movie in relevant)
        / ideal
    )


def evaluate(model: Recommender, ratings: np.ndarray, split: dict) -> dict:
    prompts = [movie.id for movie in model.prompts(set(), PROMPTS)]
    eligible_movies = {movie.id for movie in model.movies if movie.rating_count}
    result = {}
    for cohort, start, end in (
        ("validation", *split["cutoffs"]),
        ("test", split["cutoffs"][1], None),
    ):
        users = split[f"{cohort}_users"]
        subset = ratings[np.isin(ratings[:, 0], users)]
        subset = subset[np.argsort(subset[:, 0], kind="stable")]
        offsets = np.flatnonzero(np.diff(subset[:, 0])) + 1
        scores = []
        for rows in np.split(subset, offsets):
            earlier = sorted(
                (row for row in rows if row[3] < start), key=lambda row: row[3]
            )
            history = {int(row[1]): float(row[2]) / 2 for row in earlier}
            relevant = {
                int(movie)
                for _, movie, rating, timestamp in rows
                if timestamp >= start
                and (end is None or timestamp < end)
                and rating >= 8
                and int(movie) not in history
                and int(movie) in eligible_movies
            }
            if not relevant:
                continue
            answers = {movie: history[movie] for movie in prompts if movie in history}
            outputs = [
                [
                    movie.id
                    for movie, _ in model.recommend(answers, personalized=personalized)
                ]
                for personalized in (False, True)
            ]
            scores.append(
                [ndcg(ids, relevant) for ids in outputs] + [len(prompts) - len(answers)]
            )
        if not scores:
            result[cohort] = {"eligible_users": 0, "assigned_users": len(users)}
            continue
        values = np.array(scores)
        changes = values[:, 1] - values[:, 0]
        rng = np.random.default_rng(split["seed"])
        samples = np.concatenate(
            [
                changes[
                    rng.integers(len(changes), size=(min(100, 2000 - i), len(changes)))
                ].mean(axis=1)
                for i in range(0, 2000, 100)
            ]
        )
        result[cohort] = {
            "eligible_users": len(scores),
            "assigned_users": len(users),
            "popularity_ndcg_at_10": float(values[:, 0].mean()),
            "itemknn_ndcg_at_10": float(values[:, 1].mean()),
            "paired_lift": float(changes.mean()),
            "paired_bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
            "simulated_skip_rate": float(values[:, 2].mean() / len(prompts)),
        }
    return result
