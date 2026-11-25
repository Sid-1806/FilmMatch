import argparse
import json
import re
import shutil
import ssl
import tarfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import certifi
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .model import Movie, Recommender, search_key

CORPUS_URL = "https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
TEXT_WEIGHT = 0.75
ITEM_WEIGHT = 0.65


def title_key(title: str) -> str:
    title = re.sub(r"^(.*), (The|An|A)$", r"\2 \1", title, flags=re.IGNORECASE)
    return " ".join(re.findall(r"\w+", search_key(title)))


def match_descriptions(corpus: Path, movies: list[Movie]) -> dict[int, str]:
    with tarfile.open(corpus, "r:gz") as archive:
        members = {Path(entry.name).name: entry for entry in archive if entry.isfile()}
        by_name = defaultdict(set)
        metadata = archive.extractfile(members["movie.metadata.tsv"])
        for line in metadata:
            wiki, _, title, date, *_ = line.decode("utf-8").rstrip("\n").split("\t")
            if date[:4].isdigit():
                by_name[title_key(title), int(date[:4])].add(wiki)
        wanted = defaultdict(list)
        for movie in movies:
            matches = by_name[title_key(movie.title), movie.year]
            if len(matches) == 1:
                wanted[next(iter(matches))].append(movie.id)
        descriptions = {}
        for line in archive.extractfile(members["plot_summaries.txt"]):
            wiki, text = line.decode("utf-8").rstrip("\n").split("\t", 1)
            if text.strip():
                for movie_id in wanted.get(wiki, []):
                    descriptions[movie_id] = text[:4000]
    return descriptions


class ContentIndex:
    def __init__(self, movies: list[Movie], descriptions: dict[int, str]):
        self.movies = movies
        self.index = {movie.id: position for position, movie in enumerate(movies)}
        self.movie_ids = np.asarray([movie.id for movie in movies])
        self.rating_counts = np.asarray([movie.rating_count for movie in movies])
        self.texts = [descriptions.get(movie.id, "") for movie in movies]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=16000, dtype=np.float32
        )
        self.available = any(self.texts)
        self.vectors = (
            self.vectorizer.fit_transform(self.texts)
            if self.available
            else sparse.csr_matrix((len(movies), 0))
        )
        genre_names = sorted({genre for movie in movies for genre in movie.genres})
        self.genre_index = {genre: i for i, genre in enumerate(genre_names)}
        self.genres = self.genre_vectors(movies)

    def genre_vectors(self, movies: list[Movie]) -> np.ndarray:
        vectors = np.zeros((len(movies), len(self.genre_index)), dtype=np.float32)
        for pos, movie in enumerate(movies):
            ids = [self.genre_index[g] for g in movie.genres if g in self.genre_index]
            if ids:
                vectors[pos, ids] = 1 / np.sqrt(len(ids))
        return vectors

    def extend(self, additions: list[tuple[Movie, str]]) -> "ContentIndex":
        if not additions:
            return self
        movies, texts = map(list, zip(*additions, strict=True))
        result = object.__new__(ContentIndex)
        result.movies = self.movies + movies
        result.index = {movie.id: pos for pos, movie in enumerate(result.movies)}
        result.movie_ids = np.asarray([movie.id for movie in result.movies])
        result.rating_counts = np.asarray(
            [movie.rating_count for movie in result.movies]
        )
        result.texts = self.texts + texts
        result.vectorizer = self.vectorizer
        result.available = self.available
        vectors = (
            self.vectorizer.transform(texts)
            if self.available
            else sparse.csr_matrix((len(texts), 0))
        )
        result.vectors = sparse.vstack((self.vectors, vectors), format="csr")
        result.genre_index = self.genre_index
        result.genres = np.vstack((self.genres, self.genre_vectors(movies)))
        return result

    def score(
        self,
        ratings: dict[int, float],
        *,
        use_text: bool = True,
        positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        liked = [
            (self.index[i], stars - 3)
            for i, stars in ratings.items()
            if stars >= 4 and i in self.index
        ]
        candidates = slice(None) if positions is None else positions
        size = len(self.movies) if positions is None else len(positions)
        if not liked:
            return np.zeros(size, dtype=np.float32), np.full(size, -1, dtype=int)
        ids, weights = map(list, zip(*liked, strict=True))
        similarities = self.genres[candidates] @ self.genres[ids].T
        if use_text and self.available:
            docs = self.vectors[candidates]
            text_sim = (docs @ self.vectors[ids].T).toarray()
            both_have_terms = (docs.getnnz(axis=1)[:, None] > 0) & (
                self.vectors[ids].getnnz(axis=1) > 0
            )
            similarities = np.where(
                both_have_terms,
                (1 - TEXT_WEIGHT) * similarities + TEXT_WEIGHT * text_sim,
                similarities,
            )
        contributions = similarities * np.asarray(weights, dtype=np.float32)
        best = np.asarray(ids)[np.argmax(contributions, axis=1)]
        return contributions.sum(axis=1) / sum(weights), best


def hybrid_picks(
    model: Recommender,
    index: ContentIndex,
    ratings: dict[int, float],
    *,
    genre=None,
    year_from=None,
    year_to=None,
) -> list[dict]:
    item, strengths = model.score(ratings)
    content, anchors = index.score(ratings)
    rated = len(model.movies)
    scores = content.copy()
    scale = sum(abs(weight) for weight in strengths.values()) or 1
    scores[:rated] = np.where(
        model.rating_counts > 0,
        ITEM_WEIGHT * item / scale + (1 - ITEM_WEIGHT) * content[:rated],
        content[:rated],
    )
    ordered = np.lexsort((index.movie_ids, -index.rating_counts, -scores))
    picks = []
    for pos in ordered:
        movie = index.movies[pos]
        if (
            movie.id in ratings
            or (genre and genre not in movie.genres)
            or (
                year_from is not None and (movie.year is None or movie.year < year_from)
            )
            or (year_to is not None and (movie.year is None or movie.year > year_to))
            or (not movie.rating_count and scores[pos] <= 0)
        ):
            continue
        reason = None
        signal = "popular"
        if content[pos] > 0 and (
            not movie.rating_count
            or (1 - ITEM_WEIGHT) * content[pos] >= ITEM_WEIGHT * item[pos] / scale
        ):
            reason, signal = index.movies[anchors[pos]].title, "content"
        elif pos < rated:
            anchor = model.anchor(pos, item, strengths)
            if anchor:
                reason, signal = anchor.title, "ratings"
        picks.append({"movie": movie, "because": reason, "signal": signal})
        if len(picks) == 10:
            break
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the Creative Commons movie summaries"
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/MovieSummaries.tar.gz")
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args()
    if not args.corpus.is_file():
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        temp = args.corpus.with_suffix(".partial")
        with (
            urlopen(
                CORPUS_URL,
                context=ssl.create_default_context(cafile=certifi.where()),
                timeout=90,
            ) as response,
            temp.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        temp.replace(args.corpus)
    model = Recommender.load(args.model)
    descriptions = match_descriptions(args.corpus, model.movies)
    (args.model / "descriptions.json").write_text(
        json.dumps(descriptions, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Matched {len(descriptions)} of {len(model.movies)} films")


if __name__ == "__main__":
    main()
