import gzip
import json
import unicodedata
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    rating_count: int = 0


def load_model_json(path: Path) -> dict | list:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(
        path.with_suffix(path.suffix + ".gz"), "rt", encoding="utf-8"
    ) as stream:
        return json.load(stream)


def search_key(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )


def item_neighbors(
    likes: sparse.csr_matrix, neighbors: int, shrinkage: float
) -> sparse.csr_matrix:
    """Rows are liked reference movies and columns are recommendation candidates."""
    by_item = likes.T.tocsr()
    counts = np.asarray(by_item.sum(axis=1)).ravel()
    row_offsets = [0]
    columns: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start in range(0, by_item.shape[0], 128):
        overlap = (by_item[start : start + 128] @ likes).tocsr()
        for local_row in range(overlap.shape[0]):
            row = start + local_row
            left, right = overlap.indptr[local_row : local_row + 2]
            ids = overlap.indices[left:right]
            shared = overlap.data[left:right]
            keep = (ids != row) & (shared > 0)
            ids, shared = ids[keep], shared[keep]
            cosine = shared / np.sqrt(counts[row] * counts[ids])
            similarity = cosine * shared / (shared + shrinkage)
            best = np.lexsort((ids, -similarity))[:neighbors]
            columns.append(ids[best])
            weights.append(similarity[best].astype(np.float32))
            row_offsets.append(row_offsets[-1] + len(best))
    return sparse.csr_matrix(
        (np.concatenate(weights), np.concatenate(columns), row_offsets),
        shape=(len(counts), len(counts)),
    )


class Recommender:
    def __init__(self, movies: list[Movie], neighbors: sparse.csr_matrix):
        if not movies or neighbors.shape != (len(movies), len(movies)):
            raise ValueError(
                "The catalog and similarity artifact must have matching dimensions."
            )
        self.movies = movies
        self.index = {movie.id: i for i, movie in enumerate(movies)}
        self.neighbors = neighbors
        self.rating_counts = np.asarray([movie.rating_count for movie in movies])
        self.popular = np.array(
            sorted(
                range(len(movies)),
                key=lambda i: (-movies[i].rating_count, movies[i].id),
            )
        )

    @classmethod
    def load(cls, directory: Path) -> "Recommender":
        records = load_model_json(directory / "movies.json")
        movies = [Movie(**{**row, "genres": tuple(row["genres"])}) for row in records]
        return cls(movies, sparse.load_npz(directory / "neighbors.npz").tocsr())

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "movies.json").write_text(
            json.dumps([asdict(movie) for movie in self.movies], ensure_ascii=False),
            encoding="utf-8",
        )
        sparse.save_npz(directory / "neighbors.npz", self.neighbors)

    def prompts(self, excluded: set[int], limit: int = 5) -> list[Movie]:
        eligible = (
            self.movies[i]
            for i in self.popular
            if self.movies[i].rating_count and self.movies[i].id not in excluded
        )
        return list(islice(eligible, limit))

    def score(self, ratings: dict[int, float]) -> tuple[np.ndarray, dict[int, float]]:
        scores = np.zeros(len(self.movies), dtype=np.float32)
        strengths = {
            self.index[movie_id]: stars - 3
            for movie_id, stars in ratings.items()
            if movie_id in self.index and stars != 3
        }
        for row, strength in strengths.items():
            left, right = self.neighbors.indptr[row : row + 2]
            scores[self.neighbors.indices[left:right]] += (
                self.neighbors.data[left:right] * strength
            )
        return scores, strengths

    def anchor(
        self, position: int, scores: np.ndarray, strengths: dict[int, float]
    ) -> Movie | None:
        if scores[position] <= 0:
            return None
        contribution, row = max(
            (
                (float(self.neighbors[row, position]) * weight, row)
                for row, weight in strengths.items()
                if weight > 0
            ),
            default=(0, 0),
        )
        return self.movies[row] if contribution > 0 else None

    def recommend(
        self,
        ratings: dict[int, int | float],
        *,
        genre: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 10,
        personalized: bool = True,
    ) -> list[tuple[Movie, Movie | None]]:
        selected = self.popular
        scores, strengths = self.score(ratings if personalized else {})
        if strengths:
            selected = self.popular[np.argsort(-scores[self.popular], kind="stable")]
        result = []
        for idx in selected:
            movie = self.movies[idx]
            if (
                movie.id in ratings
                or (genre and genre not in movie.genres)
                or (
                    year_from is not None
                    and (movie.year is None or movie.year < year_from)
                )
                or (
                    year_to is not None and (movie.year is None or movie.year > year_to)
                )
            ):
                continue
            result.append((movie, self.anchor(idx, scores, strengths)))
            if len(result) == limit:
                break
        return result
