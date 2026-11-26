from pathlib import Path

import numpy as np
from scipy.special import expit


class NeuralRanker:
    """CPU inference for the exported PyTorch network."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights
        self.movie_ids = weights["movie_ids"]
        self.index = {int(movie): row for row, movie in enumerate(self.movie_ids)}

    @classmethod
    def load(cls, path: Path) -> "NeuralRanker":
        with np.load(path, allow_pickle=False) as artifact:
            return cls(dict(artifact.items()))

    def profile_vector(self, ratings: dict[int, float]) -> np.ndarray | None:
        points = [
            (self.index[movie] + 1, rating - 3)
            for movie, rating in ratings.items()
            if movie in self.index and rating != 3
        ]
        if not points:
            return None
        ids, values = zip(*points, strict=True)
        weights = np.asarray(values, dtype=np.float32)
        pooled = (self.weights["input"][list(ids)] * weights[:, None]).sum(
            axis=0
        ) / np.abs(weights).sum()
        hidden = np.maximum(
            0, self.weights["hidden_w"] @ pooled + self.weights["hidden_b"]
        )
        return self.weights["out_w"] @ hidden + self.weights["out_b"]

    def items(
        self, vectors: np.ndarray, positions: np.ndarray | slice = slice(None)
    ) -> np.ndarray:
        return (
            2
            * expit(
                vectors @ self.weights["items"][positions].T
                + self.weights["bias"][positions]
            )
            - 1
        )

    def score(self, ratings: dict[int, float]) -> np.ndarray | None:
        vector = self.profile_vector(ratings)
        return None if vector is None else self.items(vector)
