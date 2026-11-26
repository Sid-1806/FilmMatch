import numpy as np
from scipy.special import expit

from filmmatch.neural import NeuralRanker


def test_exported_network_pools_ratings_and_ranks():
    ranker = NeuralRanker(
        {
            "movie_ids": np.array([11, 12]),
            "input": np.array([[0.0], [2.0], [-1.0]]),
            "hidden_w": np.array([[1.0]]),
            "hidden_b": np.array([0.0]),
            "out_w": np.array([[1.0]]),
            "out_b": np.array([0.0]),
            "items": np.array([[1.0], [-1.0]]),
            "bias": np.array([0.0, 0.0]),
        }
    )
    assert ranker.score({11: 3}) is None
    assert ranker.score({99: 5}) is None
    assert np.allclose(ranker.score({11: 5}), 2 * expit([2.0, -2.0]) - 1)
    assert np.allclose(ranker.score({11: 5, 12: 1}), 2 * expit([1.5, -1.5]) - 1)
