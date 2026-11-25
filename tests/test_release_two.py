import numpy as np
from fastapi.testclient import TestClient
from scipy import sparse

from filmmatch.api import create_app
from filmmatch.content import ContentIndex, hybrid_picks
from filmmatch.model import Movie, Recommender
from filmmatch.policy import next_prompt


def test_adaptive_prompt_prefers_unrated_genres_and_excludes_skips():
    movies = [
        Movie(1, "First", 2000, ("Drama",), 100),
        Movie(2, "Second", 2000, ("Drama",), 90),
        Movie(3, "Third", 2000, ("Comedy",), 80),
        Movie(4, "Fourth", 2000, ("Action",), 70),
    ]
    model = Recommender(movies, sparse.csr_matrix((4, 4)))
    assert next_prompt(model, {}, []).id == 1
    assert next_prompt(model, {}, [1]).id == 2
    assert next_prompt(model, {1: 5}, [1]).id == 3
    assert next_prompt(model, {1: 5}, [1, 3]).id == 4
    assert next_prompt(model, {1: 5}, [5, 6, 7, 8, 9]) is None
    with TestClient(create_app(model)) as client:
        response = client.post(
            "/api/prompt", json={"ratings": [{"movie_id": 1, "stars": 5}], "shown": [1]}
        )
        assert response.json()["id"] == 3


def test_new_description_ranks_without_refitting():
    movies = [
        Movie(1, "Space story", 2020, ("Sci-Fi",), 40),
        Movie(2, "City story", 2020, ("Sci-Fi",), 30),
    ]
    model = Recommender(movies, sparse.csr_matrix((2, 2)))
    content = ContentIndex(
        movies, {1: "astronaut moon orbit launch", 2: "robot detective city crime"}
    )
    extended = content.extend(
        [(Movie(3, "New movie", 2026, ("Sci-Fi",)), "fresh astronaut moon orbit")]
    )
    assert extended.vectorizer is content.vectorizer
    assert 3 not in content.index
    scores, _ = extended.score({1: 5})
    genre_scores, _ = extended.score({1: 5}, use_text=False)
    assert scores[2] > scores[1]
    assert np.isclose(genre_scores[2], genre_scores[1])
    assert hybrid_picks(model, extended, {1: 5})[0] == {
        "movie": extended.movies[2],
        "because": "Space story",
        "signal": "content",
    }
    assert all(pick["movie"].id != 3 for pick in hybrid_picks(model, extended, {}))
