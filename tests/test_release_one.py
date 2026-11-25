import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import sparse

from filmmatch.api import create_app
from filmmatch.benchmark import evaluate, freeze, ndcg, train
from filmmatch.model import Movie, Recommender, item_neighbors


@pytest.fixture
def model():
    movies = [
        Movie(1, "Arrival", 2016, ("Sci-Fi",), 90),
        Movie(2, "Café Noir", 1995, ("Drama",), 80),
        Movie(3, "Interstellar", 2014, ("Sci-Fi",), 70),
        Movie(4, "Contact", 1997, ("Sci-Fi",), 60),
        Movie(5, "Paper Moon", 1973, ("Comedy",), 55),
        Movie(6, "Wall-E", 2008, ("Animation",), 40),
    ]
    likes = sparse.csr_matrix(
        np.array(
            [[1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 1, 1]],
            dtype=np.float32,
        )
    )
    return Recommender(movies, item_neighbors(likes, 3, 10))


def test_similarity_discounts_support():
    likes = sparse.csr_matrix(
        np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.float32)
    )
    weights = item_neighbors(likes, 1, 10)
    assert weights[0, 0] == 0
    assert weights[0, 1] == pytest.approx(2 / 12)
    assert weights[0, 2] == 0


def test_recommendation_reasons_and_filters(model):
    assert model.recommend({})[0][0].id == 1
    assert model.recommend({1: 5})[0][0].id == 3
    assert model.recommend({1: 5})[0][1].id == 1
    assert 1 not in [movie.id for movie, _ in model.recommend({1: 5})]
    assert [
        movie.id for movie, _ in model.recommend({1: 5}, genre="Sci-Fi", year_to=2000)
    ] == [4]
    assert model.recommend({1: 1})[0][0].id == 2


def test_api_supports_fixed_prompts_and_validation(model, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(create_app(model)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert [movie["id"] for movie in client.get("/api/prompts").json()] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert client.get("/api/search", params={"q": "cafe"}).json()[0]["id"] == 2
        for profile in (
            {"ratings": [{"movie_id": 999, "stars": 3}]},
            {"ratings": [{"movie_id": 1, "stars": 6}]},
            {"ratings": [{"movie_id": 1, "stars": 5}] * 2},
            {"year_from": 2020, "year_to": 2010},
        ):
            assert client.post("/api/recommend", json=profile).status_code == 422
        response = client.post(
            "/api/recommend",
            json={"ratings": [{"movie_id": 1, "stars": 5}], "genre": "Sci-Fi"},
        )
        assert response.status_code == 200
        assert response.json()[0]["because"] == "Arrival"
        assert [pick["movie"]["id"] for pick in response.json()] == [3, 4]


def test_split_is_stable_and_disjoint():
    ratings = np.array(
        [[user, 1, 10, time] for user in range(20) for time in range(100)]
    )
    split = freeze(ratings, "hash", 20, 42)
    assert split == freeze(ratings[::-1], "hash", 20, 42)
    assert split["cutoffs"] == [80, 90]
    assert (
        len(set(split["fit_users"] + split["validation_users"] + split["test_users"]))
        == 20
    )
    assert len(split["fit_users"]) == 16


def test_training_and_replay_respect_time_and_cohort():
    movies = [Movie(i, f"Movie {i}", 2000, ()) for i in range(1, 8)]
    events = np.array(
        [
            [1, 1, 10, 1],
            [1, 2, 10, 2],
            [1, 3, 10, 2],
            [1, 4, 10, 10],
            [2, 1, 10, 3],
            [2, 5, 10, 10],
            [2, 2, 10, 11],
            [3, 1, 10, 4],
            [3, 3, 10, 20],
        ]
    )
    split = {
        "cutoffs": [10, 20],
        "fit_users": [1],
        "validation_users": [2],
        "test_users": [3],
        "seed": 42,
    }
    model = train(movies, events, split)
    assert [movie.rating_count for movie in model.movies] == [1, 1, 1, 0, 0, 0, 0]
    results = evaluate(model, events, split)
    assert results["validation"]["eligible_users"] == 1
    assert results["test"]["eligible_users"] == 1
    assert results["validation"]["simulated_skip_rate"] == pytest.approx(2 / 3)
    assert ndcg([4, 3], {3}) == pytest.approx(1 / np.log2(3))
