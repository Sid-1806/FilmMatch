import io
import json
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from filmmatch import details
from filmmatch.api import create_app
from filmmatch.model import Movie, Recommender
from filmmatch.tmdb import TMDBError


def test_details_fetch_by_external_id_and_keep_ratings_separate(monkeypatch):
    class TMDBStub:
        calls = 0

        def get(self, route, **params):
            self.calls += 1
            assert route == "movie/42"
            assert params["append_to_response"] == "credits"
            return {
                "id": 42,
                "imdb_id": "tt0123456",
                "overview": "A new plot.",
                "runtime": 115,
                "vote_average": 7.15,
                "vote_count": 10,
                "poster_path": "/poster.jpg",
                "credits": {
                    "crew": [{"job": "Director", "name": "A. Director"}],
                    "cast": [{"name": "B. Actor"}],
                },
            }

    def omdb(request, **kwargs):
        assert parse_qs(urlsplit(request.full_url).query)["i"] == ["tt0123456"]
        return io.BytesIO(
            b'{"Response":"True","imdbID":"tt0123456","imdbRating":"8.4"}'
        )

    monkeypatch.setattr(details, "urlopen", omdb)
    tmdb = TMDBStub()
    service = details.MovieDetails(tmdb, "test-key")
    result = service.get("Cached plot.", 42, None)
    assert result["overview"] == "A new plot."
    assert result["runtime_minutes"] == 115
    assert result["directors"] == ["A. Director"]
    assert result["cast"] == ["B. Actor"]
    assert result["poster_url"] == "https://image.tmdb.org/t/p/w342/poster.jpg"
    assert result["imdb_rating"] == 8.4
    assert result["tmdb_rating"] == 7.2
    assert result["imdb_url"] == "https://www.imdb.com/title/tt0123456/"
    assert service.get("Cached plot.", 42, None) == result
    assert tmdb.calls == 1


def test_provider_failure_keeps_plot_and_does_not_substitute_rating(monkeypatch):
    class TMDBUnavailable:
        def get(self, route, **params):
            raise TMDBError("Unavailable.")

    monkeypatch.setattr(
        details,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            b'{"Response":"True","imdbID":"tt0000000","imdbRating":"10"}'
        ),
    )
    result = details.MovieDetails(TMDBUnavailable(), "test-key").get(
        "Cached plot.", 42, "tt0123456"
    )
    assert result["overview"] == "Cached plot."
    assert result["imdb_rating"] is None
    assert result["imdb_url"] == "https://www.imdb.com/title/tt0123456/"


def test_details_route_uses_the_catalog_mapping(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("OMDB_API_KEY", raising=False)
    monkeypatch.setenv("FILMMATCH_MODEL", str(tmp_path))
    (tmp_path / "links.json").write_text(
        json.dumps({"1": {"imdb": "tt0123456", "tmdb": 42}})
    )
    model = Recommender([Movie(1, "A Movie", 2025, ("Drama",))], csr_matrix((1, 1)))
    with TestClient(create_app(model)) as client:
        response = client.get("/api/movies/1")
        result = response.json()
        assert response.status_code == 200
        assert result["movie"]["title"] == "A Movie"
        assert result["imdb_url"] == "https://www.imdb.com/title/tt0123456/"
        assert result["imdb_rating"] is None
        assert client.get("/api/movies/2").status_code == 404
