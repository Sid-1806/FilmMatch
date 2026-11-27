from datetime import UTC, date, datetime
from threading import Event
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from filmmatch import tmdb
from filmmatch.content import ContentIndex
from filmmatch.ingest import IngestQueue
from filmmatch.model import Movie
from filmmatch.mood import LATEST_ERA, mood_picks, quiz_questions


def record(movie_id, title, **changes):
    return {
        "id": movie_id,
        "title": title,
        "release_date": "2019-11-20",
        "overview": "A crew boards an experimental spacecraft to investigate a signal.",
        "genre_ids": [878, 10751],
        "adult": False,
        **changes,
    }


def test_catalog_pages_are_bounded_and_duplicate_source_ids_are_merged():
    class StubTMDB:
        def get(self, path, **params):
            if path == "genre/movie/list":
                return {
                    "genres": [
                        {"id": 878, "name": "Science Fiction"},
                        {"id": 10751, "name": "Family"},
                    ]
                }
            assert path == "discover/movie"
            assert params["include_adult"] == "false"
            assert params["primary_release_date.lte"] == "2019-12-31"
            if (
                params["primary_release_date.gte"] == "2019-01-01"
                and params["page"] == 1
            ):
                return {
                    "results": [
                        record(1, "A Galaxy", original_title="Galaxy, A"),
                        record(2, "Deep Space"),
                        record(3, "Upcoming Film", release_date="2020-01-01"),
                        record(4, "Adult Film", adult=True),
                    ],
                    "total_pages": 2,
                }
            if params["primary_release_date.gte"] == "2019-01-01":
                return {
                    "results": [
                        record(5, "Lost in Space"),
                        record(6, "No Plot", overview=""),
                    ],
                    "total_pages": 2,
                }
            assert params["page"] == 1
            return {"results": [record(2, "Deep Space")], "total_pages": 1}

    queue = IngestQueue(
        None, ContentIndex([Movie(7, "Galaxy, A", 2019, ("Sci-Fi", "Children"))], {})
    )
    movies = tmdb.fetch_catalog(StubTMDB(), queue, date(2019, 12, 31), Event())
    assert set(movies) == {2, 5}
    assert movies[2] == {
        "title": "Deep Space",
        "description": record(2, "")["overview"],
        "genres": ["Children", "Sci-Fi"],
        "year": 2019,
    }


def test_http_error_hides_secret_from_reported_exception(monkeypatch):
    def unauthorized(request, **kwargs):
        assert parse_qs(urlsplit(request.full_url).query)["api_key"] == [
            "test-only-key"
        ]
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(tmdb, "urlopen", unauthorized)
    with pytest.raises(tmdb.TMDBError, match=r"^TMDB returned HTTP 401\.$") as error:
        tmdb.TMDB("test-only-key").get("discover/movie")
    assert error.value.__suppress_context__


def test_latest_era_shortlists_zero_rating_movies():
    year = datetime.now(UTC).year
    movies = [
        Movie(1, "Older Sci-Fi", year - 2, ("Sci-Fi",), 500),
        Movie(2, "Last Year's Sci-Fi", year - 1, ("Sci-Fi",)),
        Movie(3, "This Year's Sci-Fi", year, ("Sci-Fi",)),
        Movie(4, "Next Year's Sci-Fi", year + 1, ("Sci-Fi",)),
    ]
    answers = dict.fromkeys((question["id"] for question in quiz_questions()), "any")
    answers["era"] = LATEST_ERA
    results = mood_picks(ContentIndex(movies, {}), answers)
    assert [pick["movie"].id for pick in results] == [3, 2]
    assert results[0]["reasons"][0]["detail"] == f"Released in {year}"
