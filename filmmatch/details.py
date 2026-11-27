import json
import re
import ssl
import time
from functools import lru_cache
from textwrap import shorten
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from .tmdb import TMDB, TMDBError

IMDB_ID = re.compile(r"tt[0-9]{7,}")
POSTER_PATH = re.compile(r"/[A-Za-z0-9_-]+\.(?:jpg|png|webp)")


def score(value) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return round(number, 1) if 0 < number <= 10 else None


class MovieDetails:
    def __init__(self, tmdb: TMDB | None, omdb_key: str):
        self.tmdb = tmdb
        self.omdb_key = omdb_key
        self.tls = (
            ssl.create_default_context(cafile=certifi.where()) if omdb_key else None
        )
        self.tmdb_movie = lru_cache(maxsize=512)(self.fetch_tmdb)
        self.imdb_rating = lru_cache(maxsize=1024)(self.fetch_imdb)

    def fetch_tmdb(self, tmdb_id: int, _hour: int) -> dict:
        data = self.tmdb.get(
            f"movie/{tmdb_id}", append_to_response="credits", timeout=4
        )
        if data.get("id") != tmdb_id:
            raise TMDBError("TMDB returned a different movie.")
        credits = data.get("credits") or {}
        directors = list(
            dict.fromkeys(
                person["name"]
                for person in credits.get("crew", [])
                if person.get("job") == "Director" and person.get("name")
            )
        )[:4]
        cast = [
            person["name"] for person in credits.get("cast", []) if person.get("name")
        ][:5]
        poster = data.get("poster_path") or ""
        runtime = data.get("runtime")
        return {
            "overview": data.get("overview") or None,
            "release_date": data.get("release_date") or None,
            "runtime_minutes": runtime
            if isinstance(runtime, int) and runtime > 0
            else None,
            "directors": directors,
            "cast": cast,
            "imdb_id": data.get("imdb_id"),
            "tmdb_rating": score(data.get("vote_average"))
            if data.get("vote_count")
            else None,
            "poster_url": f"https://image.tmdb.org/t/p/w342{poster}"
            if POSTER_PATH.fullmatch(poster)
            else None,
        }

    def fetch_imdb(self, imdb_id: str, _day: int) -> float | None:
        query = urlencode({"apikey": self.omdb_key, "i": imdb_id, "type": "movie"})
        request = Request(
            f"https://www.omdbapi.com/?{query}", headers={"Accept": "application/json"}
        )
        try:
            with urlopen(request, timeout=4, context=self.tls) as response:
                data = json.load(response)
        except (OSError, ValueError):
            return None
        if (
            not isinstance(data, dict)
            or data.get("Response") != "True"
            or data.get("imdbID") != imdb_id
        ):
            return None
        return score(data.get("imdbRating"))

    def get(self, description: str, tmdb_id: int | None, imdb_id: str | None) -> dict:
        details = {
            "overview": shorten(description, width=650, placeholder="…") or None,
            "release_date": None,
            "runtime_minutes": None,
            "cast": [],
            "directors": [],
            "poster_url": None,
            "tmdb_rating": None,
        }
        now = int(time.time())
        if self.tmdb and tmdb_id:
            try:
                remote = self.tmdb_movie(tmdb_id, now // 3600)
            except (TMDBError, ValueError, TypeError, KeyError):
                remote = {}
            for key in details:
                if remote.get(key):
                    details[key] = remote[key]
            imdb_id = remote.get("imdb_id") or imdb_id
        if not isinstance(imdb_id, str) or not IMDB_ID.fullmatch(imdb_id):
            imdb_id = None
        details["imdb_url"] = (
            f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
        )
        details["imdb_rating"] = (
            self.imdb_rating(imdb_id, now // 86400)
            if imdb_id and self.omdb_key
            else None
        )
        return details
