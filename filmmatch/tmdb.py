import json
import logging
import ssl
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from threading import Event, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import psycopg2

from .content import title_key
from .ingest import IngestQueue

logger = logging.getLogger(__name__)
FIRST_YEAR = 2019
PAGES_PER_YEAR = 4
RECENT_DAYS = 90
RECENT_PAGES = 5
API_ROOT = "https://api.themoviedb.org/3"
GENRE_ALIASES = {"Science Fiction": "Sci-Fi", "Family": "Children"}


class TMDBError(Exception):
    pass


class TMDB:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tls = ssl.create_default_context(cafile=certifi.where())

    def get(self, path: str, **params) -> dict:
        query = urlencode({"api_key": self.api_key, "language": "en-US", **params})
        request = Request(
            f"{API_ROOT}/{path}?{query}", headers={"Accept": "application/json"}
        )
        try:
            with urlopen(request, timeout=10, context=self.tls) as response:
                data = json.load(response)
        except HTTPError as error:
            raise TMDBError(f"TMDB returned HTTP {error.code}.") from None
        except (URLError, TimeoutError, ValueError):
            raise TMDBError(
                "TMDB could not be reached or returned invalid JSON."
            ) from None
        if not isinstance(data, dict):
            raise TMDBError("TMDB returned an unexpected response.")
        return data


def release_windows(today: date):
    for year in range(FIRST_YEAR, today.year + 1):
        yield date(year, 1, 1), min(date(year, 12, 31), today), PAGES_PER_YEAR
    yield today - timedelta(days=RECENT_DAYS), today, RECENT_PAGES


def normalize_movie(
    record: dict, genres: dict[int, str], today: date
) -> tuple[int, dict] | None:
    try:
        released = date.fromisoformat(record["release_date"])
        tmdb_id = record["id"]
        title = record["title"].strip()
        description = record["overview"].strip()
        tags = sorted({genres[g] for g in record["genre_ids"] if g in genres})
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if (
        type(tmdb_id) is not int
        or tmdb_id <= 0
        or record.get("adult") is True
        or not date(FIRST_YEAR, 1, 1) <= released <= today
        or not title
        or len(title) > 160
        or len(description) < 20
        or not tags
    ):
        return None
    return tmdb_id, {
        "title": title,
        "description": description[:4000],
        "genres": tags,
        "year": released.year,
    }


def fetch_catalog(client: TMDB, queue: IngestQueue, today: date, stop: Event) -> dict:
    supported = queue.base.genre_index
    names = {
        genre["id"]: GENRE_ALIASES.get(genre["name"], genre["name"])
        for genre in client.get("genre/movie/list")["genres"]
    }
    genres = {genre_id: name for genre_id, name in names.items() if name in supported}
    base_titles = {(title_key(movie.title), movie.year) for movie in queue.base.movies}
    movies = {}
    for start, end, page_limit in release_windows(today):
        for page in range(1, page_limit + 1):
            if stop.is_set():
                return {}
            response = client.get(
                "discover/movie",
                **{
                    "primary_release_date.gte": start.isoformat(),
                    "primary_release_date.lte": end.isoformat(),
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "page": page,
                },
            )
            for record in response["results"]:
                normalized = normalize_movie(record, genres, today)
                if normalized:
                    tmdb_id, payload = normalized
                    title = (title_key(payload["title"]), payload["year"])
                    original = record.get("original_title")
                    original_title = (
                        (title_key(original), payload["year"])
                        if isinstance(original, str)
                        else title
                    )
                    if title not in base_titles and original_title not in base_titles:
                        movies[tmdb_id] = payload
            if page >= response["total_pages"]:
                break
    return movies


class TMDBSync:
    def __init__(self, queue: IngestQueue, client: TMDB):
        self.queue = queue
        self.client = client
        self.stop = Event()
        self.thread = Thread(target=self.run, name="tmdb-sync", daemon=True)

    def initialize(self) -> None:
        with closing(self.queue.connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(620983)")
            cur.execute("""CREATE TABLE IF NOT EXISTS tmdb_sync (
                singleton boolean PRIMARY KEY CHECK (singleton),
                due_at timestamptz NOT NULL
            )""")
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1)

    def claim(self) -> bool:
        with closing(self.queue.connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO tmdb_sync (singleton, due_at)
                VALUES (true, now() + interval '15 minutes')
                ON CONFLICT (singleton) DO UPDATE SET due_at=EXCLUDED.due_at
                WHERE tmdb_sync.due_at <= now()
                RETURNING singleton""")
            return cur.fetchone() is not None

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                if self.claim():
                    movies = fetch_catalog(
                        self.client, self.queue, datetime.now(UTC).date(), self.stop
                    )
                    if movies and not self.stop.is_set():
                        self.queue.publish_tmdb(movies)
                        logger.info(
                            "TMDB sync processed %s eligible movies", len(movies)
                        )
                    elif not self.stop.is_set():
                        logger.warning("TMDB sync returned no eligible movies")
            except TMDBError as error:
                logger.warning("TMDB sync did not complete. %s", error)
            except (KeyError, TypeError, ValueError, psycopg2.Error) as error:
                logger.warning("TMDB sync did not complete (%s).", type(error).__name__)
            self.stop.wait(60)
