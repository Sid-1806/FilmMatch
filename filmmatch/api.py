import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, model_validator

from .content import ContentIndex, hybrid_picks
from .ingest import IngestQueue
from .model import Movie, Recommender, search_key
from .policy import PROMPT_LIMIT, next_prompt


def search_order(movie: Movie, query: str) -> tuple:
    title = search_key(movie.title)
    return (not title.startswith(query), -movie.rating_count, title, movie.id)


class Catalog:
    def __init__(self, model: Recommender, database_url: str | None):
        self.model = model
        self.database_url = database_url
        self.version = sha256(
            json.dumps([asdict(movie) for movie in model.movies]).encode()
        ).hexdigest()

    def connect(self):
        from psycopg2 import connect

        return connect(self.database_url, connect_timeout=5)

    def initialize(self) -> None:
        if not self.database_url:
            return
        from contextlib import closing

        from psycopg2.extras import execute_values

        with (
            closing(self.connect()) as connection,
            connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("SELECT pg_advisory_xact_lock(620981)")
            cursor.execute("""CREATE TABLE IF NOT EXISTS film_catalog (
                version text NOT NULL, id integer NOT NULL, search_title text NOT NULL,
                rating_count integer NOT NULL, PRIMARY KEY (version, id)
            )""")
            cursor.execute(
                "SELECT count(*) FROM film_catalog WHERE version = %s", (self.version,)
            )
            if cursor.fetchone()[0] == len(self.model.movies):
                return
            execute_values(
                cursor,
                """INSERT INTO film_catalog (version, id, search_title, rating_count)
                VALUES %s ON CONFLICT (version, id) DO NOTHING""",
                [
                    (
                        self.version,
                        movie.id,
                        search_key(movie.title),
                        movie.rating_count,
                    )
                    for movie in self.model.movies
                ],
            )

    def search(self, query: str, limit: int) -> list[Movie]:
        normalized = search_key(query)
        if not self.database_url:
            return sorted(
                (
                    movie
                    for movie in self.model.movies
                    if normalized in search_key(movie.title)
                ),
                key=lambda movie: search_order(movie, normalized),
            )[:limit]
        from contextlib import closing

        with closing(self.connect()) as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT id FROM film_catalog WHERE version = %s AND strpos(search_title, %s) > 0
                ORDER BY starts_with(search_title, %s) DESC, rating_count DESC, search_title COLLATE "C", id
                LIMIT %s""",
                (self.version, normalized, normalized, limit),
            )
            return [
                self.model.movies[self.model.index[movie_id]]
                for (movie_id,) in cursor.fetchall()
            ]


class Rating(BaseModel):
    movie_id: Annotated[int, Field(gt=0, strict=True)]
    stars: Annotated[int, Field(ge=1, le=5, strict=True)]


class RecommendationRequest(BaseModel):
    ratings: Annotated[list[Rating], Field(max_length=50)] = Field(default_factory=list)
    genre: Annotated[str, Field(min_length=1, max_length=32)] | None = None
    year_from: Annotated[int, Field(ge=1800, le=2200)] | None = None
    year_to: Annotated[int, Field(ge=1800, le=2200)] | None = None

    @model_validator(mode="after")
    def validate_profile(self):
        if len({rating.movie_id for rating in self.ratings}) != len(self.ratings):
            raise ValueError("Rate each movie once.")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("The start year must precede the end year.")
        return self


class PromptRequest(RecommendationRequest):
    shown: Annotated[
        list[Annotated[int, Field(gt=0)]], Field(max_length=PROMPT_LIMIT)
    ] = Field(default_factory=list)


class MovieSubmission(BaseModel):
    title: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
    ]
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=20, max_length=4000)
    ]
    genres: Annotated[
        list[Annotated[str, StringConstraints(min_length=1, max_length=32)]],
        Field(min_length=1, max_length=8),
    ]
    year: Annotated[int, Field(ge=1800, le=2200)]


def create_app(model: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        directory = Path(os.getenv("FILMMATCH_MODEL", "artifacts/model"))
        loaded = model or Recommender.load(directory)
        catalog = Catalog(loaded, os.getenv("DATABASE_URL"))
        catalog.initialize()
        app.state.model = loaded
        app.state.catalog = catalog
        app.state.genres = sorted(
            {genre for movie in loaded.movies for genre in movie.genres}
        )
        path = directory / "descriptions.json"
        descriptions = (
            {
                int(key): value
                for key, value in json.loads(path.read_text(encoding="utf-8")).items()
            }
            if model is None and path.exists()
            else {}
        )
        app.state.content = ContentIndex(loaded.movies, descriptions)
        queue = (
            IngestQueue(catalog.connect, app.state.content)
            if catalog.database_url
            else None
        )
        app.state.queue = queue
        if queue:
            queue.initialize()
        try:
            yield
        finally:
            if queue:
                queue.close()

    api = FastAPI(title="FilmMatch", lifespan=lifespan)

    def snapshot() -> ContentIndex:
        return api.state.queue.snapshot if api.state.queue else api.state.content

    def ingest_queue(authorization: str | None) -> IngestQueue:
        token = os.getenv("FILMMATCH_ADMIN_TOKEN")
        if not token or not compare_digest(
            (authorization or "").encode(), f"Bearer {token}".encode()
        ):
            raise HTTPException(
                status_code=401,
                detail="Admin authorization required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if api.state.queue is None:
            raise HTTPException(status_code=503, detail="Ingestion requires Postgres.")
        return api.state.queue

    @api.get("/api/health")
    def health():
        return {"status": "ok"}

    @api.get("/api/catalog")
    def catalog():
        return {"genres": api.state.genres}

    @api.get("/api/search")
    def search(
        q: Annotated[str, Query(min_length=1, max_length=100)],
        limit: Annotated[int, Query(ge=1, le=25)] = 10,
    ):
        query = search_key(q.strip())
        if not query:
            return []
        candidates = api.state.catalog.search(q.strip(), limit)
        added = snapshot().movies[len(api.state.model.movies) :]
        candidates.extend(movie for movie in added if query in search_key(movie.title))
        return sorted(candidates, key=lambda movie: search_order(movie, query))[:limit]

    @api.get("/api/prompts")
    def prompts():
        return api.state.model.prompts(set())

    @api.post("/api/prompt")
    def prompt(profile: PromptRequest):
        return next_prompt(
            api.state.model,
            {rating.movie_id: rating.stars for rating in profile.ratings},
            list(dict.fromkeys(profile.shown)),
        )

    @api.post("/api/recommend")
    def recommend(profile: RecommendationRequest):
        ratings = {rating.movie_id: rating.stars for rating in profile.ratings}
        content = snapshot()
        if ratings.keys() - content.index.keys():
            raise HTTPException(
                status_code=422, detail="A rated movie is missing from the catalog."
            )
        if profile.genre and profile.genre not in api.state.genres:
            raise HTTPException(status_code=422, detail="Unknown genre.")
        return hybrid_picks(
            api.state.model,
            content,
            ratings,
            genre=profile.genre,
            year_from=profile.year_from,
            year_to=profile.year_to,
        )

    @api.post("/api/admin/movies", status_code=202)
    def submit_movie(
        movie: MovieSubmission,
        idempotency_key: Annotated[str, Header(pattern=r"^[A-Za-z0-9._-]{8,100}$")],
        authorization: Annotated[str | None, Header()] = None,
    ):
        queue = ingest_queue(authorization)
        if set(movie.genres) - set(api.state.genres):
            raise HTTPException(status_code=422, detail="Unknown genre.")
        payload = {**movie.model_dump(), "genres": sorted(set(movie.genres))}
        job = queue.submit(idempotency_key, payload)
        if job is None:
            raise HTTPException(
                status_code=409,
                detail="That key already belongs to another submission.",
            )
        return job

    @api.get("/api/admin/movies/{movie_id}")
    def movie_status(
        movie_id: int, authorization: Annotated[str | None, Header()] = None
    ):
        job = ingest_queue(authorization).status(movie_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    @api.get("/api/evidence")
    def evidence():
        path = Path(os.getenv("FILMMATCH_EVIDENCE", "benchmarks/release-1.json"))
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail="Published benchmark is pending."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @api.get("/api/evidence/release-2")
    def release_two_evidence():
        path = Path("benchmarks/release-2.json")
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail="Published benchmark is pending."
            )
        return json.loads(path.read_text(encoding="utf-8"))

    frontend = Path(os.getenv("FILMMATCH_FRONTEND", "frontend/dist"))
    if (frontend / "index.html").is_file():
        api.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

        @api.get("/", include_in_schema=False)
        def index():
            return FileResponse(frontend / "index.html")

    return api


app = create_app()
