import json
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .model import Movie, Recommender, search_key


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


def create_app(model: Recommender | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loaded = model or Recommender.load(
            Path(os.getenv("FILMMATCH_MODEL", "artifacts/model"))
        )
        catalog = Catalog(loaded, os.getenv("DATABASE_URL"))
        catalog.initialize()
        app.state.model = loaded
        app.state.catalog = catalog
        app.state.genres = sorted(
            {genre for movie in loaded.movies for genre in movie.genres}
        )
        yield

    api = FastAPI(title="FilmMatch", lifespan=lifespan)

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
        return api.state.catalog.search(q.strip(), limit) if q.strip() else []

    @api.get("/api/prompts")
    def prompts():
        return api.state.model.prompts(set())

    @api.post("/api/recommend")
    def recommend(profile: RecommendationRequest):
        ratings = {rating.movie_id: rating.stars for rating in profile.ratings}
        if ratings.keys() - api.state.model.index.keys():
            raise HTTPException(
                status_code=422, detail="A rated movie is missing from the catalog."
            )
        if profile.genre and profile.genre not in api.state.genres:
            raise HTTPException(status_code=422, detail="Unknown genre.")
        picks = api.state.model.recommend(
            ratings,
            genre=profile.genre,
            year_from=profile.year_from,
            year_to=profile.year_to,
        )
        return [
            {"movie": movie, "because": anchor.title if anchor else None}
            for movie, anchor in picks
        ]

    @api.get("/api/evidence")
    def evidence():
        path = Path(os.getenv("FILMMATCH_EVIDENCE", "benchmarks/release-1.json"))
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
