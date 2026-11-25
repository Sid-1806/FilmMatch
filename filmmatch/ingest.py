import logging
from contextlib import closing
from threading import Event, Thread

from psycopg2.extras import Json

from .content import ContentIndex
from .model import Movie

logger = logging.getLogger(__name__)


def added_movie(movie_id: int, payload: dict) -> tuple[Movie, str]:
    return Movie(
        movie_id, payload["title"], payload["year"], tuple(payload["genres"])
    ), payload["description"]


class IngestQueue:
    def __init__(self, connect, index: ContentIndex):
        self.connect = connect
        self.base = index
        self.snapshot = index
        self.published: tuple[int, ...] = ()
        self.stop = Event()
        self.wake = Event()
        self.thread = Thread(target=self.run, name="film-ingest", daemon=True)

    def initialize(self) -> None:
        with closing(self.connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(620982)")
            cur.execute("""CREATE TABLE IF NOT EXISTS movie_ingests (
                id integer GENERATED ALWAYS AS IDENTITY (START WITH 1000000) PRIMARY KEY,
                request_key text NOT NULL UNIQUE, payload jsonb NOT NULL,
                status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','ready','failed')),
                attempts smallint NOT NULL DEFAULT 0,
                next_attempt_at timestamptz NOT NULL DEFAULT now(),
                created_at timestamptz NOT NULL DEFAULT now(), ready_at timestamptz
            )""")
        self.refresh()
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=6)

    def submit(self, key: str, payload: dict) -> dict | None:
        with closing(self.connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO movie_ingests (request_key,payload) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (key, Json(payload)),
            )
            cur.execute(
                "SELECT id,status,payload FROM movie_ingests WHERE request_key=%s",
                (key,),
            )
            movie_id, status, stored = cur.fetchone()
            if payload != stored:
                return None
        self.wake.set()
        return {"id": movie_id, "status": status}

    def status(self, movie_id: int) -> dict | None:
        with closing(self.connect()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status,attempts,created_at,ready_at FROM movie_ingests WHERE id=%s",
                (movie_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return dict(
                zip(
                    ("id", "status", "attempts", "created_at", "ready_at"),
                    (movie_id, *row),
                    strict=True,
                )
            )

    def process_one(self) -> bool:
        with closing(self.connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("""SELECT id,payload,attempts FROM movie_ingests
                WHERE status='pending' AND next_attempt_at<=now()
                ORDER BY next_attempt_at,id LIMIT 1 FOR UPDATE SKIP LOCKED""")
            row = cur.fetchone()
            if row is None:
                return False
            movie_id, payload, attempts = row
            attempts += 1
            try:
                movie, description = added_movie(movie_id, payload)
                self.base.genre_vectors([movie])
                if self.base.available:
                    self.base.vectorizer.transform([description])
            except Exception:
                logger.exception("Ingest processing failed for movie %s", movie_id)
                cur.execute(
                    """UPDATE movie_ingests SET attempts=%s, status=%s,
                    next_attempt_at=now()+%s*interval '1 second' WHERE id=%s""",
                    (
                        attempts,
                        "failed" if attempts >= 3 else "pending",
                        2**attempts,
                        movie_id,
                    ),
                )
            else:
                cur.execute(
                    "UPDATE movie_ingests SET attempts=%s,status='ready',ready_at=now() WHERE id=%s",
                    (attempts, movie_id),
                )
        return True

    def refresh(self) -> None:
        with closing(self.connect()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,payload FROM movie_ingests WHERE status='ready' ORDER BY id"
            )
            rows = cur.fetchall()
        ids = tuple(movie_id for movie_id, _ in rows)
        if ids != self.published:
            self.snapshot = self.base.extend(
                [added_movie(movie_id, data) for movie_id, data in rows]
            )
            self.published = ids

    def run(self) -> None:
        while not self.stop.is_set():
            self.wake.clear()
            try:
                for _ in range(8):
                    if self.stop.is_set() or not self.process_one():
                        break
                self.refresh()
            except Exception:
                logger.exception("Ingest worker could not process the queue")
            self.wake.wait(3)
