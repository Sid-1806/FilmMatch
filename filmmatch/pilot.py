import os
import time
from contextlib import closing
from hashlib import sha256
from random import Random
from statistics import median
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from psycopg2.extras import Json
from pydantic import UUID4, BaseModel, Field

from .content import ContentIndex, hybrid_picks
from .model import Recommender
from .policy import PROMPT_LIMIT, next_prompt

PROTOCOL = "five-prompts-v1"
POLICIES = ("popular", "adaptive")
ORDER_SEED = 42


class Consent(BaseModel):
    consent: Literal[True]


class PilotEvent(BaseModel):
    revision: Annotated[int, Field(ge=0, le=30)]
    action: Literal["rate", "pick", "next", "prefer"]
    movie_id: Annotated[int, Field(gt=0)] | None = None
    stars: Annotated[int, Field(ge=1, le=5, strict=True)] | None = None
    preference: Literal["first", "second", "tie"] | None = None


def new_round(policy: str, now: float) -> dict:
    return {
        "policy": policy,
        "started_at": now,
        "shown": [],
        "ratings": {},
        "picks": [],
    }


def summarize(states: list[dict]) -> dict:
    paired = [
        state
        for state in states
        if len(state["rounds"]) == 2
        and all("result" in round_ for round_ in state["rounds"])
    ]
    complete = [state for state in paired if "preference" in state]
    result = {
        "protocol": PROTOCOL,
        "order_seed": ORDER_SEED,
        "prompt_limit": PROMPT_LIMIT,
        "started": len(states),
        "paired": len(paired),
        "completed": len(complete),
    }
    if not paired:
        return result
    preference = dict.fromkeys((*POLICIES, "tie"), 0)
    for state in complete:
        selected = state["preference"]
        if selected == "tie":
            preference["tie"] += 1
        else:
            preference[state["rounds"][int(selected == "second")]["policy"]] += 1
    outcomes = {
        policy: [
            round_["result"]
            for state in paired
            for round_ in state["rounds"]
            if round_["policy"] == policy
        ]
        for policy in POLICIES
    }
    results = {}
    for policy, rows in outcomes.items():
        seconds = [row["elapsed_s"] for row in rows if row["found"]]
        results[policy] = {
            "found": len(seconds),
            "total_skips": sum(row["skips"] for row in rows),
            "median_success_seconds": median(seconds) if seconds else None,
        }
    pairs = dict.fromkeys(("both", "popular_only", "adaptive_only", "neither"), 0)
    for state in paired:
        found = [
            round_["policy"] for round_ in state["rounds"] if round_["result"]["found"]
        ]
        key = (
            "neither"
            if not found
            else f"{found[0]}_only"
            if len(found) == 1
            else "both"
        )
        pairs[key] += 1
    return {
        **result,
        "policies": results,
        "preferences": preference,
        "paired_picks": pairs,
        "order_among_paired": {
            policy: sum(state["rounds"][0]["policy"] == policy for state in paired)
            for policy in POLICIES
        },
        "app_versions": sorted({state["app_version"] for state in paired}),
    }


class Pilot:
    def __init__(self, connect, model: Recommender, content: ContentIndex):
        self.connect = connect
        self.model = model
        self.content = content

    def initialize(self) -> None:
        with (
            closing(self.connect()) as connection,
            connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("SELECT pg_advisory_xact_lock(620983)")
            cursor.execute("""CREATE TABLE IF NOT EXISTS pilot_sessions (
                sequence bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                protocol text NOT NULL, session_hash text NOT NULL, state jsonb NOT NULL,
                UNIQUE (protocol, session_hash)
            )""")

    def start(self, key: str) -> dict:
        with (
            closing(self.connect()) as connection,
            connection,
            connection.cursor() as cursor,
        ):
            cursor.execute("SELECT pg_advisory_xact_lock(620983)")
            cursor.execute(
                "SELECT state FROM pilot_sessions WHERE protocol=%s AND session_hash=%s",
                (PROTOCOL, key),
            )
            existing = cursor.fetchone()
            if existing:
                return existing[0]
            cursor.execute(
                "SELECT count(*) FROM pilot_sessions WHERE protocol=%s", (PROTOCOL,)
            )
            position = cursor.fetchone()[0]
            first = Random(f"{ORDER_SEED}:{position // 2}").randrange(2)
            policy = POLICIES[first ^ (position % 2)]
            state = {
                "revision": 0,
                "phase": "prompt",
                "app_version": os.getenv("RENDER_GIT_COMMIT", "local"),
                "rounds": [new_round(policy, time.time())],
            }
            cursor.execute(
                "INSERT INTO pilot_sessions (protocol, session_hash, state) VALUES (%s,%s,%s)",
                (PROTOCOL, key, Json(state)),
            )
            return state

    def read(self, key: str) -> dict:
        with closing(self.connect()) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT state FROM pilot_sessions WHERE protocol=%s AND session_hash=%s",
                (PROTOCOL, key),
            )
            row = cursor.fetchone()
        if row is None:
            raise HTTPException(404, "Pilot session not found.")
        return row[0]

    def edit(self, key: str, event: PilotEvent) -> dict:
        with (
            closing(self.connect()) as connection,
            connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(
                "SELECT state FROM pilot_sessions WHERE protocol=%s AND session_hash=%s FOR UPDATE",
                (PROTOCOL, key),
            )
            row = cursor.fetchone()
            if row is None:
                raise HTTPException(404, "Pilot session not found.")
            state = row[0]
            if event.revision < state["revision"]:
                return state
            if event.revision != state["revision"]:
                raise HTTPException(409, "Reload the current pilot step.")
            self.advance(state, event, time.time())
            state["revision"] += 1
            cursor.execute(
                "UPDATE pilot_sessions SET state=%s WHERE protocol=%s AND session_hash=%s",
                (Json(state), PROTOCOL, key),
            )
            return state

    def advance(self, state: dict, event: PilotEvent, now: float) -> None:
        round_ = state["rounds"][-1]
        ratings = {int(movie): stars for movie, stars in round_["ratings"].items()}
        if state["phase"] == "prompt" and event.action == "rate":
            prompt = next_prompt(self.model, ratings, round_["shown"], round_["policy"])
            if prompt is None or prompt.id != event.movie_id:
                raise HTTPException(409, "Answer the current movie prompt.")
            round_["shown"].append(prompt.id)
            if event.stars is not None:
                ratings[prompt.id] = event.stars
                round_["ratings"][str(prompt.id)] = event.stars
            if len(round_["shown"]) == PROMPT_LIMIT:
                round_["picks"] = [
                    pick["movie"].id
                    for pick in hybrid_picks(self.model, self.content, ratings)
                ]
                state["phase"] = "shortlist"
        elif state["phase"] == "shortlist" and event.action == "pick":
            if event.movie_id is not None and event.movie_id not in round_["picks"]:
                raise HTTPException(422, "Choose a film from the shortlist.")
            round_["result"] = {
                "found": event.movie_id is not None,
                "elapsed_s": round(max(0, now - round_["started_at"]), 3),
                "skips": len(round_["shown"]) - len(ratings),
            }
            state["phase"] = "between" if len(state["rounds"]) == 1 else "preference"
        elif state["phase"] == "between" and event.action == "next":
            state["rounds"].append(
                new_round(POLICIES[int(round_["policy"] == POLICIES[0])], now)
            )
            state["phase"] = "prompt"
        elif (
            state["phase"] == "preference"
            and event.action == "prefer"
            and event.preference
        ):
            state["preference"] = event.preference
            state["phase"] = "complete"
        else:
            raise HTTPException(409, "That action is unavailable in the current step.")

    def display(self, state: dict) -> dict:
        round_ = state["rounds"][-1]
        response = {
            "revision": state["revision"],
            "phase": state["phase"],
            "round": len(state["rounds"]),
            "answered": len(round_["shown"]),
            "prompt_limit": PROMPT_LIMIT,
        }
        if state["phase"] == "prompt":
            ratings = {int(movie): stars for movie, stars in round_["ratings"].items()}
            response["prompt"] = next_prompt(
                self.model, ratings, round_["shown"], round_["policy"]
            )
        elif state["phase"] == "shortlist":
            response["picks"] = [
                self.content.movies[self.content.index[movie]]
                for movie in round_["picks"]
            ]
        return response

    def summary(self) -> dict:
        with closing(self.connect()) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT state FROM pilot_sessions WHERE protocol=%s", (PROTOCOL,)
            )
            return summarize([state for (state,) in cursor.fetchall()])


router = APIRouter(prefix="/api/pilot")
SessionHeader = Annotated[UUID4, Header(alias="X-Pilot-Session")]


def study(request: Request) -> Pilot:
    pilot = request.app.state.pilot
    if pilot is None:
        raise HTTPException(503, "The pilot needs Postgres.")
    return pilot


@router.post("/start")
def start_session(_: Consent, request: Request, session: SessionHeader):
    pilot = study(request)
    return pilot.display(pilot.start(sha256(session.bytes).hexdigest()))


@router.get("/session")
def read_session(request: Request, session: SessionHeader):
    pilot = study(request)
    return pilot.display(pilot.read(sha256(session.bytes).hexdigest()))


@router.post("/event")
def event(event: PilotEvent, request: Request, session: SessionHeader):
    pilot = study(request)
    return pilot.display(pilot.edit(sha256(session.bytes).hexdigest(), event))


@router.get("/summary")
def summary(request: Request):
    return study(request).summary()
