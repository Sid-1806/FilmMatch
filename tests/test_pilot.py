import numpy as np
import pytest
from fastapi import HTTPException
from scipy import sparse

from filmmatch.content import ContentIndex
from filmmatch.model import Movie, Recommender
from filmmatch.pilot import Pilot, PilotEvent, new_round, summarize


def test_five_prompts_then_fresh_counterbalanced_round():
    genres = ("Drama", "Drama", "Sci-Fi", "Comedy", "Drama", "Mystery")
    movies = [
        Movie(i, f"Film {i}", 2000, (genre,), 100 - i)
        for i, genre in enumerate(genres, start=1)
    ]
    pilot = Pilot(
        None,
        Recommender(movies, sparse.csr_matrix(np.zeros((6, 6)))),
        ContentIndex(movies, {}),
    )
    state = {
        "phase": "prompt",
        "revision": 0,
        "app_version": "test",
        "rounds": [new_round("popular", 0.0)],
    }

    def act(action, **changes):
        pilot.advance(state, PilotEvent(action=action, revision=0, **changes), 10.0)

    with pytest.raises(HTTPException, match="unavailable"):
        act("pick")
    for i in range(5):
        prompt = pilot.display(state)["prompt"]
        act("rate", movie_id=prompt.id, stars=None if i == 0 else 5)
    assert state["phase"] == "shortlist"
    assert "1" not in state["rounds"][0]["ratings"]
    assert len(state["rounds"][0]["ratings"]) == 4
    with pytest.raises(HTTPException, match="from the shortlist"):
        act("pick", movie_id=999)
    act("pick", movie_id=pilot.display(state)["picks"][0].id)
    act("next")
    assert state["rounds"][1]["policy"] == "adaptive"
    assert pilot.display(state)["answered"] == 0
    assert state["rounds"][1]["ratings"] == {}
    for _ in range(5):
        act("rate", movie_id=pilot.display(state)["prompt"].id)
    act("pick")
    result = summarize([state])
    assert result["paired"] == 1 and result["completed"] == 0
    act("prefer", preference="first")
    result = summarize([state])
    assert result["policies"]["popular"] == {
        "found": 1,
        "total_skips": 1,
        "median_success_seconds": 10.0,
    }
    assert result["policies"]["adaptive"]["total_skips"] == 5
    assert result["paired_picks"]["popular_only"] == 1
    assert result["preferences"]["popular"] == 1
