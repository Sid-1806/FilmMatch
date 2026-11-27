from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from filmmatch.api import create_app
from filmmatch.content import ContentIndex
from filmmatch.model import Movie, Recommender
from filmmatch.mood import mood_picks, quiz_questions


def neutral_answers():
    return {question["id"]: "any" for question in quiz_questions()}


def test_mood_changes_order_and_explains_genre_and_plot_matches():
    comedy = Movie(1, "Festival", 2005, ("Comedy",), 10)
    horror = Movie(2, "Fight to Survive", 2015, ("Horror",), 50)
    comedy_without_description = Movie(3, "Comedy Club", 2002, ("Comedy",), 100)
    new_movie = Movie(4, "New Friends", 2016, ("Comedy",), 0)
    index = ContentIndex(
        [comedy, horror, comedy_without_description],
        {1: "festival friendship laughter teacher", 2: "hostage escape survival coach"},
    ).extend([(new_movie, "teacher and coach")])
    answers = neutral_answers()
    assert mood_picks(index, answers)[0]["movie"] == comedy_without_description

    light = mood_picks(index, {**answers, "feeling": "lighter", "intensity": "low"})
    assert light[0]["movie"] == comedy
    assert {reason["detail"] for reason in light[0]["reasons"]} == {"Tagged Comedy"}
    tense = mood_picks(index, {**answers, "intensity": "high"})
    assert tense[0]["movie"] == horror

    mentor = mood_picks(index, {**answers, "characters": "mentor"})
    assert mentor[0]["movie"] == new_movie
    assert mentor[0]["reasons"][0]["detail"].startswith("Plot mentions ")
    assert "teacher" in mentor[0]["reasons"][0]["detail"]


def test_superhero_answers_filter_by_title_and_plot_evidence():
    spider_man = Movie(1, "Spider-Man: Brand New Day", 2026, ("Action",))
    batman = Movie(2, "Batman", 2025, ("Action",), 90)
    heist = Movie(3, "Crime on the Run", 2026, ("Action",), 120)
    cameo = Movie(4, "The Crossover", 2026, ("Action",), 300)
    false_match = Movie(5, "Spidermania", 2026, ("Action",), 999)
    index = ContentIndex(
        [spider_man, batman, heist, cameo, false_match],
        {
            3: "A heist, a chase and a daring escape.",
            4: "The villain tracks Peter Parker across the city.",
        },
    )
    answers = {**neutral_answers(), "genre": "Action", "hook": "superhero"}
    generic = mood_picks(index, answers)
    assert {pick["movie"] for pick in generic} == {spider_man, batman, cameo}
    picks = mood_picks(index, {**answers, "characters": "spider-man"})
    assert [pick["movie"] for pick in picks] == [spider_man, cameo]
    assert picks[0]["reasons"][0] == {
        "answer": "Spider-Man",
        "detail": "Title mentions Spider-Man",
    }
    assert picks[1]["reasons"][0]["detail"] == "Plot mentions Peter Parker"


def test_quiz_api_validates_answers_and_enforces_explicit_filters(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    movies = [
        Movie(1, "Space Cartoons", 2010, ("Sci-Fi", "Animation"), 40),
        Movie(2, "Old Cartoons", 2009, ("Sci-Fi", "Animation"), 60),
        Movie(3, "Space Frights", 2021, ("Sci-Fi", "Animation", "Horror"), 80),
        Movie(4, "Live Action Space", 2021, ("Sci-Fi",), 100),
        Movie(5, "Undated Animation", None, ("Sci-Fi", "Animation"), 90),
    ]
    model = Recommender(movies, csr_matrix((len(movies), len(movies))))
    with TestClient(create_app(model)) as client:
        questions = client.get("/api/mood/questions").json()
        assert len(questions) == 12
        assert all(
            "any" in {option["id"] for option in q["options"]} for q in questions
        )
        answers = neutral_answers()
        variant = next(q["variant"] for q in questions if q["id"] == "characters")
        assert variant["when"] == {"question": "hook", "option": "superhero"}
        assert {"spider-man", "any"} <= {option["id"] for option in variant["options"]}
        superhero = {**answers, "hook": "superhero", "characters": "spider-man"}
        response = client.post("/api/mood/recommend", json={"answers": superhero})
        assert response.status_code == 200
        filters = {
            **answers,
            "genre": "Sci-Fi",
            "era": "recent",
            "animation": "animated",
            "avoid": "Horror",
        }
        result = client.post("/api/mood/recommend", json={"answers": filters})
        assert result.status_code == 200
        assert [pick["movie"]["id"] for pick in result.json()] == [1]
        result = client.post(
            "/api/mood/recommend", json={"answers": {**filters, "genre": "Horror"}}
        )
        assert result.json() == []
        for invalid in (
            {},
            {**answers, "extra": "any"},
            {**answers, "era": "futuristic"},
            {**answers, "characters": "spider-man"},
            {**answers, "hook": "superhero", "characters": "mentor"},
        ):
            assert (
                client.post(
                    "/api/mood/recommend", json={"answers": invalid}
                ).status_code
                == 422
            )
