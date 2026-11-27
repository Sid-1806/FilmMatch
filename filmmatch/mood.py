from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .content import ContentIndex
from .model import Movie

ANY = "any"
LIMIT = 10
LATEST_ERA = "this-or-last-year"
LATEST_LABEL = "This year or last year"


@dataclass(frozen=True)
class Option:
    label: str
    genres: tuple[str, ...] = ()
    words: str = ""
    discourage: tuple[str, ...] = ()
    weight: float = 1.0


GENRES = (
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
)
ERAS = {
    "classic": ("Before 1980", 1800, 1979),
    "eighties-nineties": ("1980s or 1990s", 1980, 1999),
    "noughties": ("2000s", 2000, 2009),
    "recent": ("2010 or newer", 2010, 2200),
}

QUESTIONS = {
    "feeling": (
        "How do you want to feel afterward?",
        {
            "lighter": Option(
                "Lighter and in a better mood",
                ("Comedy", "Children", "Musical"),
                "celebration reunion laughter friendship",
                ("Horror", "War", "Thriller"),
                2,
            ),
            "thrilled": Option(
                "Buzzing with excitement",
                ("Action", "Adventure", "Thriller"),
                "chase escape heist rescue",
                weight=2,
            ),
            "moved": Option(
                "Moved or a little teary",
                ("Drama", "Romance"),
                "grief loss reunion farewell sacrifice",
                weight=2,
            ),
            "curious": Option(
                "Still thinking about it",
                ("Mystery", "Sci-Fi", "Drama"),
                "identity memory investigation discovery",
                weight=2,
            ),
        },
    ),
    "intensity": (
        "How much tension are you up for?",
        {
            "low": Option(
                "Keep it light",
                ("Comedy", "Children"),
                "holiday festival friendship",
                ("Horror", "Thriller", "War"),
                1.5,
            ),
            "some": Option(
                "A little suspense", ("Mystery", "Adventure"), "clue investigation"
            ),
            "high": Option(
                "Put me on the edge of my seat",
                ("Thriller", "Horror", "Action"),
                "pursuit escape hostage survival",
                weight=1.5,
            ),
        },
    ),
    "genre": (
        "Is there a genre you want tonight?",
        {genre: Option(genre) for genre in GENRES},
    ),
    "setting": (
        "Where should the story take you?",
        {
            "everyday": Option(
                "A recognizable everyday world",
                words="workplace school neighborhood apartment",
            ),
            "city": Option(
                "A city with a life of its own",
                words="city streets subway downtown metropolis",
            ),
            "history": Option(
                "Somewhere in the past",
                ("Western",),
                "historical empire medieval kingdom frontier",
            ),
            "elsewhere": Option(
                "Another world entirely",
                ("Sci-Fi", "Fantasy"),
                "planet spaceship dimension realm",
            ),
        },
    ),
    "hook": (
        "What would pull you into the story?",
        {
            "puzzle": Option(
                "A mystery to solve", ("Mystery",), "clue detective evidence riddle"
            ),
            "relationship": Option(
                "A relationship worth rooting for",
                ("Romance",),
                "relationship wedding couple romance",
            ),
            "mission": Option(
                "A mission that could go wrong",
                ("Action", "Adventure"),
                "mission heist rescue escape",
            ),
            "comic": Option(
                "An escalating comedy of errors",
                ("Comedy",),
                "misunderstanding disguise pretend mistake",
            ),
        },
    ),
    "characters": (
        "Who would you like to spend the story with?",
        {
            "outsider": Option(
                "An outsider finding their place",
                words="outsider newcomer outcast bullying",
            ),
            "team": Option(
                "A group figuring things out together",
                words="team crew squad teammates",
            ),
            "family": Option(
                "A family with history", words="siblings parents family inheritance"
            ),
            "mentor": Option(
                "A mentor and their student",
                words="mentor teacher student coach apprentice",
            ),
        },
    ),
    "theme": (
        "What theme sounds right tonight?",
        {
            "friendship": Option(
                "Friendship and belonging",
                words="friendship friends community belonging",
            ),
            "redemption": Option(
                "Second chances", words="redemption forgiveness reconciliation fresh"
            ),
            "justice": Option(
                "Justice and consequences",
                words="justice trial courtroom revenge corruption",
            ),
            "discovery": Option(
                "Discovery and the unknown",
                words="discovery expedition exploration experiment",
            ),
        },
    ),
    "scale": (
        "How big should the story feel?",
        {
            "personal": Option(
                "Personal and close to home",
                words="home family marriage childhood neighborhood",
            ),
            "journey": Option(
                "A trip somewhere new", ("Adventure",), "journey travel road expedition"
            ),
            "epic": Option(
                "Big stakes and spectacle",
                ("Action", "Fantasy", "Sci-Fi"),
                "battle invasion kingdom apocalypse",
            ),
        },
    ),
    "discovery": (
        "How familiar should the picks be?",
        {
            "popular": Option("Start with widely rated titles"),
            "lesser-known": Option("Dig for lesser-known titles"),
        },
    ),
    "era": (
        "Which release era sounds best?",
        {
            **{key: Option(label) for key, (label, _, _) in ERAS.items()},
            LATEST_ERA: Option(LATEST_LABEL),
        },
    ),
    "animation": (
        "How do you feel about animation tonight?",
        {
            "animated": Option("Animation only"),
            "skip-animation": Option("Leave out animation"),
        },
    ),
    "avoid": (
        "Is there a genre you'd rather leave out?",
        {genre: Option(genre) for genre in ("Horror", "Romance", "War")},
    ),
}
SIGNAL_QUESTIONS = (
    "feeling",
    "intensity",
    "setting",
    "hook",
    "characters",
    "theme",
    "scale",
)


class MoodRequest(BaseModel):
    answers: Annotated[
        dict[str, str], Field(min_length=len(QUESTIONS), max_length=len(QUESTIONS))
    ]

    @model_validator(mode="after")
    def validate_answers(self):
        if self.answers.keys() != QUESTIONS.keys() or any(
            answer != ANY and answer not in QUESTIONS[key][1]
            for key, answer in self.answers.items()
        ):
            raise ValueError("Choose one available option for each question.")
        return self


def quiz_questions() -> list[dict]:
    return [
        {
            "id": key,
            "label": question,
            "options": [
                *(
                    {"id": value, "label": option.label}
                    for value, option in options.items()
                ),
                {"id": ANY, "label": "No preference"},
            ],
        }
        for key, (question, options) in QUESTIONS.items()
    ]


def matches(
    movie: Movie, answers: dict[str, str], era: tuple[str, int, int] | None
) -> bool:
    animation = answers["animation"]
    return (
        (answers["genre"] == ANY or answers["genre"] in movie.genres)
        and answers["avoid"] not in movie.genres
        and (
            animation == ANY
            or ("Animation" in movie.genres) == (animation == "animated")
        )
        and (era is None or (movie.year is not None and era[1] <= movie.year <= era[2]))
    )


def genre_hits(index: ContentIndex, genres: tuple[str, ...]) -> np.ndarray:
    columns = [
        index.genre_index[genre] for genre in genres if genre in index.genre_index
    ]
    return np.any(index.genres[:, columns] > 0, axis=1)


def mood_picks(index: ContentIndex, answers: dict[str, str]) -> list[dict]:
    options = [
        QUESTIONS[key][1][answers[key]]
        for key in SIGNAL_QUESTIONS
        if answers[key] != ANY
    ]
    scores = np.zeros(len(index.movies))
    reasons = []
    vocabulary = None
    if options:
        queries = (
            index.vectorizer.transform([option.words for option in options])
            if index.available
            else None
        )
        if queries is not None:
            vocabulary = index.vectorizer.get_feature_names_out()
        for pos, option in enumerate(options):
            query = queries[pos] if queries is not None else None
            similarity = (
                index.vectors @ query.toarray().ravel() if query is not None else 0
            )
            contribution = option.weight * (
                0.65 * genre_hits(index, option.genres)
                + 0.7 * np.sqrt(similarity)
                - 0.6 * genre_hits(index, option.discourage)
            )
            scores += contribution
            reasons.append((option, query, contribution))

    discovery = answers["discovery"]
    popularity = np.log1p(index.rating_counts)
    popularity /= max(float(popularity.max(initial=0)), 1.0)
    if discovery == "popular":
        scores += 0.75 * popularity
    elif discovery == "lesser-known":
        scores += 0.75 * (1 - popularity)

    era = ERAS.get(answers["era"])
    sort_keys = [index.movie_ids, -index.rating_counts, -scores]
    if answers["era"] == LATEST_ERA:
        year = datetime.now(UTC).year
        era = (LATEST_LABEL, year - 1, year)
        sort_keys.insert(1, np.asarray([-(movie.year or 0) for movie in index.movies]))
    order = np.lexsort(tuple(sort_keys))
    picks = []
    for pos in order:
        movie = index.movies[pos]
        if not matches(movie, answers, era):
            continue
        evidence = []
        ranked_reasons = sorted(reasons, key=lambda entry: -entry[2][pos])
        for option, query, contribution in ranked_reasons:
            if contribution[pos] <= 0:
                break
            tags = [genre for genre in option.genres if genre in movie.genres]
            if tags:
                detail = f"Tagged {' · '.join(tags)}"
            elif query is not None and vocabulary is not None:
                overlap = index.vectors[pos].multiply(query)
                words = vocabulary[overlap.indices]
                detail = f"Plot mentions {', '.join(words[:3])}" if words.size else ""
            else:
                detail = ""
            if detail:
                evidence.append({"answer": option.label, "detail": detail})
        if answers["genre"] != ANY:
            evidence.append({"answer": "Genre", "detail": f"Tagged {answers['genre']}"})
        if era:
            evidence.append({"answer": era[0], "detail": f"Released in {movie.year}"})
        if answers["animation"] == "animated":
            evidence.append({"answer": "Animation only", "detail": "Tagged Animation"})
        if discovery != ANY or not evidence:
            label = (
                QUESTIONS["discovery"][1][discovery].label
                if discovery != ANY
                else "Catalog match"
            )
            evidence.append(
                {
                    "answer": label,
                    "detail": f"{movie.rating_count:,} historical MovieLens ratings",
                }
            )
        picks.append({"movie": movie, "reasons": evidence[:2]})
        if len(picks) == LIMIT:
            break
    return picks
