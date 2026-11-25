from collections import Counter

from .model import Movie, Recommender

PROMPT_LIMIT = 5
PROMPT_POOL = 200


def next_prompt(
    model: Recommender,
    ratings: dict[int, float],
    shown: list[int],
    policy: str = "adaptive",
) -> Movie | None:
    if len(shown) >= PROMPT_LIMIT:
        return None
    candidates = [
        movie
        for movie in model.prompts(set(), PROMPT_POOL)
        if movie.id not in ratings and movie.id not in shown
    ]
    if not candidates:
        return None
    if policy == "popular":
        return candidates[0]
    prior = ratings if policy == "adaptive" else shown
    genres = Counter(
        genre
        for movie_id in prior
        if movie_id in model.index
        for genre in model.movies[model.index[movie_id]].genres
    )
    # Preserve popularity order on ties and prefer movies whose genres are known.
    return min(
        candidates,
        key=lambda movie: (
            sum(genres[g] for g in movie.genres) / len(movie.genres)
            if movie.genres
            else float("inf")
        ),
    )
