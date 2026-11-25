import argparse
import csv
import json
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sklearn

from .__main__ import clean_sha
from .benchmark import digest, load_data, ndcg, write_json
from .content import TEXT_WEIGHT, ContentIndex, match_descriptions
from .model import Recommender
from .policy import PROMPT_LIMIT, PROMPT_POOL, next_prompt


def first_ratings(archive: Path) -> dict[int, int]:
    first: dict[int, int] = {}
    with ZipFile(archive) as source:
        filename = next(
            name for name in source.namelist() if Path(name).name == "ratings.csv"
        )
        with TextIOWrapper(source.open(filename), encoding="utf-8-sig") as stream:
            rows = csv.reader(stream)
            next(rows)
            for _, movie, _, timestamp in rows:
                movie_id, date = int(movie), int(timestamp)
                if date < first.get(movie_id, float("inf")):
                    first[movie_id] = date
    return first


def summaries(scores: dict[str, list[float]], primary: str, seed: int) -> dict:
    results = {name: np.asarray(values) for name, values in scores.items()}
    n = len(results[primary])
    if not n:
        return {"users": 0, "ndcg_at_10": {}, "paired_differences": {}}
    pairs = {}
    for name, values in results.items():
        if name == primary:
            continue
        paired = results[primary] - values
        rng = np.random.default_rng(seed)
        draws = np.concatenate(
            [paired[rng.integers(n, size=(100, n))].mean(axis=1) for _ in range(20)]
        )
        pairs[f"{primary}_minus_{name}"] = {
            "difference": float(paired.mean()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    return {
        "users": n,
        "ndcg_at_10": {name: float(values.mean()) for name, values in results.items()},
        "paired_differences": pairs,
    }


def replay(
    model: Recommender, history: dict[int, float], relevant: set[int], policy: str
) -> tuple[float, float]:
    shown: list[int] = []
    answers: dict[int, float] = {}
    while prompt := next_prompt(model, answers, shown, policy):
        shown.append(prompt.id)
        if prompt.id in history:
            answers[prompt.id] = history[prompt.id]
    picks = [movie.id for movie, _ in model.recommend(answers)]
    return ndcg(picks, relevant), (len(shown) - len(answers)) / len(shown)


def experiment(
    model: Recommender,
    ratings: np.ndarray,
    split: dict,
    content: ContentIndex,
    first: dict[int, int],
) -> tuple[dict, dict, int]:
    warm_ids = {movie.id for movie in model.movies if movie.rating_count}
    cold_positions = np.array(
        [
            pos
            for pos, movie in enumerate(content.movies)
            if content.texts[pos]
            and first.get(movie.id, float("inf")) >= split["cutoffs"][0]
        ],
        dtype=int,
    )
    cold_ids = set(map(int, content.movie_ids[cold_positions]))
    policy_reports = {}
    cold_reports = {}
    for cohort, start, end in (
        ("validation", *split["cutoffs"]),
        ("test", split["cutoffs"][1], None),
    ):
        policy_scores: dict[str, list] = {
            key: [] for key in ("popular", "genre_balanced", "adaptive")
        }
        policy_skips: dict[str, list] = {key: [] for key in policy_scores}
        cold_scores: dict[str, list] = {
            key: [] for key in ("genre_only", "text_and_genre")
        }
        events = ratings[np.isin(ratings[:, 0], split[f"{cohort}_users"])]
        events = events[np.lexsort((events[:, 3], events[:, 0]))]
        for rows in np.split(events, np.flatnonzero(np.diff(events[:, 0])) + 1):
            history = {
                int(movie): float(stars) / 2
                for _, movie, stars, date in rows
                if date < start
            }
            relevant = {
                int(movie)
                for _, movie, stars, date in rows
                if stars >= 8
                and date >= start
                and (end is None or date < end)
                and int(movie) not in history
            }
            warm_relevant = relevant & warm_ids
            if warm_relevant:
                for policy, measured in policy_scores.items():
                    score, skipped = replay(model, history, warm_relevant, policy)
                    measured.append(score)
                    policy_skips[policy].append(skipped)
            cold_relevant = relevant & cold_ids
            anchors = dict(
                [(movie, stars) for movie, stars in history.items() if stars >= 4][-5:]
            )
            if cold_relevant and anchors:
                for variant, measured in cold_scores.items():
                    scores, _ = content.score(
                        anchors,
                        use_text=variant == "text_and_genre",
                        positions=cold_positions,
                    )
                    order = np.argsort(-scores, kind="stable")
                    picks = [
                        int(movie)
                        for movie in content.movie_ids[cold_positions[order]]
                        if int(movie) not in history
                    ][:10]
                    measured.append(ndcg(picks, cold_relevant))
        policy_reports[cohort] = {
            **summaries(policy_scores, "adaptive", split["seed"]),
            "skip_rate": {
                key: float(np.mean(values))
                for key, values in policy_skips.items()
                if values
            },
        }
        cold_reports[cohort] = summaries(cold_scores, "text_and_genre", split["seed"])
    return policy_reports, cold_reports, len(cold_positions)


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Evaluate release 2 from the frozen MovieLens split"
    )
    cli.add_argument("--archive", type=Path, default=Path("data/ml-32m.zip"))
    cli.add_argument("--split", type=Path, default=Path("artifacts/split-32m.json"))
    cli.add_argument("--model", type=Path, default=Path("artifacts/model-32m"))
    cli.add_argument("--corpus", type=Path, default=Path("data/MovieSummaries.tar.gz"))
    cli.add_argument("--out", type=Path, default=Path("artifacts/release-2.json"))
    args = cli.parse_args()
    source_sha = clean_sha()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    training = json.loads((args.model / "training.json").read_text(encoding="utf-8"))
    if (
        digest(args.archive) != split["archive_sha256"]
        or digest(args.split) != training["split_sha256"]
    ):
        cli.error("The archive, model, and release 1 split must match.")
    _, ratings = load_data(args.archive, split["sample_limit"], split["seed"])
    print(
        "MovieLens cohort loaded. Matching descriptions and checking first-rating timestamps.",
        flush=True,
    )
    model = Recommender.load(args.model)
    descriptions = match_descriptions(args.corpus, model.movies)
    policy, cold, cold_candidates = experiment(
        model,
        ratings,
        split,
        ContentIndex(model.movies, descriptions),
        first_ratings(args.archive),
    )
    write_json(
        args.out,
        {
            "dataset": args.archive.stem,
            "archive_sha256": split["archive_sha256"],
            "split_sha256": training["split_sha256"],
            "corpus_sha256": digest(args.corpus),
            "source_model_sha": training["code_sha"],
            "neighbors_sha256": digest(args.model / "neighbors.npz"),
            "code_sha": source_sha,
            "seed": split["seed"],
            "sampled_users": split["sample_limit"],
            "catalog_size": len(model.movies),
            "descriptions": len(descriptions),
            "cold_candidates": cold_candidates,
            "config": {
                "prompt_budget": PROMPT_LIMIT,
                "prompt_pool": PROMPT_POOL,
                "text_weight": TEXT_WEIGHT,
                "cold_profile_max_likes": 5,
                "bootstrap_samples": 2000,
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
            },
            "prompt_replay": policy,
            "cold_items": cold,
        },
    )
    print(json.dumps({"prompt_replay": policy, "cold_items": cold}, indent=2))


if __name__ == "__main__":
    main()
