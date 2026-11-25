import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import scipy

from .benchmark import (
    NEIGHBORS,
    PROMPTS,
    SEED,
    SHRINKAGE,
    digest,
    evaluate,
    freeze,
    load_data,
    train,
    write_json,
)
from .model import Recommender


def clean_sha() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("Commit code and configuration before publishing a benchmark.")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="FilmMatch chronological benchmark")
    parser.add_argument("command", choices=["freeze", "train", "eval"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("artifacts/split.json"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmark.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Require a clean source commit and record its Git SHA",
    )
    args = parser.parse_args()
    archive_sha = digest(args.archive)
    if args.command == "freeze":
        _, ratings = load_data(args.archive, args.sample, args.seed)
        write_json(args.split, freeze(ratings, archive_sha, args.sample, args.seed))
        print(f"Split written to {args.split}")
        return
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if split["archive_sha256"] != archive_sha:
        raise SystemExit("The data archive does not match the frozen split.")
    source_sha = clean_sha() if args.publish else None
    movies, ratings = load_data(args.archive, split["sample_limit"], split["seed"])
    config = {
        "neighbors": NEIGHBORS,
        "shrinkage": SHRINKAGE,
        "prompts": PROMPTS,
        "k": 10,
        "relevant_stars": 4,
        "bootstrap_samples": 2000,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    provenance = {
        "archive_sha256": archive_sha,
        "split_sha256": digest(args.split),
        "seed": split["seed"],
        "sample_limit": split["sample_limit"],
        "cutoffs": split["cutoffs"],
        "config": config,
    }
    if args.command == "train":
        train(movies, ratings, split).save(args.artifacts)
        write_json(
            args.artifacts / "training.json", {**provenance, "code_sha": source_sha}
        )
        print(f"Model written to {args.artifacts}")
        return
    training = json.loads(
        (args.artifacts / "training.json").read_text(encoding="utf-8")
    )
    if any(training.get(key) != value for key, value in provenance.items()):
        raise SystemExit(
            "Train the model against this split and configuration before evaluation."
        )
    if args.publish and training["code_sha"] != source_sha:
        raise SystemExit(
            "Publishable training and evaluation must use the same clean source commit."
        )
    results = evaluate(Recommender.load(args.artifacts), ratings, split)
    write_json(
        args.out,
        {
            **training,
            "dataset": args.archive.stem,
            "sampled_users": len(np.unique(ratings[:, 0])),
            "metrics": results,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
