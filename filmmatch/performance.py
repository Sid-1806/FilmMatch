import argparse
import cProfile
import hashlib
import json
import pstats
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

from .benchmark import write_json
from .content import ContentIndex, hybrid_picks
from .model import Recommender


def load_profile(directory: Path, output: Path) -> None:
    model = Recommender.load(directory)
    rows = json.loads((directory / "descriptions.json").read_text(encoding="utf-8"))
    index = ContentIndex(model.movies, {int(key): text for key, text in rows.items()})
    ratings = {movie.id: 5 for movie in model.prompts(set())}
    profiler = cProfile.Profile()
    with profiler:
        for _ in range(120):
            hybrid_picks(model, index, ratings)
    with output.open("w", encoding="utf-8") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(24)


def measure(origin: str, count: int, rate: int) -> dict:
    with httpx.Client(base_url=origin, timeout=8, trust_env=False) as client:
        response = client.get("/api/prompts")
        response.raise_for_status()
        titles = [movie["id"] for movie in response.json()]
        payloads = [
            {
                "model": "itemknn",
                "ratings": [
                    {"movie_id": movie, "stars": stars}
                    for movie, stars in zip(titles, ratings)
                ],
            }
            for ratings in ((5,), (5, 4), (5, 2, 4), (4, 4, 5, 2, 1), (1, 2, 3, 4, 5))
        ]

        def request(body):
            started = time.perf_counter()
            try:
                response = client.post("/api/recommend", json=body)
                response.raise_for_status()
                result = hashlib.sha256(
                    json.dumps(response.json(), sort_keys=True).encode()
                ).hexdigest()
            except (httpx.HTTPError, ValueError):
                result = None
            return (time.perf_counter() - started) * 1000, result

        for body in payloads * 4:
            if request(body)[1] is None:
                raise SystemExit("Benchmark warm-up failed.")
        with ThreadPoolExecutor(max_workers=4) as pool:
            start = time.perf_counter()
            futures = []
            for i in range(count):
                time.sleep(max(0, start + i / rate - time.perf_counter()))
                futures.append(pool.submit(request, payloads[i % len(payloads)]))
            results = [job.result() for job in futures]
            elapsed = time.perf_counter() - start
    success = [latency for latency, sha in results if sha]
    if not success:
        raise SystemExit("All measured requests failed.")
    return {
        "offered_requests": count,
        "target_rps": rate,
        "successful_requests": len(success),
        "errors": count - len(success),
        "throughput_rps": len(success) / elapsed,
        "p50_ms": float(np.percentile(success, 50)),
        "p95_ms": float(np.percentile(success, 95)),
        "response_sha256": hashlib.sha256(
            json.dumps([sha for _, sha in results]).encode()
        ).hexdigest(),
    }


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Measure the recommendation API under a fixed request schedule"
    )
    cli.add_argument("--origin", default="http://127.0.0.1:8088")
    cli.add_argument("--sha", required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--profile", type=Path)
    args = cli.parse_args()
    result = measure(args.origin, 300, 20)
    write_json(
        args.out,
        {
            "code_sha": args.sha,
            "config": {"warmups": 20, "workers": 4, "model": "itemknn"},
            **result,
        },
    )
    print(json.dumps(result, indent=2), flush=True)
    if args.profile:
        load_profile(Path("artifacts/model"), args.profile)


if __name__ == "__main__":
    main()
