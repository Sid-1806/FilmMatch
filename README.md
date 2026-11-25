# FilmMatch

FilmMatch helps a new viewer find a movie from a handful of ratings. The project compares the effect of asking better questions under a fixed five-prompt budget. The public benchmark is generated from committed run records.

## Release 1 · fixed prompts

The app starts with popular movies and updates the top ten after each rating. Search for a title you know, rate or skip a suggested title, and filter the shortlist by genre and decade. An ItemKNN explanation names a rated movie when its similarity contributed positively. Ratings are stored in the browser and sent with recommendation requests.

## Run locally

Start the Docker engine and run the Compose stack.

```sh
docker compose up --build --wait
```

Open [FilmMatch on localhost](http://127.0.0.1:8000). The `web` container builds React, downloads MovieLens Latest Small, trains the development model, and seeds the Postgres search catalog on startup. The API is bound to loopback on the host. Postgres is reachable from the Compose network.

For local development with Python 3.12 and Node 24 run these commands from the repository root.

```sh
python -m pip install -r requirements-dev.txt
python -m filmmatch.download ml-latest-small
python -m filmmatch freeze --archive data/ml-latest-small.zip
python -m filmmatch train --archive data/ml-latest-small.zip
npm ci --prefix frontend
npm run build --prefix frontend
python -m uvicorn filmmatch.api:app
```

`DATABASE_URL` selects Postgres when set. The Python-only command uses the model catalog in memory for local development. Set `FILMMATCH_MODEL` to point the API at another trained artifact directory.

## Reproduce the benchmark

Run the following on the source revision identified in the committed benchmark. `--publish` verifies the working tree is clean and records the source SHA. The run record is written to `artifacts/benchmark.json` after evaluation.

```sh
python -m filmmatch.download ml-32m
python -m filmmatch freeze --archive data/ml-32m.zip --split artifacts/split-32m.json
python -m filmmatch train --archive data/ml-32m.zip --split artifacts/split-32m.json --artifacts artifacts/model-32m --publish
python -m filmmatch eval --archive data/ml-32m.zip --split artifacts/split-32m.json --artifacts artifacts/model-32m --publish
```

Select users with a seeded hash before splitting them into disjoint fit, validation, and test cohorts. Use global chronological cutoffs. Each evaluated profile receives the same popular prompts. A prior rating answers a prompt and an absent rating is a simulated skip. Relevant items have a later rating of at least four stars, were available in the fit catalog, and lack a prior rating from that user. The report includes eligible user counts and a paired bootstrap interval for the difference in NDCG@10. The small archive exercises the pipeline locally. [Committed benchmark records](benchmarks/README.md)

## Tests and deployment

```sh
python -m ruff check filmmatch tests
python -m pytest -q
cd frontend
npx playwright install chromium
npm run test:e2e
```

Browser tests expect the built app at `http://127.0.0.1:8000`. Set `FILMMATCH_E2E_URL` to test another deployment and `FILMMATCH_BROWSER_CHANNEL` to use an installed Chromium channel.

Set up an Always Free VM with Docker Compose and a TLS reverse proxy to port 8000. Clone the GitHub repository into `~/filmmatch` on the VM. Set a private `FILMMATCH_DB_PASSWORD` in its `.env` using URL-safe characters. Configure the GitHub `production` environment with the variables `FILMMATCH_DEPLOY_HOST` (`user@host`) and `FILMMATCH_PUBLIC_URL`, and the secrets `FILMMATCH_DEPLOY_KEY` and `FILMMATCH_KNOWN_HOSTS`. On a successful push to `main`, CI deploys that tested revision and verifies public health and evidence. Run the desktop and mobile smoke tests against the resulting URL to close the release gate.

## Data credit

FilmMatch uses MovieLens data from GroupLens Research. Use and redistribution of MovieLens data and transformations follow the respective dataset terms. This non-commercial project has no affiliation with GroupLens or the University of Minnesota. [Latest Small terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) · [32M terms](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems. [Paper](https://doi.org/10.1145/2827872)
