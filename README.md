# FilmMatch

FilmMatch helps a new viewer find a movie from a handful of ratings. The project compares the effect of asking better questions under a fixed five-prompt budget. The public benchmark is generated from committed run records.

## Release 1 · fixed prompts

The app starts with popular movies and updates the top ten after each rating. Search for a title you know, rate or skip a suggested title, and filter the shortlist by genre and decade. An ItemKNN explanation names a rated movie when its similarity contributed positively. Ratings are stored in the browser and sent with recommendation requests.

On the held-out test cohort, ItemKNN scored **0.2395 NDCG@10** and popularity scored **0.2415** across **555 eligible viewers**. The paired difference was **−0.0020**, with a 95% bootstrap interval of **[−0.0058, 0.0017]**. Popularity led on the point estimate, and the interval includes zero. About **77%** of prompts in that replay were simulated skips. [Committed run](benchmarks/release-1.json)

## Release 2 · adaptive prompts and new movies

Adaptive prompts come from a popular pool and favor genres the visitor has rated least. A skip spends one prompt and removes that title from later prompts. The feed blends rating similarity with description and genre similarity and can rank new movies from content alone.

Fixed popular prompts led adaptive prompts on validation, so the app defaults to Popular and offers Adaptive as a choice. In the test replay on **555 eligible viewers**, fixed popularity reached **0.2395 NDCG@10**, adaptive reached **0.2366**, and fixed genre balance reached **0.2342**. The adaptive-minus-popularity difference was **−0.0029**, with a paired 95% bootstrap interval of **[−0.0058, −0.0001]**. The simulated skip rates were **77.0%** for popularity and **80.9%** for adaptive prompts. Description-plus-genre matching and genre-only matching both scored **0.0255** on the narrow cold-item slice of **12 test viewers**. The cold-item comparison gives limited evidence for deciding how much the descriptions help. [Committed run](benchmarks/release-2.json)

Administrators submit `{title, description, genres, year}` to `POST /api/admin/movies` with an `Authorization: Bearer` token and an `Idempotency-Key`. Poll `GET /api/admin/movies/{id}` with the same token. Postgres persists the job. A worker in the service retries transient processing errors and adds the new title to the feed without refitting the vectorizer. Render generates `FILMMATCH_ADMIN_TOKEN` in the service environment. The local Compose stack supplies a development token for integration tests.

Plot summaries come from the CMU Movie Summary Corpus under CC BY-SA. The join uses normalized title and release year and accepts unique matches. The cold-item experiment uses only uniquely matched titles from the 2012 Wikipedia snapshot whose first MovieLens rating was after the fit cutoff. The genre-only and text-plus-genre runs rank the same candidates. The prompt replay holds ItemKNN fixed across policies. [CMU corpus](https://www.cs.cmu.edu/~ark/personas/) · [Bamman, O’Connor, and Smith, ACL 2013](https://aclanthology.org/P13-1035/)

## Run locally

Start the Docker engine and run the Compose stack.

```sh
docker compose up --build --wait
```

Open [FilmMatch on localhost](http://127.0.0.1:8088). The `web` container builds React, downloads MovieLens Latest Small, trains the development model, and seeds the Postgres search catalog on startup. The API is bound to loopback on the host. Set `FILMMATCH_PORT` to use another host port. Postgres is reachable from the Compose network.

For local development with Python 3.12 and Node 24 run these commands from the repository root.

```sh
python -m pip install -r requirements-dev.txt
python -m filmmatch.download ml-latest-small
python -m filmmatch freeze --archive data/ml-latest-small.zip
python -m filmmatch train --archive data/ml-latest-small.zip
python -m filmmatch.content
npm ci --prefix frontend
npm run build --prefix frontend
python -m uvicorn filmmatch.api:app --port 8088
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

To reproduce the release 2 experiments, keep those artifacts and switch to the source revision in the release 2 run record. Run `python -m filmmatch.content` to fetch the CMU corpus and then `python -m filmmatch.release_two`.

Select users with a seeded hash before splitting them into disjoint fit, validation, and test cohorts. Use global chronological cutoffs. Each evaluated profile receives the same popular prompts. A prior rating answers a prompt and an absent rating is a simulated skip. Relevant items have a later rating of at least four stars, were available in the fit catalog, and lack a prior rating from that user. The report includes eligible user counts and a paired bootstrap interval for the difference in NDCG@10. The small archive exercises the pipeline locally. [Committed benchmark records](benchmarks/README.md)

## Tests and deployment

```sh
python -m ruff check filmmatch tests
python -m pytest -q
cd frontend
npx playwright install chromium
npm run test:e2e
```

Browser tests expect the built app at `http://127.0.0.1:8088`. Set `FILMMATCH_E2E_URL` to test another deployment and `FILMMATCH_BROWSER_CHANNEL` to use an installed Chromium channel.

For a public deployment without a payment card, create a Neon Free Postgres database and connect the GitHub repository to Render as a Blueprint. The blueprint uses the Dockerfile and Render Free. Set `DATABASE_URL` in Render to the database's TLS connection string. Subsequent changes to `main` deploy after CI passes. Open `/api/health` and `/api/evidence` on the public URL and run the desktop and mobile smoke tests against that URL to close the release gate. The service can take about a minute to wake after idle time. [Render Free Docker hosting](https://render.com/hosting/docker) · [Connect to Neon](https://neon.com/docs/connect/connect-from-any-app)

## Data credit

FilmMatch uses MovieLens data from GroupLens Research. Use and redistribution of MovieLens data and transformations follow the respective dataset terms. This non-commercial project has no affiliation with GroupLens or the University of Minnesota. [Latest Small terms](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) · [32M terms](https://files.grouplens.org/datasets/movielens/ml-32m-README.html)

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems. [Paper](https://doi.org/10.1145/2827872)
