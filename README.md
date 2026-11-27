# FilmMatch

FilmMatch helps a new viewer find a movie from a handful of ratings. The project compares the effect of asking better questions under a fixed five-prompt budget. The public benchmark is generated from committed run records.

For a pick based on what sounds good right now, open the [Mood Quiz](https://filmmatch.onrender.com/?mood=1).

## Release 1 · fixed prompts

The app starts with popular movies and updates the top ten after each rating. Search for a title you know, rate or skip a suggested title, and filter the shortlist by genre and decade. An ItemKNN explanation names a rated movie when its similarity contributed positively. Ratings are stored in the browser and sent with recommendation requests.

On the held-out test cohort, ItemKNN scored **0.2395 NDCG@10** and popularity scored **0.2415** across **555 eligible viewers**. The paired difference was **−0.0020**, with a 95% bootstrap interval of **[−0.0058, 0.0017]**. Popularity led on the point estimate, and the interval includes zero. About **77%** of prompts in that replay were simulated skips. [Committed run](benchmarks/release-1.json)

## Release 2 · adaptive prompts and new movies

Adaptive prompts come from a popular pool and favor genres the visitor has rated least. A skip spends one prompt and removes that title from later prompts. The feed blends rating similarity with description and genre similarity and can rank new movies from content alone.

Fixed popular prompts led adaptive prompts on validation, so the app defaults to Popular and offers Adaptive as a choice. In the test replay on **555 eligible viewers**, fixed popularity reached **0.2395 NDCG@10**, adaptive reached **0.2366**, and fixed genre balance reached **0.2342**. The adaptive-minus-popularity difference was **−0.0029**, with a paired 95% bootstrap interval of **[−0.0058, −0.0001]**. The simulated skip rates were **77.0%** for popularity and **80.9%** for adaptive prompts. Description-plus-genre matching and genre-only matching both scored **0.0255** on the narrow cold-item slice of **12 test viewers**. The cold-item comparison gives limited evidence for deciding how much the descriptions help. [Committed run](benchmarks/release-2.json)

Administrators submit `{title, description, genres, year}` to `POST /api/admin/movies` with an `Authorization: Bearer` token and an `Idempotency-Key`. Poll `GET /api/admin/movies/{id}` with the same token. Postgres persists the job. A worker in the service retries transient processing errors and adds the new title to the feed without refitting the vectorizer. Render generates `FILMMATCH_ADMIN_TOKEN` in the service environment. The local Compose stack supplies a development token for integration tests.

Plot summaries come from the CMU Movie Summary Corpus under CC BY-SA. The join uses normalized title and release year and accepts unique matches. The cold-item experiment uses only uniquely matched titles from the 2012 Wikipedia snapshot whose first MovieLens rating was after the fit cutoff. The genre-only and text-plus-genre runs rank the same candidates. The prompt replay holds ItemKNN fixed across policies. [CMU corpus](https://www.cs.cmu.edu/~ark/personas/) · [Bamman, O’Connor, and Smith, ACL 2013](https://aclanthology.org/P13-1035/)

## Release 3 · Model Lab

Compare ItemKNN and the pooled PyTorch model with the same profile, or hold one model fixed and edit one rating. Model Lab shows each film’s position before and after. The app serves the exported weights with NumPy, so the production container can run both rankers without PyTorch.

On the frozen warm-item comparison with shared content scores and fixed prompts, ItemKNN scored **0.2644** and the neural model scored **0.2554** on validation NDCG@10. On **555 eligible test viewers**, ItemKNN scored **0.2410** and the neural model scored **0.2260**. The neural-minus-ItemKNN difference was **−0.0151**, with a paired 95% bootstrap interval of **[−0.0217, −0.0094]**. ItemKNN remains the default based on validation. Neural training loss continued to fall after validation performance peaked, so the export uses the first epoch. [Committed run](benchmarks/release-3.json)

Profiling identified the sparse text product and full catalog copies on the recommendation path. Keeping the catalog sparse and densifying only the handful of profile vectors reduced local API p95 from **31.65 ms** to **27.00 ms** and p50 from **19.82 ms** to **15.03 ms**. Both runs delivered all **300 requests** at an offered **20 requests per second** with identical response hashes. These are laptop Docker measurements. [Performance run](benchmarks/release-3-performance.json)

## Release 4 · Mood Quiz

Answer 12 multiple-choice questions about the feeling, intensity, story, and filters you want tonight. Every question lets you leave the preference open. The quiz returns up to ten films and lets you edit a single answer to see an updated shortlist. The quiz works without an account or ratings and doesn’t persist its answers.

Choose **A superhero story** to get a hero follow-up. Choose **Spider-Man** and **This year or last year** to find recent Spider-Man movies. The quiz looks for named heroes and aliases in titles and plot summaries. When you choose a specific hero, titles that name them rank ahead of films that mention them in the plot. Each pick identifies the matching text.

The ranker enforces explicit genre, release era, and animation filters. It scores the films that pass using hand-written genre weights and TF-IDF matches against plot summaries, with rating counts as a tie-breaker and an optional familiarity preference. Explanations show the actual genre tags or plot terms behind a pick. Titles without summaries can match on genres, and new catalog entries use the same scoring path. The rules approximate how a film might feel, and the held-out MovieLens ratings evaluate the taste-based recommenders. Mood fit has no user-labeled benchmark yet.

`GET /api/mood/questions` returns the options. Send their IDs as an `answers` object to `POST /api/mood/recommend`.

## New releases from TMDB

Set `TMDB_API_KEY` in the FilmMatch web service's Render environment to enable the import. For local Compose, set the same variable in your shell or the ignored `.env` file. The key stays on the server and Postgres must be configured.

On startup, a background job checks whether a refresh is due. It imports a bounded set of popular releases for each year since 2019 and gives the last 90 days additional coverage. It keeps released titles with usable plots and genres, maps genres to MovieLens labels, and updates records by TMDB ID. Successful refreshes are due again in seven days. Failed requests retry after 15 minutes while the app continues serving its current catalog. On Render Free, a due refresh starts once the app wakes. [TMDB movie discovery](https://developer.themoviedb.org/reference/discover-movie) · [Render free hosting](https://render.com/docs/free)

Imported movies appear in search and the Mood Quiz once they’re indexed. Choose **This year or last year** in the quiz to focus on recent releases. The taste-based feed uses the content path for imported films and MovieLens rating counts remain unchanged. `GET /api/catalog` reports whether import is enabled and how many TMDB titles are indexed. Saved benchmark results use the original frozen MovieLens splits.

## Case study

**Which five questions should a movie recommender ask?** I held the recommender and prompt budget fixed and replayed the same held-out viewers across three prompt policies. The adaptive rule traded familiar titles for genre coverage. It increased simulated skips and lost to popular prompts, so the live flow defaults to popular prompts and keeps adaptive prompts available to try. The replay treats a missing rating as a skip. [Prompt experiment](benchmarks/release-2.json)

**How do we serve new movies?** A Postgres job adds a title to the search catalog and transforms its description with the existing TF-IDF vocabulary. The ingest test verifies that the same key returns the same movie ID. The held-out cold-item sample was too small to establish a benefit from text. [Cold-item experiment](benchmarks/release-2.json)

**What earned its place in production?** ItemKNN won on validation and kept its lead on test, so I made it the default and exposed both rankers in Model Lab. Profiling then identified a sparse matrix operation that dominated the content-scoring path. Using a dense representation for only the small profile side lowered request latency while preserving the ordered response hashes. [Model comparison](benchmarks/release-3.json) · [Performance record](benchmarks/release-3-performance.json)

## Run locally

Start the Docker engine and run the Compose stack.

```sh
docker compose up --build --wait
```

Open [FilmMatch on localhost](http://127.0.0.1:8088). The `web` container builds React, loads the versioned demo catalog, and seeds Postgres on startup. The Docker build uses the committed serving artifacts. The API is bound to loopback on the host. Set `FILMMATCH_PORT` to use another host port. Postgres is reachable from the Compose network.

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

To regenerate the packaged catalog, train MovieLens Latest Small into `artifacts/catalog-deploy`, run `python -m filmmatch.content --model artifacts/catalog-deploy`, then run `python -m filmmatch.package_catalog`. The packaged manifest records the training config and content hashes. [Catalog sources and terms](models/README.md)

## Reproduce the benchmark

Run the following on the source revision identified in the committed benchmark. `--publish` verifies the working tree is clean and records the source SHA. The run record is written to `artifacts/benchmark.json` after evaluation.

```sh
python -m filmmatch.download ml-32m
python -m filmmatch freeze --archive data/ml-32m.zip --split artifacts/split-32m.json
python -m filmmatch train --archive data/ml-32m.zip --split artifacts/split-32m.json --artifacts artifacts/model-32m --publish
python -m filmmatch eval --archive data/ml-32m.zip --split artifacts/split-32m.json --artifacts artifacts/model-32m --publish
```

To reproduce the release 2 experiments, keep those artifacts and switch to the source revision in the release 2 run record. Run `python -m filmmatch.content` to fetch the CMU corpus and then `python -m filmmatch.release_two`.

For release 3, use the neural run’s source revision and install `requirements-train.txt`. Train on the same artifacts and split. Use `--device cpu` when CUDA is unavailable. Use the data archive, split, and model directory for MovieLens Latest Small to reproduce the separate browser-demo checkpoint.

```sh
python -m filmmatch.train_neural --archive data/ml-32m.zip --split artifacts/split-32m.json --model artifacts/model-32m --weights artifacts/neural-32m.npz --out artifacts/release-3.json --device cuda
```

For the API comparison, start each recorded Compose revision in turn and run `python -m filmmatch.performance --sha <served-sha> --out artifacts/performance.json --profile artifacts/profile.txt`. The harness sends the same warmed request schedule to each revision and hashes the ordered responses.

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

This product uses the TMDB API but is not endorsed or certified by TMDB. [The Movie Database](https://www.themoviedb.org/)

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems. [Paper](https://doi.org/10.1145/2827872)
