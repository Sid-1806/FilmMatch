# Browser-demo weights

`neural-small.npz` is the NumPy export of the MovieLens Latest Small challenger. The development run record identifies its configuration, source revision and weight hash. The MovieLens 32M comparison is the model-selection experiment. MovieLens-derived artifacts follow the dataset's non-commercial terms and are credited in the repository README.

`catalog` supplies the versioned demo assets. The movie metadata and ItemKNN neighbors are derived from [MovieLens Latest Small](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) and distributed under the included MovieLens terms.

`catalog/descriptions.json.gz` uses plot summaries by English-language Wikipedia contributors collected by David Bamman, Brendan O’Connor, and Noah A. Smith in the [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/). FilmMatch joins a subset by normalized title and year, trims descriptions and re-encodes them as compressed JSON. The resulting descriptions are distributed under [CC BY-SA 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/). The source corpus includes the source Wikipedia page IDs.
