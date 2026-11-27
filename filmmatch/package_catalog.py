import argparse
import gzip
import io
import json
import shutil
from csv import DictReader
from pathlib import Path
from zipfile import ZipFile

from .benchmark import digest, write_json


def package_links(archive: ZipFile, out: Path) -> None:
    member = next(name for name in archive.namelist() if Path(name).name == "links.csv")
    with io.TextIOWrapper(archive.open(member), encoding="utf-8") as source:
        links = {
            row["movieId"]: {
                "imdb": f"tt{row['imdbId']}" if row["imdbId"] else None,
                "tmdb": int(row["tmdbId"]) if row["tmdbId"] else None,
            }
            for row in DictReader(source)
        }
    out.write_bytes(
        gzip.compress(json.dumps(links, separators=(",", ":")).encode(), mtime=0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package the MovieLens demo catalog for Docker"
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/catalog-deploy"))
    parser.add_argument(
        "--archive", type=Path, default=Path("data/ml-latest-small.zip")
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/MovieSummaries.tar.gz")
    )
    parser.add_argument("--out", type=Path, default=Path("models/catalog"))
    args = parser.parse_args()
    training = json.loads((args.model / "training.json").read_text(encoding="utf-8"))
    if training["archive_sha256"] != digest(args.archive):
        parser.error("Catalog training and the MovieLens archive must match.")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("movies.json", "descriptions.json"):
        (args.out / f"{name}.gz").write_bytes(
            gzip.compress((args.model / name).read_bytes(), mtime=0)
        )
    shutil.copyfile(args.model / "neighbors.npz", args.out / "neighbors.npz")
    with ZipFile(args.archive) as archive:
        package_links(archive, args.out / "links.json.gz")
        readme = next(
            name for name in archive.namelist() if Path(name).name == "README.txt"
        )
        license_text = archive.read(readme).replace(b"\n=======\n", b"\n\n", 1)
        (args.out / "MOVIELENS-LICENSE.txt").write_bytes(license_text)
    names = ("movies.json.gz", "descriptions.json.gz", "neighbors.npz", "links.json.gz")
    write_json(
        args.out / "manifest.json",
        {
            "training": training,
            "corpus_sha256": digest(args.corpus),
            "sha256": {name: digest(args.out / name) for name in names},
        },
    )


if __name__ == "__main__":
    main()
