import argparse
import shutil
import ssl
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import certifi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an official MovieLens archive"
    )
    parser.add_argument("dataset", choices=["ml-latest-small", "ml-32m"])
    parser.add_argument("--directory", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    destination = args.directory / f"{args.dataset}.zip"
    temporary = destination.with_suffix(".partial")
    context = ssl.create_default_context(cafile=certifi.where())
    url = f"https://files.grouplens.org/datasets/movielens/{args.dataset}.zip"
    with (
        urlopen(url, context=context, timeout=90) as response,
        temporary.open("wb") as stream,
    ):
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    with ZipFile(temporary) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise ValueError(f"Invalid MovieLens archive entry {bad_entry}")
    temporary.replace(destination)
    print(f"{destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
