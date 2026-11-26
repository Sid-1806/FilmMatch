import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .__main__ import clean_sha
from .benchmark import digest, load_data, ndcg, write_json
from .content import ITEM_WEIGHT, ContentIndex, match_descriptions
from .model import Recommender
from .neural import NeuralRanker
from .release_two import summaries

DIMENSION = 32
HIDDEN = 64
MAX_TARGETS = 24
BATCH_SIZE = 1024
MAX_EPOCHS = 12
PATIENCE = 3
LEARNING_RATE = 0.001
SEQUENCE = 5


class PooledNetwork(nn.Module):
    def __init__(self, movies: int):
        super().__init__()
        self.input = nn.Embedding(movies + 1, DIMENSION, padding_idx=0)
        self.hidden = nn.Linear(DIMENSION, HIDDEN)
        self.out = nn.Linear(HIDDEN, DIMENSION)
        self.items = nn.Embedding(movies, DIMENSION)
        self.bias = nn.Embedding(movies, 1)
        nn.init.normal_(self.input.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.zeros_(self.bias.weight)

    def encode(self, ids, strengths):
        weighted = self.input(ids) * strengths.unsqueeze(-1)
        pooled = weighted.sum(1) / strengths.abs().sum(1, keepdim=True).clamp_min(0.5)
        return self.out(self.hidden(pooled).relu())

    def forward(self, ids, strengths, targets):
        features = self.encode(ids, strengths)
        return (features * self.items(targets)).sum(1) + self.bias(targets).squeeze(1)

    def export(self, movie_ids: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "movie_ids": movie_ids,
            **{
                key: tensor.detach().cpu().numpy().copy()
                for key, tensor in {
                    "input": self.input.weight,
                    "hidden_w": self.hidden.weight,
                    "hidden_b": self.hidden.bias,
                    "out_w": self.out.weight,
                    "out_b": self.out.bias,
                    "items": self.items.weight,
                    "bias": self.bias.weight.squeeze(1),
                }.items()
            },
        }


def by_user(events: np.ndarray):
    rows = events[np.lexsort((events[:, 3], events[:, 0]))]
    return np.split(rows, np.flatnonzero(np.diff(rows[:, 0])) + 1)


def examples(
    model: Recommender, ratings: np.ndarray, split: dict
) -> torch.utils.data.TensorDataset:
    fit = ratings[
        (ratings[:, 3] < split["cutoffs"][0])
        & np.isin(ratings[:, 0], split["fit_users"])
    ]
    rng = np.random.default_rng(split["seed"])
    groups = by_user(fit)
    count = sum(min(max(0, len(rows) - 1), MAX_TARGETS) for rows in groups)
    ids = np.zeros((count, SEQUENCE), dtype=np.int32)
    stars = np.zeros((count, SEQUENCE), dtype=np.float32)
    targets = np.empty(count, dtype=np.int64)
    labels = np.empty(count, dtype=np.float32)
    i = 0
    for rows in groups:
        if len(rows) <= 1:
            continue
        movie_positions = [model.index[int(movie)] for movie in rows[:, 1]]
        strengths = rows[:, 2] / 2 - 3
        stops = (
            rng.choice(len(rows) - 1, min(MAX_TARGETS, len(rows) - 1), replace=False)
            + 1
        )
        for t in stops:
            earlier = int(np.searchsorted(rows[:, 3], rows[t, 3], side="left"))
            if not earlier:
                continue
            start = max(0, earlier - SEQUENCE)
            length = earlier - start
            ids[i, :length] = [pos + 1 for pos in movie_positions[start:earlier]]
            stars[i, :length] = strengths[start:earlier]
            targets[i] = movie_positions[t]
            labels[i] = rows[t, 2] >= 8
            i += 1
    if not i:
        raise ValueError("No observed training targets have strictly earlier ratings.")
    return torch.utils.data.TensorDataset(
        *(torch.from_numpy(value[:i]) for value in (ids, stars, targets, labels))
    )


def prepare_replay(
    model: Recommender,
    content: ContentIndex,
    ratings: np.ndarray,
    split: dict,
    cohort: str,
    positions: np.ndarray,
):
    start, end = (
        split["cutoffs"] if cohort == "validation" else (split["cutoffs"][1], None)
    )
    warm = {model.movies[pos].id for pos in positions}
    prompts = [movie.id for movie in model.prompts(set())]
    cases = []
    for events in by_user(ratings[np.isin(ratings[:, 0], split[f"{cohort}_users"])]):
        history = {
            int(movie): float(stars) / 2 for _, movie, stars, ts in events if ts < start
        }
        relevant = {
            int(movie)
            for _, movie, stars, ts in events
            if ts >= start
            and (end is None or ts < end)
            and stars >= 8
            and movie in warm
            and movie not in history
        }
        if not relevant:
            continue
        profile = {movie: history[movie] for movie in prompts if movie in history}
        text, _ = content.score(profile, positions=positions)
        item, strengths = model.score(profile)
        hybrid = (
            ITEM_WEIGHT
            * item[positions]
            / (sum(abs(v) for v in strengths.values()) or 1)
            + (1 - ITEM_WEIGHT) * text
        )
        cases.append((profile, relevant, (1 - ITEM_WEIGHT) * text, hybrid))
    return cases


def scores(cases, candidate_ids, positions, neural: NeuralRanker | None = None):
    result = []
    for profile, relevant, text, baseline in cases:
        if neural is None:
            ranking = baseline
        else:
            vector = neural.profile_vector(profile)
            ranking = (
                text
                if vector is None
                else ITEM_WEIGHT * neural.items(vector, positions) + text
            )
        order = np.argsort(-ranking, kind="stable")
        picks = [
            int(candidate_ids[pos])
            for pos in order[: 10 + len(profile)]
            if candidate_ids[pos] not in profile
        ][:10]
        result.append(ndcg(picks, relevant))
    return result


def main() -> None:
    cli = argparse.ArgumentParser(
        description="Train and evaluate the pooled FilmMatch challenger"
    )
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--split", type=Path, required=True)
    cli.add_argument("--model", type=Path, required=True)
    cli.add_argument("--corpus", type=Path, default=Path("data/MovieSummaries.tar.gz"))
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = cli.parse_args()
    source = clean_sha()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    training = json.loads((args.model / "training.json").read_text(encoding="utf-8"))
    if (
        digest(args.archive) != split["archive_sha256"]
        or digest(args.split) != training["split_sha256"]
    ):
        cli.error("Use the frozen split and matching ItemKNN artifacts.")
    torch.manual_seed(split["seed"])
    torch.set_num_threads(2)
    model = Recommender.load(args.model)
    device = torch.device(args.device)
    print(f"Loading {args.archive.name} for {device} training", flush=True)
    _, ratings = load_data(args.archive, split["sample_limit"], split["seed"])
    content = ContentIndex(model.movies, match_descriptions(args.corpus, model.movies))
    positions = model.popular[model.rating_counts[model.popular] > 0]
    ids = content.movie_ids[positions]
    validation = prepare_replay(model, content, ratings, split, "validation", positions)
    if not validation:
        raise SystemExit("The frozen validation slice has no eligible users.")
    dataset = examples(model, ratings, split)
    print(
        f"Training on {len(dataset)} observed labels with {len(validation)} validation viewers",
        flush=True,
    )
    batches = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    net = PooledNetwork(len(model.movies)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    loss = nn.BCEWithLogitsLoss()
    best_score, best_epoch, best_state = -1.0, 0, None
    epochs = []
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        running_loss = 0.0
        for profile, strengths, targets, labels in batches:
            optimizer.zero_grad(set_to_none=True)
            value = loss(
                net(profile.to(device), strengths.to(device), targets.to(device)),
                labels.to(device),
            )
            value.backward()
            optimizer.step()
            running_loss += float(value.detach()) * len(labels)
        neural = NeuralRanker(net.export(content.movie_ids))
        score = float(np.mean(scores(validation, ids, positions, neural)))
        epochs.append(
            {
                "epoch": epoch,
                "bce": running_loss / len(dataset),
                "validation_ndcg_at_10": score,
            }
        )
        print(epochs[-1], flush=True)
        if score > best_score:
            best_state, best_score, best_epoch = (
                copy.deepcopy(net.state_dict()),
                score,
                epoch,
            )
        if epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    net.eval()
    artifact = net.export(content.movie_ids)
    neural = NeuralRanker(artifact)
    anchor = int(ids[0])
    torch_profile = torch.tensor([[model.index[anchor] + 1]], device=device)
    with torch.no_grad():
        vector = net.encode(torch_profile, torch.tensor([[2.0]], device=device))
        output = 2 * torch.sigmoid(vector @ net.items.weight.T + net.bias.weight.T) - 1
    parity_error = float(
        np.max(np.abs(output.cpu().numpy().ravel() - neural.score({anchor: 5})))
    )
    if parity_error > 1e-5:
        raise SystemExit(f"Export did not match PyTorch (max error {parity_error}).")
    compared = {}
    for cohort, cases in (
        ("validation", validation),
        ("test", prepare_replay(model, content, ratings, split, "test", positions)),
    ):
        compared[cohort] = summaries(
            {
                "itemknn": scores(cases, ids, positions),
                "neural": scores(cases, ids, positions, neural),
            },
            "neural",
            split["seed"],
        )
    winner = (
        "neural"
        if compared["validation"]["ndcg_at_10"]["neural"]
        > compared["validation"]["ndcg_at_10"]["itemknn"]
        else "itemknn"
    )
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.weights, **artifact)
    write_json(
        args.out,
        {
            "dataset": args.archive.stem,
            "archive_sha256": split["archive_sha256"],
            "split_sha256": training["split_sha256"],
            "corpus_sha256": digest(args.corpus),
            "code_sha": source,
            "seed": split["seed"],
            "device": args.device,
            "torch_version": torch.__version__,
            "model_sha256": digest(args.weights),
            "training_examples": len(dataset),
            "best_epoch": best_epoch,
            "export_max_abs_error": parity_error,
            "selection": winner,
            "metrics": compared,
            "epochs": epochs,
            "config": {
                "dimension": DIMENSION,
                "hidden": HIDDEN,
                "max_targets_per_user": MAX_TARGETS,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "lr": LEARNING_RATE,
                "optimizer": "AdamW",
                "loss": "BCEWithLogitsLoss",
            },
        },
    )
    print(json.dumps({"selection": winner, "metrics": compared}, indent=2))


if __name__ == "__main__":
    main()
