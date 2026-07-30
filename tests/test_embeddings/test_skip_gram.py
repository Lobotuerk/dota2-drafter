"""Tests for skip_gram module."""

import torch
from pathlib import Path
from dota2drafter.embeddings.skip_gram import SkipGramModel, SkipGramDataset
from dota2drafter.embeddings.data_extractor import SkipGramPair


def test_skip_gram_model_init() -> None:
    model = SkipGramModel(num_heroes=127, embed_dim=64)

    assert model.num_heroes == 127
    assert model.embed_dim == 64
    assert model.target_embedding.num_embeddings == 128
    assert model.context_embedding.num_embeddings == 128


def test_skip_gram_forward_no_negatives() -> None:
    model = SkipGramModel(num_heroes=20, embed_dim=32)
    model.eval()

    centers = torch.tensor([1, 5, 10])
    contexts = torch.tensor([2, 6, 11])

    scores = model(centers, contexts)
    assert scores.shape == (3,)


def test_skip_gram_forward_with_negatives() -> None:
    model = SkipGramModel(num_heroes=20, embed_dim=32)
    model.eval()

    centers = torch.tensor([1, 5])
    contexts = torch.tensor([2, 6])
    negatives = torch.tensor([[3, 4], [7, 8]])

    pos_scores, neg_scores = model(centers, contexts, negatives)
    assert pos_scores.shape == (2,)
    assert neg_scores.shape == (2, 2)


def test_skip_gram_loss_computation() -> None:
    model = SkipGramModel(num_heroes=20, embed_dim=32)

    pos_scores = torch.tensor([0.5, 0.3])
    neg_scores = torch.tensor([[-0.2, -0.1], [0.1, -0.3]])

    loss = model.compute_loss(pos_scores, neg_scores)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_skip_gram_dataset() -> None:
    pairs = [
        SkipGramPair(center=1, context=2, negative=3),
        SkipGramPair(center=5, context=6, negative=7),
    ]

    dataset = SkipGramDataset(pairs)
    assert len(dataset) == 2

    center, context, neg = dataset[0]
    assert center == 1
    assert context == 2
    assert neg == 3


def test_skip_gram_train_epoch() -> None:
    model = SkipGramModel(num_heroes=20, embed_dim=32)

    pairs = [
        SkipGramPair(center=i % 20 + 1, context=(i + 1) % 20 + 1, negative=(i + 2) % 20 + 1)
        for i in range(50)
    ]
    dataset = SkipGramDataset(pairs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model.train_epoch(dataloader, optimizer)

    assert loss > 0
    assert loss < 10.0


def test_skip_gram_save_load(tmp_path: Path) -> None:  # type: ignore[name-defined]
    model = SkipGramModel(num_heroes=20, embed_dim=32)
    save_path = tmp_path / "skipgram.pt"

    model.save(save_path)
    loaded = SkipGramModel.load(save_path, num_heroes=20, embed_dim=32)

    orig_embed = model.get_embeddings()
    loaded_embed = loaded.get_embeddings()
    torch.testing.assert_close(orig_embed, loaded_embed)


def test_skip_gram_embeddings_shape() -> None:
    model = SkipGramModel(num_heroes=127, embed_dim=64)
    embeddings = model.get_embeddings()

    assert embeddings.shape == (128, 64)
