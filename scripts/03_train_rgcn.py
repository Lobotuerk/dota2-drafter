#!/usr/bin/env python3
"""Train or predict with the Relational GNN (HeroRGCN).

Wraps ``dota2drafter.embeddings.train_rgcn`` for training and provides
a ``predict`` mode that loads the RGCN model and extracts structural
hero embeddings.

Usage::

    # Train
    python scripts/03_train_rgcn.py --mode train \\
        --data_dir data \\
        --frozen_embeddings_path models/skip_gram_dgi.pt \\
        --output_file models/rgcn.pt

    # Predict (extract hero embeddings)
    python scripts/03_train_rgcn.py --mode predict \\
        --data_dir data \\
        --frozen_embeddings_path models/skip_gram_dgi.pt \\
        --rgcn_path models/rgcn.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.logging import RichHandler

from dota2drafter.embeddings.data_extractor import DataExtractor
from dota2drafter.embeddings.train_rgcn import load_rgcn_embeddings, train_rgcn

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or predict with the Relational GNN (HeroRGCN).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="Mode: train or predict (default: train)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches (default: data)",
    )
    parser.add_argument(
        "--frozen_embeddings_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to DGI embeddings from stage 02 (default: models/skip_gram_dgi.pt)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="models/rgcn.pt",
        help="Path to save the trained RGCN model (train mode, default: models/rgcn.pt)",
    )
    parser.add_argument(
        "--rgcn_path",
        type=str,
        default="models/rgcn.pt",
        help="Path to loaded RGCN model weights (predict mode, default: models/rgcn.pt)",
    )
    parser.add_argument("--d_model", type=int, default=64, help="Embedding dimension (default: 64)")
    parser.add_argument(
        "--num_relations", type=int, default=3, help="Number of edge types (default: 3)"
    )
    parser.add_argument(
        "--rgcn_epochs", type=int, default=20, help="RGCN training epochs (default: 20)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1.5e-3, help="Learning rate (default: 1.5e-3)"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of RGCN layers (default: 2)"
    )
    parser.add_argument(
        "--percentile_keep", type=float, default=0.80, help="Percentile threshold to keep only top-N strongest edges (default: 0.80)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help='Device: "cpu" or "cuda" (auto-detect if None)'
    )
    parser.add_argument(
        "--num_heroes", type=int, default=127, help="Number of heroes (default: 127)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    args = parse_args()

    if args.mode == "train":
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
            sys.exit(1)

        frozen_path = Path(args.frozen_embeddings_path)
        if not frozen_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Frozen embeddings not found: {frozen_path}"
            )
            sys.exit(1)

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]Training RGCN model...[/bold blue]")
        result = train_rgcn(
            data_dir=args.data_dir,
            frozen_embeddings_path=args.frozen_embeddings_path,
            output_file=args.output_file,
            d_model=args.d_model,
            num_relations=args.num_relations,
            rgcn_epochs=args.rgcn_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            num_layers=args.num_layers,
            percentile_keep=args.percentile_keep,
        )
        console.print(f"[bold green]Saved RGCN model to: {result}[/bold green]")

    elif args.mode == "predict":
        frozen_path = Path(args.frozen_embeddings_path)
        if not frozen_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Frozen embeddings not found: {frozen_path}"
            )
            sys.exit(1)

        rgcn_path = Path(args.rgcn_path)
        if not rgcn_path.exists():
            console.print(f"[bold red]Error:[/bold red] RGCN model not found: {rgcn_path}")
            sys.exit(1)

        model = load_rgcn_embeddings(
            model_path=args.rgcn_path,
            frozen_embeddings_path=args.frozen_embeddings_path,
            d_model=args.d_model,
            num_heroes=args.num_heroes,
            num_relations=args.num_relations,
        )

        # Build the hero graph and extract embeddings
        data_dir = Path(args.data_dir)
        extractor = DataExtractor(num_heroes=args.num_heroes)
        batches = extractor.load_batches(data_dir)
        hero_graph = extractor.build_hero_graph(batches)

        with torch.no_grad():
            h_gnn = model.get_embeddings(hero_graph)

        console.print(f"[bold blue]Extracted RGCN embeddings: shape {h_gnn.shape}[/bold blue]")
        console.print(h_gnn)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        logger.exception("Train RGCN script failed with an error:")
        sys.exit(1)
