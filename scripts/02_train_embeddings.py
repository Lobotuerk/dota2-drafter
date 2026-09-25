#!/usr/bin/env python3
"""Train or predict with Unsupervised Hero Embeddings (Skip-Gram + DGI).

Wraps ``dota2drafter.embeddings.pretrainer.train_embeddings`` for training
and provides a ``predict`` mode that loads and prints hero embeddings.

Usage::

    # Train
    python scripts/02_train_embeddings.py --mode train \\
        --data_dir data --output_file models/skip_gram_dgi.pt

    # Predict (print embedding for a specific hero)
    python scripts/02_train_embeddings.py --mode predict \\
        --output_file models/skip_gram_dgi.pt --hero_id 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console
from rich.logging import RichHandler

from dota2drafter.embeddings.pretrainer import load_frozen_embeddings, train_embeddings

logger = logging.getLogger(__name__)
console = Console()


import os
from dota2drafter.config import load_config

def parse_args(config=None, args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or predict with hero embeddings (Skip-Gram + DGI).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
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
        "--output_file",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Model save/load path (default: models/skip_gram_dgi.pt)",
    )
    
    embed_dim_default = config.model.d_model if config else 64
    skip_gram_lr_default = config.training.skip_gram_lr if config else 1e-2
    dgi_lr_default = config.training.dgi_lr if config else 1e-2
    batch_size_default = config.training.batch_size if config else 256
    wilson_threshold_default = config.graph.wilson_threshold if config else 0.50
    gamma_default = config.graph.gamma if config else 0.80

    parser.add_argument("--embed_dim", type=int, default=embed_dim_default, help=f"Embedding dimension (default: {embed_dim_default})")
    parser.add_argument(
        "--skip_gram_epochs", type=int, default=10, help="Skip-Gram training epochs (default: 10)"
    )
    parser.add_argument("--dgi_epochs", type=int, default=20, help="DGI training epochs (default: 20)")
    parser.add_argument(
        "--skip_gram_lr", type=float, default=skip_gram_lr_default, help=f"Skip-Gram learning rate (default: {skip_gram_lr_default})"
    )
    parser.add_argument(
        "--dgi_lr", type=float, default=dgi_lr_default, help=f"DGI learning rate (default: {dgi_lr_default})"
    )
    parser.add_argument("--batch_size", type=int, default=batch_size_default, help=f"Batch size (default: {batch_size_default})")
    parser.add_argument(
        "--device", type=str, default=None, help='Device: "cpu" or "cuda" (auto-detect if None)'
    )
    parser.add_argument(
        "--hero_id",
        type=int,
        default=1,
        help="Hero ID to print embedding for (predict mode, default: 1)",
    )
    parser.add_argument(
        "--num_heroes", type=int, default=127, help="Number of heroes (default: 127)"
    )
    parser.add_argument(
        "--wilson_threshold", type=float, default=wilson_threshold_default, help=f"Wilson score threshold (default: {wilson_threshold_default})"
    )
    parser.add_argument(
        "--gamma", type=float, default=gamma_default, help=f"Decay factor per major patch (default: {gamma_default})"
    )
    pub_data_dir_default = (
        config.training.pub_data_dir
        if config and hasattr(config.training, "pub_data_dir")
        else "data"
    )
    parser.add_argument(
        "--pub_data_dir",
        type=str,
        default=pub_data_dir_default,
        help=f"Directory with high-MMR pub games (games_batch_*.pt) (default: {pub_data_dir_default})",
    )
    parser.add_argument(
        "--include_pubs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include high-MMR pub games in training (default: True)",
    )
    return parser.parse_args(args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    
    # Pre-parse --config to load dynamic defaults
    config_path = "config.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
            
    config = None
    if os.path.exists(config_path):
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")

    args = parse_args(config)

    if args.mode == "train":
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
            sys.exit(1)

        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print("[bold blue]Training hero embeddings...[/bold blue]")
        result = train_embeddings(
            data_dir=args.data_dir,
            output_file=args.output_file,
            embed_dim=args.embed_dim,
            skip_gram_epochs=args.skip_gram_epochs,
            dgi_epochs=args.dgi_epochs,
            skip_gram_lr=args.skip_gram_lr,
            dgi_lr=args.dgi_lr,
            batch_size=args.batch_size,
            device=args.device,
            wilson_threshold=args.wilson_threshold,
            gamma=args.gamma,
            include_pubs=args.include_pubs,
            pub_data_dir=args.pub_data_dir,
        )
        console.print(f"[bold green]Saved embeddings to: {result}[/bold green]")

    elif args.mode == "predict":
        output_path = Path(args.output_file)
        if not output_path.exists():
            console.print(f"[bold red]Error:[/bold red] Embedding file not found: {output_path}")
            sys.exit(1)

        embeddings = load_frozen_embeddings(
            weights_path=args.output_file,
            embed_dim=args.embed_dim,
            num_heroes=args.num_heroes,
        )
        hero_id = args.hero_id
        emb = embeddings.weight[hero_id]
        console.print(f"[bold]Hero {hero_id} embedding:[/bold]")
        console.print(emb.tolist())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        logger.exception("Train embeddings script failed with an error:")
        sys.exit(1)
