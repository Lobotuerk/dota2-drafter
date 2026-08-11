#!/usr/bin/env python3
"""Train or predict with the Hierarchical Sequence Transformer.

Wraps ``TransformerTrainer`` for training and provides a ``predict``
mode that loads the best checkpoint and runs a single match through
the model.

Prerequisites: run ``scripts/01a_gather_leagues.py`` and
``scripts/01b_gather_matches.py`` first to prepare the data, then
``scripts/01c_build_comfort.py``.

Usage::

    # Train
    python scripts/04_train_transformer.py --mode train \\
        --data_dir data \\
        --rgcn_path models/rgcn.pt \\
        --comfort_path data/player_comfort.pt \\
        --checkpoint_dir checkpoints

    # Predict
    python scripts/04_train_transformer.py --mode predict \\
        --data_dir data \\
        --rgcn_path models/rgcn.pt \\
        --comfort_path data/player_comfort.pt \\
        --checkpoint_dir checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.logging import RichHandler

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.training.transformer_trainer import TransformerTrainer, TrainingConfig
from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or predict with the Hierarchical Sequence Transformer.",
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
        "--rgcn_path",
        type=str,
        default="models/rgcn.pt",
        help="Path to RGCN weights (default: models/rgcn.pt)",
    )
    parser.add_argument(
        "--comfort_path",
        type=str,
        default="data/player_comfort.pt",
        help="Path to player_comfort.pt (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument("--d_model", type=int, default=64, help="Transformer d_model (default: 64)")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers (default: 2)"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=128,
        help="Feedforward dimension (default: 128)",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=1e-5,
        help="Learning rate for pre-trained Transformer backbone (default: 1e-5)",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=1e-3,
        help="Learning rate for linear head (default: 1e-3)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 256)")
    parser.add_argument(
        "--device", type=str, default=None, help='Device: "cpu" or "cuda" (auto-detect if None)'
    )
    parser.add_argument(
        "--num_heroes", type=int, default=127, help="Number of heroes (default: 127)"
    )
    parser.add_argument(
        "--percentile_keep", type=float, default=0.80, help="Percentile threshold to keep only top-N strongest edges (default: 0.80)"
    )
    parser.add_argument(
        "--label_smoothing_eps", type=float, default=0.15, help="Label smoothing epsilon value (default: 0.15)"
    )
    parser.add_argument(
        "--step_loss_gamma",
        type=float,
        default=0.0,
        help="Gamma for step-weighted loss (default: 0.0 to disable)",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default="true",
        help="Augmentation setting: 'true' (all 448), 'false' (none), or an integer representing the maximum number of variations allowed per original match (e.g. 5, 10, 20). (default: 'true')",
    )
    parser.add_argument(
        "--mlm_epochs", type=int, default=0, help="Number of MLM pre-training epochs (default: 0, meaning skip)"
    )
    parser.add_argument(
        "--frozen_embeddings_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to frozen skip-gram/DGI embeddings (default: models/skip_gram_dgi.pt)",
    )
    return parser.parse_args()


def load_hero_indexer(data_dir: str) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json if available."""
    indexer = HeroIndexer()
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if indexer_path.exists():
        import json
        with open(indexer_path, "r") as f:
            hero_data = json.load(f)
        # Reconstruct hero list from mapping
        heroes = [{"id": api_id, "playable": True} for api_id in hero_data.keys()]
        indexer.build_mapping(heroes)
    return indexer


def load_data(data_dir: str):
    """Load all match batches from the data directory."""
    x_drafts, y_labels, radiant_players, dire_players = [], [], [], []
    data_path = Path(data_dir)

    for pt_file in sorted(data_path.glob("drafts_batch_*.pt")):
        try:
            batch = torch.load(pt_file, weights_only=True)
        except FileNotFoundError:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not load {pt_file}, skipping.")
            continue

        for i in range(len(batch["x"])):
            x_drafts.append(batch["x"][i])
            y_labels.append(batch["y"][i])
            radiant_players.append(batch["radiant_players"][i])
            dire_players.append(batch["dire_players"][i])
            
            # Extract patch IDs if present in the dataset (backward compatibility)
            if "patch_ids" in batch:
                if "patch_ids_list" not in locals():
                    patch_ids_list = []
                patch_ids_list.append(batch["patch_ids"][i])

    if "patch_ids_list" in locals():
        return x_drafts, y_labels, radiant_players, dire_players, patch_ids_list
    return x_drafts, y_labels, radiant_players, dire_players, None


def load_h_gnn(
        rgcn_path: Path,
        frozen_embeddings_path: Path,
        max_hero_idx: int,
        d_model: int,
        data_dir: Path,
        percentile_keep: float = 0.80,
    ) -> torch.Tensor:
        """Load RGCN embeddings, dynamically extracting them if a state_dict is provided."""
        h_gnn_loaded = torch.load(rgcn_path, weights_only=True)
        if isinstance(h_gnn_loaded, dict) and any(k.startswith("rgcn_layers.") for k in h_gnn_loaded):
            from dota2drafter.embeddings.rgcn import HeroRGCN
            from dota2drafter.embeddings.data_extractor import DataExtractor

            frozen_weights = torch.load(frozen_embeddings_path, weights_only=True)

            rgcn_model = HeroRGCN.load(
                path=rgcn_path,
                frozen_embeddings=frozen_weights,
                d_model=d_model,
            )

            extractor = DataExtractor(num_heroes=max_hero_idx)
            batches = extractor.load_batches(data_dir)
            hero_graph = extractor.build_pruned_hero_graph(batches, percentile_keep=percentile_keep)

            h_gnn = rgcn_model.get_embeddings(hero_graph)
            console.print(f"[bold green]Extracted raw H_GNN embeddings of shape {tuple(h_gnn.shape)} from loaded model state_dict.[/bold green]")
            return h_gnn
        return h_gnn_loaded


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

        rgcn_path = Path(args.rgcn_path)
        if not rgcn_path.exists():
            console.print(f"[bold red]Error:[/bold red] RGCN model not found: {rgcn_path}")
            sys.exit(1)

        comfort_path = Path(args.comfort_path)
        if not comfort_path.exists():
            console.print(f"[bold red]Error:[/bold red] Comfort map not found: {comfort_path}")
            sys.exit(1)

        console.print("[bold blue]Loading data...[/bold blue]")
        x_drafts, y_labels, radiant_players, dire_players, patch_ids = load_data(args.data_dir)

        # Dynamically compute max hero index from loaded data
        max_hero_idx = args.num_heroes
        for draft in x_drafts:
            max_hero_idx = max(max_hero_idx, int(draft[:, 2].max().item()))
        console.print(f"[bold green]Detected actual maximum hero index in dataset: {max_hero_idx}[/bold green]")

        console.print("[bold blue]Loading RGCN embeddings...[/bold blue]")
        h_gnn = load_h_gnn(
            Path(args.rgcn_path),
            Path(args.frozen_embeddings_path),
            max_hero_idx,
            args.d_model,
            Path(args.data_dir),
            percentile_keep=args.percentile_keep,
        )

        console.print("[bold blue]Loading comfort map...[/bold blue]")
        player_comfort_map = torch.load(args.comfort_path, weights_only=True)

        console.print("[bold blue]Loading hero indexer...[/bold blue]")
        hero_indexer = load_hero_indexer(args.data_dir)
        player_input_dim = hero_indexer.get_contiguous_count() if hero_indexer.get_contiguous_count() > 0 else max_hero_idx
        console.print(f"[bold green]Player input dim (vocab size): {player_input_dim}[/bold green]")

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = MatchNetwork(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_heroes=max_hero_idx,
            player_input_dim=player_input_dim,
            h_gnn=h_gnn,
        ).to(device)

        # Parse augment argument
        if args.augment.lower() == "true":
            augment_val = True
        elif args.augment.lower() == "false":
            augment_val = False
        else:
            try:
                augment_val = int(args.augment)
            except ValueError:
                augment_val = True  # Default to True on invalid string

        config = TrainingConfig(
            learning_rate=args.learning_rate,
            lr_backbone=args.lr_backbone,
            lr_head=args.lr_head,
            step_loss_gamma=args.step_loss_gamma,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            device=str(device),
            checkpoint_dir=args.checkpoint_dir,
            label_smoothing_eps=args.label_smoothing_eps,
            augment=augment_val,
        )

        console.print("[bold blue]Training transformer model...[/bold blue]")
        trainer = TransformerTrainer(model, config)

        # 1. MLM Pre-training stage (if requested)
        if args.mlm_epochs > 0:
            console.print(f"[bold yellow]Stage 4a: Running MLM Pre-training for {args.mlm_epochs} epochs...[/bold yellow]")
            mlm_metrics = trainer.mlm_train(
                x_drafts=x_drafts,
                y_labels=y_labels,
                radiant_players=radiant_players,
                dire_players=dire_players,
                player_comfort_map=player_comfort_map,
                num_epochs=args.mlm_epochs,
                patch_ids=patch_ids,
            )
            console.print("[bold green]MLM Pre-training complete! Transitioning to standard fine-tuning...[/bold green]")

        # 2. Main Win-Probability Fine-tuning stage
        console.print("[bold yellow]Stage 4b: Running Win-Probability Fine-tuning...[/bold yellow]")
        metrics = trainer.train(
            x_drafts=x_drafts,
            y_labels=y_labels,
            radiant_players=radiant_players,
            dire_players=dire_players,
            player_comfort_map=player_comfort_map,
        )

        console.print(f"[bold green]Training complete. Best epoch: {metrics.best_epoch}, "
                      f"Best AUC: {metrics.best_roc_auc:.4f}[/bold green]")

    elif args.mode == "predict":
        checkpoint_dir = Path(args.checkpoint_dir)
        best_checkpoint = checkpoint_dir / "best_model.pt"

        if not best_checkpoint.exists():
            console.print(f"[bold red]Error:[/bold red] Best checkpoint not found: {best_checkpoint}")
            console.print("[bold yellow]Hint:[/bold yellow] Run training first with --mode train")
            sys.exit(1)

        rgcn_path = Path(args.rgcn_path)
        if not rgcn_path.exists():
            console.print(f"[bold red]Error:[/bold red] RGCN model not found: {rgcn_path}")
            sys.exit(1)

        comfort_path = Path(args.comfort_path)
        if not comfort_path.exists():
            console.print(f"[bold red]Error:[/bold red] Comfort map not found: {comfort_path}")
            sys.exit(1)

        console.print("[bold blue]Loading data for prediction...[/bold blue]")
        x_drafts, y_labels, radiant_players, dire_players, patch_ids = load_data(args.data_dir)

        # Dynamically compute max hero index from loaded data
        max_hero_idx = args.num_heroes
        for draft in x_drafts:
            max_hero_idx = max(max_hero_idx, int(draft[:, 2].max().item()))
        console.print(f"[bold green]Detected actual maximum hero index in dataset: {max_hero_idx}[/bold green]")

        h_gnn = load_h_gnn(
            Path(args.rgcn_path),
            Path(args.frozen_embeddings_path),
            max_hero_idx,
            args.d_model,
            Path(args.data_dir),
        )
        player_comfort_map = torch.load(args.comfort_path, weights_only=True)

        console.print("[bold blue]Loading hero indexer...[/bold blue]")
        hero_indexer = load_hero_indexer(args.data_dir)
        player_input_dim = hero_indexer.get_contiguous_count() if hero_indexer.get_contiguous_count() > 0 else max_hero_idx
        console.print(f"[bold green]Player input dim (vocab size): {player_input_dim}[/bold green]")

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = MatchNetwork(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_heroes=max_hero_idx,
            player_input_dim=player_input_dim,
            h_gnn=h_gnn,
        ).to(device)

        config = TrainingConfig(device=str(device))
        trainer_instance = TransformerTrainer(model, config)
        trainer_instance.load_checkpoint(best_checkpoint)

        console.print(f"[bold blue]Running prediction on {len(x_drafts)} matches...[/bold blue]")
        for i in range(min(5, len(x_drafts))):
            x_draft = x_drafts[i].unsqueeze(0).to(device)

            # Build player comfort tensor for this match
            comfort_rows = []
            for account_id in radiant_players[i] + dire_players[i]:
                if account_id in player_comfort_map:
                    comfort_rows.append(player_comfort_map[account_id])
                else:
                    comfort_rows.append(torch.zeros(player_input_dim))
            player_comfort = torch.stack(comfort_rows).unsqueeze(0).to(device)

            prob = model.predict_proba(x_draft, player_comfort)
            console.print(f"  Match {i}: win probability = {prob.item():.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        logger.exception("Train transformer script failed with an error:")
        sys.exit(1)
