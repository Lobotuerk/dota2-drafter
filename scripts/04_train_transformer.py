#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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

import argparse
import logging
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.logging import RichHandler

# Disable optimized Scaled Dot Product Attention (SDPA) backends (FlashAttention, Memory-Efficient)
# and force stable 'math_sdp' fallback. This prevents CUDA crashes (e.g. CUDA error: unknown error)
# on newer GPU architectures and virtualized environments like WSL2, with zero impact on small sequence lengths.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = True

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.training.transformer_trainer import TransformerTrainer, TrainingConfig
from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)
console = Console()


import os
from dota2drafter.config import load_config

def parse_args(config=None, args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or predict with the Hierarchical Sequence Transformer.",
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
    
    d_model_default = config.model.d_model if config else 64
    nhead_default = config.model.nhead if config else 4
    num_layers_default = config.model.num_layers_transformer if config else 2
    dim_feedforward_default = config.model.dim_feedforward if config else 128
    dropout_default = config.model.dropout if config else 0.1
    learning_rate_default = config.training.learning_rate if config else 1e-4
    batch_size_default = config.training.batch_size if config else 16
    wilson_threshold_default = config.graph.wilson_threshold if config else 0.50
    gamma_default = config.graph.gamma if config else 0.80
    label_smoothing_eps_default = config.training.label_smoothing_eps if config else 0.15
    step_loss_gamma_default = config.training.step_loss_gamma if config else 0.0
    augment_default = str(config.training.augment) if config else "false"
    checkpoint_metric_default = (
        config.training.checkpoint_metric
        if config and hasattr(config.training, "checkpoint_metric")
        else "val_auc"
    )
    aw_tau_start_default = config.training.aw_tau_start if config and hasattr(config.training, "aw_tau_start") else 0.15
    aw_tau_end_default = config.training.aw_tau_end if config and hasattr(config.training, "aw_tau_end") else 0.08
    aw_tau_decay_epochs_default = config.training.aw_tau_decay_epochs if config and hasattr(config.training, "aw_tau_decay_epochs") else 50
    aw_clip_min_default = config.training.aw_clip_min if config and hasattr(config.training, "aw_clip_min") else 0.1
    aw_clip_max_default = config.training.aw_clip_max if config and hasattr(config.training, "aw_clip_max") else 10.0
    stage_default = config.training.stage if config and hasattr(config.training, "stage") else 1
    draft_sample_weight_default = (
        config.training.draft_sample_weight
        if config and hasattr(config.training, "draft_sample_weight")
        else 5.0
    )
    pub_data_dir_default = (
        config.training.pub_data_dir
        if config and hasattr(config.training, "pub_data_dir")
        else "data"
    )

    parser.add_argument("--d_model", type=int, default=d_model_default, help=f"Transformer d_model (default: {d_model_default})")
    parser.add_argument("--nhead", type=int, default=nhead_default, help=f"Number of attention heads (default: {nhead_default})")
    parser.add_argument(
        "--num_layers", type=int, default=num_layers_default, help=f"Number of transformer layers (default: {num_layers_default})"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=dim_feedforward_default,
        help=f"Feedforward dimension (default: {dim_feedforward_default})",
    )
    parser.add_argument("--dropout", type=float, default=dropout_default, help=f"Dropout rate (default: {dropout_default})")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--patience", type=int, default=25, help="Early stopping patience (default: 25)"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=stage_default,
        choices=[1, 2],
        help=f"Two-stage pipeline stage: 1 for Value Head, 2 for Policy Head (default: {stage_default})",
    )
    parser.add_argument(
        "--pub_data_dir",
        type=str,
        default=pub_data_dir_default,
        help=f"Directory for high-MMR pub games batches (default: {pub_data_dir_default})",
    )
    parser.add_argument(
        "--draft_sample_weight",
        type=float,
        default=draft_sample_weight_default,
        help=f"Sample weight multiplier for draft games in Stage 1 fine-tuning (default: {draft_sample_weight_default})",
    )
    parser.add_argument(
        "--pub_epochs",
        type=int,
        default=None,
        help="Number of epochs for pub games pre-training in Stage 1 (default: num_epochs)",
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint for Stage 2 training (default: checkpoint_dir/stage1_best_model.pt)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=learning_rate_default, help=f"Learning rate (default: {learning_rate_default})"
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
    parser.add_argument("--batch_size", type=int, default=batch_size_default, help=f"Batch size (default: {batch_size_default})")
    parser.add_argument(
        "--device", type=str, default=None, help='Device: "cpu" or "cuda" (auto-detect if None)'
    )
    parser.add_argument(
        "--num_heroes", type=int, default=127, help="Number of heroes (default: 127)"
    )
    parser.add_argument(
        "--wilson_threshold", type=float, default=wilson_threshold_default, help=f"Wilson Score threshold for pruning edges (default: {wilson_threshold_default})"
    )
    parser.add_argument(
        "--gamma", type=float, default=gamma_default, help=f"Decay factor per major patch (default: {gamma_default})"
    )
    parser.add_argument(
        "--label_smoothing_eps", type=float, default=label_smoothing_eps_default, help=f"Label smoothing epsilon value (default: {label_smoothing_eps_default})"
    )
    parser.add_argument(
        "--step_loss_gamma",
        type=float,
        default=step_loss_gamma_default,
        help=f"Gamma for step-weighted loss (default: {step_loss_gamma_default})",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default=augment_default,
        help=f"Augmentation setting (default: '{augment_default}')",
    )
    parser.add_argument(
        "--slot_tau_start",
        type=float,
        default=0.30,
        help="Starting temperature for slot routing annealing (default: 0.30)",
    )
    parser.add_argument(
        "--slot_tau_end",
        type=float,
        default=0.05,
        help="Ending temperature for slot routing annealing (default: 0.05)",
    )
    parser.add_argument(
        "--slot_tau_decay_epochs",
        type=int,
        default=100,
        help="Number of epochs to decay slot temperature over (default: 100)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name to log metrics (e.g., 'dota2-drafter'). If not provided, wandb is disabled.",
    )
    parser.add_argument(
        "--frozen_embeddings_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to frozen skip-gram/DGI embeddings (default: models/skip_gram_dgi.pt)",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        choices=["val_auc", "val_top5_acc", "val_loss"],
        default=checkpoint_metric_default,
        help=(
            f"Metric for checkpoint selection and early stopping: 'val_auc', 'val_top5_acc', "
            f"or 'val_loss' (default: '{checkpoint_metric_default}')"
        ),
    )
    parser.add_argument(
        "--aw_tau_start",
        type=float,
        default=aw_tau_start_default,
        help=f"Starting temperature for AW-MLM annealing (default: {aw_tau_start_default})",
    )
    parser.add_argument(
        "--aw_tau_end",
        type=float,
        default=aw_tau_end_default,
        help=f"Ending temperature for AW-MLM annealing (default: {aw_tau_end_default})",
    )
    parser.add_argument(
        "--aw_tau_decay_epochs",
        type=int,
        default=aw_tau_decay_epochs_default,
        help=f"Epochs to decay AW-MLM temperature over (default: {aw_tau_decay_epochs_default})",
    )
    parser.add_argument(
        "--aw_clip_min",
        type=float,
        default=aw_clip_min_default,
        help=f"Minimum clip weight for AW-MLM (default: {aw_clip_min_default})",
    )
    parser.add_argument(
        "--aw_clip_max",
        type=float,
        default=aw_clip_max_default,
        help=f"Maximum clip weight for AW-MLM (default: {aw_clip_max_default})",
    )
    return parser.parse_args(args)


def load_hero_indexer(data_dir: str) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json."""
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found at: {indexer_path}")
    import json
    with open(indexer_path) as f:
        hero_data = json.load(f)
    # Reconstruct hero list from mapping
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    indexer = HeroIndexer()
    indexer.build_mapping(heroes)
    return indexer


def load_data(data_dir: str):
    """Load all match batches from the data directory."""
    x_drafts, y_labels, radiant_players, dire_players = [], [], [], []
    patch_ids_list = []
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
            
            # Extract patch IDs if present in the dataset
            if "patch_ids" in batch:
                patch_ids_list.append(batch["patch_ids"][i].item() if hasattr(batch["patch_ids"][i], 'item') else batch["patch_ids"][i])

    if patch_ids_list:
        return x_drafts, y_labels, radiant_players, dire_players, patch_ids_list
        
    return x_drafts, y_labels, radiant_players, dire_players, None


def load_pub_data(data_dir: str):
    """Load high-MMR pub match batches from the data directory (games_batch_*.pt)."""
    x_pubs, y_pubs = [], []
    patch_ids_list = []
    data_path = Path(data_dir)

    pub_files = sorted(data_path.glob("games_batch_*.pt"))
    if not pub_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No pub game batches (games_batch_*.pt) found in {data_dir}.")
        return [], [], None

    for pt_file in pub_files:
        try:
            batch = torch.load(pt_file, weights_only=True)
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] Could not load {pt_file}: {e}, skipping.")
            continue

        for i in range(len(batch["x"])):
            x_pubs.append(batch["x"][i])
            y_pubs.append(batch["y"][i])
            if "patch_ids" in batch:
                patch_ids_list.append(
                    batch["patch_ids"][i].item()
                    if hasattr(batch["patch_ids"][i], "item")
                    else batch["patch_ids"][i]
                )

    patch_ids = patch_ids_list if patch_ids_list else None
    return x_pubs, y_pubs, patch_ids


def load_h_gnn(
        rgcn_path: Path,
        frozen_embeddings_path: Path,
        max_hero_idx: int,
        d_model: int,
        data_dir: Path,
        wilson_threshold: float = 0.50,
        gamma: float = 0.80,
        device: torch.device | str = "cpu",
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
            hero_graph = extractor.build_pruned_hero_graph(
                batches,
                wilson_threshold=wilson_threshold,
                gamma=gamma,
            )

            h_gnn = rgcn_model.get_embeddings(hero_graph, device=torch.device(device))
            console.print(f"[bold green]Extracted raw H_GNN embeddings of shape {tuple(h_gnn.shape)} from loaded model state_dict.[/bold green]")
            return h_gnn
        return h_gnn_loaded


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

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        console.print("[bold blue]Loading RGCN embeddings...[/bold blue]")
        h_gnn = load_h_gnn(
            Path(args.rgcn_path),
            Path(args.frozen_embeddings_path),
            max_hero_idx,
            args.d_model,
            Path(args.data_dir),
            wilson_threshold=args.wilson_threshold,
            gamma=args.gamma,
            device=device,
        )

        console.print("[bold blue]Loading comfort map...[/bold blue]")
        player_comfort_map = torch.load(args.comfort_path, weights_only=True)

        console.print("[bold blue]Loading hero indexer...[/bold blue]")
        hero_indexer = load_hero_indexer(args.data_dir)
        # The comfort matrix was built using the full vocab size from hero_indexer (which can be larger than max_hero_idx seen in current batches)
        if len(player_comfort_map) > 0:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)
        else:
            player_input_dim = max_hero_idx * 2
        
        console.print(f"[bold green]Player input dim (vocab size * 2): {player_input_dim}[/bold green]")

        num_patches_dynamic = 30
        if patch_ids is not None:
            max_p = max([p.item() if hasattr(p, 'item') else p for p in patch_ids])
            num_patches_dynamic = max(30, max_p + 10)

        model = MatchNetwork(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_heroes=max_hero_idx,
            player_input_dim=player_input_dim,
            h_gnn=h_gnn,
            num_patches=num_patches_dynamic,
        ).to(device)

        # Parse augment argument
        if isinstance(args.augment, str):
            if args.augment.lower() == "true":
                augment_val = True
            elif args.augment.lower() == "false":
                augment_val = False
            else:
                try:
                    augment_val = int(args.augment)
                except ValueError:
                    augment_val = False
        else:
            augment_val = args.augment

        # Load pub games for Stage 1 if requested
        x_pubs, y_pubs, patch_ids_pubs = [], [], None
        if args.stage == 1:
            console.print(f"[bold blue]Loading high-MMR pub games from {args.pub_data_dir}...[/bold blue]")
            x_pubs, y_pubs, patch_ids_pubs = load_pub_data(args.pub_data_dir)
            if not x_pubs:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] No pub games found in {args.pub_data_dir}. "
                    "Stage 1 will proceed directly to fine-tuning on draft games."
                )

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
            slot_tau_start=args.slot_tau_start,
            slot_tau_end=args.slot_tau_end,
            slot_tau_decay_epochs=args.slot_tau_decay_epochs,
            patience=args.patience,
            checkpoint_metric=args.checkpoint_metric,
            aw_tau_start=args.aw_tau_start,
            aw_tau_end=args.aw_tau_end,
            aw_tau_decay_epochs=args.aw_tau_decay_epochs,
            aw_clip_min=args.aw_clip_min,
            aw_clip_max=args.aw_clip_max,
            stage=args.stage,
            draft_sample_weight=args.draft_sample_weight,
            pub_data_dir=args.pub_data_dir,
            pub_epochs=args.pub_epochs,
            stage1_checkpoint_path=args.stage1_checkpoint,
        )

        if args.wandb_project:
            try:
                import wandb
                wandb.init(project=args.wandb_project, config=vars(args))
            except ImportError:
                console.print("[bold yellow]Warning:[/bold yellow] wandb package not found. Install it to log metrics.")

        console.print(f"[bold blue]Training transformer model (Stage {args.stage})...[/bold blue]")
        trainer = TransformerTrainer(model, config)

        metrics = trainer.train(
            x_drafts=x_drafts,
            y_labels=y_labels,
            radiant_players=radiant_players,
            dire_players=dire_players,
            player_comfort_map=player_comfort_map,
            patch_ids=patch_ids,
            x_pubs=x_pubs if args.stage == 1 else None,
            y_pubs=y_pubs if args.stage == 1 else None,
            patch_ids_pubs=patch_ids_pubs if args.stage == 1 else None,
            stage1_checkpoint_path=args.stage1_checkpoint,
        )

        if args.wandb_project:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

        score_desc = f"Best {config.checkpoint_metric}: {metrics.best_checkpoint_value:.4f}" if metrics.best_checkpoint_value != float("-inf") else f"Best Top-5: {metrics.best_mlm_top5_acc:.4f}"
        console.print(f"[bold green]Training complete. Best epoch: {metrics.best_epoch}, {score_desc}[/bold green]")

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

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        h_gnn = load_h_gnn(
            Path(args.rgcn_path),
            Path(args.frozen_embeddings_path),
            max_hero_idx,
            args.d_model,
            Path(args.data_dir),
            device=device,
        )
        player_comfort_map = torch.load(args.comfort_path, weights_only=True)

        console.print("[bold blue]Loading hero indexer...[/bold blue]")
        hero_indexer = load_hero_indexer(args.data_dir)
        # The comfort matrix was built using the full vocab size from hero_indexer (which can be larger than max_hero_idx seen in current batches)
        if len(player_comfort_map) > 0:
            first_tensor = next(iter(player_comfort_map.values()))
            player_input_dim = first_tensor.size(0)
        else:
            player_input_dim = max_hero_idx * 2
        
        console.print(f"[bold green]Player input dim (vocab size * 2): {player_input_dim}[/bold green]")

        num_patches_dynamic = 30
        if patch_ids is not None:
            max_p = max([p.item() if hasattr(p, 'item') else p for p in patch_ids])
            num_patches_dynamic = max(30, max_p + 10)

        model = MatchNetwork(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_heroes=max_hero_idx,
            player_input_dim=player_input_dim,
            h_gnn=h_gnn,
            num_patches=num_patches_dynamic,
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

            match_patch_id = None
            if patch_ids is not None:
                match_patch_id = torch.tensor([patch_ids[i]], dtype=torch.long, device=device)

            prob = model.predict_proba(x_draft, player_comfort, patch_ids=match_patch_id)
            console.print(f"  Match {i}: win probability = {prob.item():.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        logger.exception("Train transformer script failed with an error:")
        sys.exit(1)
