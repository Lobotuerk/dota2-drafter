#!/usr/bin/env python3
"""End-to-end Multi-Objective Hyperparameter Tuning Script using Optuna.

Tunes embedding pre-training (Stage 2), Relational GNN (Stage 3), and Hierarchical Transformer (Stage 4)
jointly by wrapping them into a single objective function. Operates on multiple objectives (e.g.,
maximizing both Top-5 MLM Accuracy and win-rate ROC-AUC score).

Usage::

    python scripts/07_tune_pipeline.py \\
        --data_dir data \\
        --comfort_path data/player_comfort.pt \\
        --n_trials 20 \\
        --skip_gram_epochs 5 \\
        --dgi_epochs 10 \\
        --rgcn_epochs 10 \\
        --transformer_epochs 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import optuna
from rich.console import Console
from rich.logging import RichHandler

# Disable SDPA backends for CUDA stability in virtual environments
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

from dota2drafter.embeddings.pretrainer import train_embeddings
from dota2drafter.embeddings.train_rgcn import train_rgcn
from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.training.transformer_trainer import TransformerTrainer, TrainingConfig
from dota2drafter.processor.hero_indexer import HeroIndexer

logger = logging.getLogger(__name__)
console = Console()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end multi-objective hyperparameter tuning for Dota 2 Drafter."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches (default: data)",
    )
    parser.add_argument(
        "--comfort_path",
        type=str,
        default="data/player_comfort.pt",
        help="Path to player_comfort.pt (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run (default: 20)",
    )
    parser.add_argument(
        "--skip_gram_epochs",
        type=int,
        default=5,
        help="Skip-Gram training epochs per trial (default: 5)",
    )
    parser.add_argument(
        "--dgi_epochs",
        type=int,
        default=10,
        help="DGI training epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--rgcn_epochs",
        type=int,
        default=10,
        help="RGCN training epochs per trial (default: 10)",
    )
    parser.add_argument(
        "--transformer_epochs",
        type=int,
        default=20,
        help="Transformer training epochs per trial (default: 20)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="dota2_pipeline_tuning",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for Optuna storage (e.g., sqlite:///optuna.db). If None, runs in-memory.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights & Biases project name to log metrics (e.g., 'dota2-drafter'). If not provided, wandb is disabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to train on (auto-detect if None)',
    )
    parser.add_argument(
        "--llm_patience",
        type=int,
        default=25,
        help="Early stopping patience for LLM/Transformer training (default: 25)",
    )
    parser.add_argument(
        "--pub_data_dir",
        type=str,
        default="data",
        help="Directory with high-MMR pub games (games_batch_*.pt) (default: data)",
    )
    parser.add_argument(
        "--include_pubs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include high-MMR pub games in Skip-Gram and RGCN training (default: True)",
    )
    parser.add_argument(
        "--two_stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run the decoupled two-stage Transformer training pipeline (default: True)",
    )
    parser.add_argument(
        "--pub_epochs",
        type=int,
        default=10,
        help="Number of epochs for pub games pre-training in Stage 1 (default: 10)",
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=None,
        help="Number of epochs for draft games fine-tuning in Stage 1 (default: transformer_epochs)",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=None,
        help="Number of epochs for AW-MLM policy training in Stage 2 (default: transformer_epochs)",
    )
    parser.add_argument(
        "--draft_sample_weight",
        type=float,
        default=5.0,
        help="Sample weight multiplier for draft games in Stage 1 fine-tuning (default: 5.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose INFO logging during trials (default: False)",
    )
    return parser.parse_args(args)


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


def load_hero_indexer(data_dir: str) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json."""
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found at: {indexer_path}")
    import json
    with open(indexer_path) as f:
        hero_data = json.load(f)
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    indexer = HeroIndexer()
    indexer.build_mapping(heroes)
    return indexer


def load_h_gnn(
    rgcn_path: Path,
    frozen_embeddings_path: Path,
    max_hero_idx: int,
    d_model: int,
    data_dir: Path,
    wilson_threshold: float = 0.50,
    gamma: float = 0.80,
    include_pubs: bool = True,
    pub_data_dir: str | Path | None = None,
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
        batches = extractor.load_batches(
            data_dir,
            include_pubs=include_pubs,
            pub_data_dir=pub_data_dir,
        )
        hero_graph = extractor.build_pruned_hero_graph(
            batches,
            wilson_threshold=wilson_threshold,
            gamma=gamma,
        )

        h_gnn = rgcn_model.get_embeddings(hero_graph, device=torch.device(device))
        return h_gnn
    return h_gnn_loaded


def make_objective(
    args: argparse.Namespace,
    x_drafts: list[torch.Tensor],
    y_labels: list[torch.Tensor],
    radiant_players: list[list[int]],
    dire_players: list[list[int]],
    patch_ids: list[int] | None,
    player_comfort_map: dict[int, torch.Tensor],
    max_hero_idx: int,
    player_input_dim: int,
    device: torch.device,
    x_pubs: list[torch.Tensor] | None = None,
    y_pubs: list[torch.Tensor] | None = None,
    patch_ids_pubs: list[int] | None = None,
):
    """Creates the objective function bound with pre-loaded training data."""
    
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # --- 1. Sample Hyperparameters ---
        # Embedding Size (shared across embeddings, RGCN, and Transformer)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        
        # Ensure attention head count divides d_model neatly
        nhead_choices = [4, 8]
        nhead = trial.suggest_categorical("nhead", nhead_choices)

        # Feed-forward size for the transformer
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [1024, 2048])
        
        # GNN and Transformer layers
        # num_layers_rgcn = trial.suggest_int("num_layers_rgcn", 1, 4, step=1)
        num_layers_rgcn = 3
        num_layers_transformer = trial.suggest_int("num_layers_transformer", 7, 13, step=1)
        
        # Transformer-specific regularization and scheduling
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
        lr_head = trial.suggest_float("lr_head", 1e-5, 1e-3, log=True)
        lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
        label_smoothing_eps = trial.suggest_float("label_smoothing_eps", 0.0, 0.25)

        # Two-stage specific hyperparameters
        if args.two_stage:
            draft_sample_weight = trial.suggest_float("draft_sample_weight", 2.0, 8.0)
            aw_tau_start = trial.suggest_float("aw_tau_start", 0.10, 0.25)
            aw_tau_end = trial.suggest_float("aw_tau_end", 0.05, 0.12)
        else:
            step_loss_gamma = trial.suggest_float("step_loss_gamma", 0.0, 1.5)
        
        # Graph building parameters
        wilson_threshold = trial.suggest_float("wilson_threshold", 0.40, 0.60)
        gamma = trial.suggest_float("gamma", 0.60, 0.90)

        # Augmentation settings: 0 (disabled), 5, 10
        augment = trial.suggest_categorical("augment", [0, 5, 25])

        # Batch size for the transformer training loader
        batch_size = trial.suggest_categorical("batch_size", [64, 128])

        # Learning rates for the different pre-training stages
        skip_gram_lr = trial.suggest_float("skip_gram_lr", 5e-4, 2e-2, log=True)
        dgi_lr = trial.suggest_float("dgi_lr", 5e-4, 2e-2, log=True)
        rgcn_lr = trial.suggest_float("rgcn_lr", 5e-5, 5e-3, log=True)

        console.print(f"\n[bold magenta]Starting Trial {trial.number}[/bold magenta]")
        console.print(f"Parameters: d_model={d_model}, dim_feedforward={dim_feedforward}, nhead={nhead}, rgcn_layers={num_layers_rgcn}, transformer_layers={num_layers_transformer}, augment={augment}, batch_size={batch_size}")

        # --- 2. Setup WandB Logging (if enabled) ---
        use_wandb = False
        if args.wandb_project:
            try:
                import wandb
                use_wandb = True
                run_name = f"trial-{trial.number}-dm{d_model}-ff{dim_feedforward}-tl{num_layers_transformer}-nh{nhead}-aug{augment}-bs{batch_size}"
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=trial.params,
                    reinit=True,
                    group=args.study_name,
                )
            except ImportError:
                console.print("[bold yellow]Warning:[/bold yellow] wandb package not found. Run 'pip install wandb' to log metrics.")

        try:
            # Use temporary directory to isolate pipeline weights per trial
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                frozen_embeddings_path = tmp_path / "skip_gram_dgi.pt"
                rgcn_output_path = tmp_path / "rgcn.pt"

                # --- Stage 1: Train Embeddings ---
                try:
                    train_embeddings(
                        data_dir=args.data_dir,
                        output_file=frozen_embeddings_path,
                        embed_dim=d_model,
                        skip_gram_epochs=args.skip_gram_epochs,
                        dgi_epochs=args.dgi_epochs,
                        skip_gram_lr=skip_gram_lr,
                        dgi_lr=dgi_lr,
                        device=str(device),
                        wilson_threshold=wilson_threshold,
                        gamma=gamma,
                        include_pubs=args.include_pubs,
                        pub_data_dir=args.pub_data_dir,
                    )
                except Exception as e:
                    console.print(f"[bold red]Stage 1 (Embeddings) failed:[/bold red] {e}")
                    raise optuna.exceptions.TrialPruned()

                # --- Stage 2: Train RGCN ---
                try:
                    train_rgcn(
                        data_dir=args.data_dir,
                        frozen_embeddings_path=frozen_embeddings_path,
                        output_file=rgcn_output_path,
                        d_model=d_model,
                        num_relations=3,
                        rgcn_epochs=args.rgcn_epochs,
                        learning_rate=rgcn_lr,
                        device=str(device),
                        num_layers=num_layers_rgcn,
                        wilson_threshold=wilson_threshold,
                        gamma=gamma,
                        include_pubs=args.include_pubs,
                        pub_data_dir=args.pub_data_dir,
                    )
                except Exception as e:
                    console.print(f"[bold red]Stage 2 (RGCN) failed:[/bold red] {e}")
                    raise optuna.exceptions.TrialPruned()

                # --- Stage 3: Train Transformer ---
                try:
                    h_gnn = load_h_gnn(
                        rgcn_path=rgcn_output_path,
                        frozen_embeddings_path=frozen_embeddings_path,
                        max_hero_idx=max_hero_idx,
                        d_model=d_model,
                        data_dir=Path(args.data_dir),
                        wilson_threshold=wilson_threshold,
                        gamma=gamma,
                        include_pubs=args.include_pubs,
                        pub_data_dir=args.pub_data_dir,
                        device=device,
                    )

                    all_patches = []
                    if patch_ids is not None:
                        all_patches.extend(patch_ids)
                    if patch_ids_pubs is not None:
                        all_patches.extend(patch_ids_pubs)
                    num_patches_dynamic = 30
                    if all_patches:
                        max_p = max([p.item() if hasattr(p, "item") else p for p in all_patches])
                        num_patches_dynamic = max(30, max_p + 10)

                    model = MatchNetwork(
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers_transformer,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        num_heroes=max_hero_idx,
                        player_input_dim=player_input_dim,
                        h_gnn=h_gnn,
                        num_patches=num_patches_dynamic,
                    ).to(device)

                    ckpt_dir = tmp_path / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                    if args.two_stage:
                        stage1_epochs = args.stage1_epochs or args.transformer_epochs
                        stage2_epochs = args.stage2_epochs or args.transformer_epochs

                        config_s1 = TrainingConfig(
                            learning_rate=learning_rate,
                            lr_head=lr_head,
                            lr_backbone=lr_backbone,
                            num_epochs=stage1_epochs,
                            pub_epochs=args.pub_epochs,
                            draft_sample_weight=draft_sample_weight,
                            batch_size=batch_size,
                            device=str(device),
                            checkpoint_dir=str(ckpt_dir),
                            label_smoothing_eps=label_smoothing_eps,
                            augment=augment,
                            patience=args.llm_patience,
                            stage=1,
                        )
                        trainer_s1 = TransformerTrainer(model, config_s1)
                        metrics_s1 = trainer_s1.train(
                            x_drafts=x_drafts,
                            y_labels=y_labels,
                            radiant_players=radiant_players,
                            dire_players=dire_players,
                            player_comfort_map=player_comfort_map,
                            patch_ids=patch_ids,
                            x_pubs=x_pubs,
                            y_pubs=y_pubs,
                            patch_ids_pubs=patch_ids_pubs,
                        )

                        config_s2 = TrainingConfig(
                            learning_rate=learning_rate,
                            lr_head=lr_head,
                            lr_backbone=lr_backbone,
                            num_epochs=stage2_epochs,
                            batch_size=batch_size,
                            device=str(device),
                            checkpoint_dir=str(ckpt_dir),
                            label_smoothing_eps=label_smoothing_eps,
                            augment=augment,
                            patience=args.llm_patience,
                            aw_tau_start=aw_tau_start,
                            aw_tau_end=aw_tau_end,
                            stage=2,
                            stage1_checkpoint_path=ckpt_dir / "stage1_best_model.pt",
                        )
                        trainer_s2 = TransformerTrainer(model, config_s2)
                        metrics_s2 = trainer_s2.train(
                            x_drafts=x_drafts,
                            y_labels=y_labels,
                            radiant_players=radiant_players,
                            dire_players=dire_players,
                            player_comfort_map=player_comfort_map,
                            patch_ids=patch_ids,
                            stage1_checkpoint_path=ckpt_dir / "stage1_best_model.pt",
                        )

                        best_top5 = metrics_s2.best_mlm_top5_acc
                        best_auc = (
                            metrics_s1.best_val_auc
                            if metrics_s1.best_val_auc > 0
                            else (max(metrics_s1.val_auc_scores) if metrics_s1.val_auc_scores else 0.5)
                        )
                    else:
                        config = TrainingConfig(
                            learning_rate=learning_rate,
                            lr_head=lr_head,
                            lr_backbone=lr_backbone,
                            step_loss_gamma=step_loss_gamma,
                            num_epochs=args.transformer_epochs,
                            batch_size=batch_size,
                            device=str(device),
                            checkpoint_dir=str(ckpt_dir),
                            label_smoothing_eps=label_smoothing_eps,
                            augment=augment,
                            patience=args.llm_patience,
                        )

                        trainer = TransformerTrainer(model, config)
                        metrics = trainer.train(
                            x_drafts=x_drafts,
                            y_labels=y_labels,
                            radiant_players=radiant_players,
                            dire_players=dire_players,
                            player_comfort_map=player_comfort_map,
                            patch_ids=patch_ids,
                        )

                        best_top5 = metrics.best_mlm_top5_acc
                        best_auc = max(metrics.val_auc_scores) if metrics.val_auc_scores else 0.5

                    console.print(f"[bold green]Trial {trial.number} complete:[/bold green] Best Top-5 MLM={best_top5:.4f}, Best AUC={best_auc:.4f}")
                    if use_wandb:
                        import wandb
                        wandb.log({"trial_top5_acc": best_top5, "trial_val_auc": best_auc})
                    return best_top5, best_auc

                except Exception as e:
                    console.print(f"[bold red]Stage 3 (Transformer) failed:[/bold red] {e}")
                    raise optuna.exceptions.TrialPruned()
        finally:
            if use_wandb:
                try:
                    import wandb
                    wandb.finish()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    return objective


def main() -> None:
    args = parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Data directory not found: {data_dir}")
        sys.exit(1)

    comfort_path = Path(args.comfort_path)
    if not comfort_path.exists():
        console.print(f"[bold red]Error:[/bold red] Comfort map not found: {comfort_path}")
        sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(f"[bold blue]Using device:[/bold blue] {device}")

    # --- Pre-load datasets ONCE to optimize memory/speed ---
    console.print("[bold blue]Pre-loading dataset batches into memory...[/bold blue]")
    x_drafts, y_labels, radiant_players, dire_players, patch_ids = load_data(args.data_dir)
    x_pubs, y_pubs, patch_ids_pubs = load_pub_data(args.pub_data_dir)

    max_hero_idx = 127
    for draft in x_drafts:
        max_hero_idx = max(max_hero_idx, int(draft[:, 2].max().item()))
    for pub in x_pubs:
        max_hero_idx = max(max_hero_idx, int(pub[:, 2].max().item()))
    console.print(f"[bold green]Max hero index detected in dataset: {max_hero_idx}[/bold green]")

    player_comfort_map = torch.load(args.comfort_path, weights_only=True)
    if len(player_comfort_map) > 0:
        first_tensor = next(iter(player_comfort_map.values()))
        player_input_dim = first_tensor.size(0)
    else:
        player_input_dim = max_hero_idx * 2

    # --- Setup Optuna Multi-Objective Study ---
    console.print(f"[bold blue]Initializing multi-objective study '{args.study_name}'...[/bold blue]")
    
    # Maximize both Top-5 MLM accuracy and ROC-AUC
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )

    objective_fn = make_objective(
        args=args,
        x_drafts=x_drafts,
        y_labels=y_labels,
        radiant_players=radiant_players,
        dire_players=dire_players,
        patch_ids=patch_ids,
        player_comfort_map=player_comfort_map,
        max_hero_idx=max_hero_idx,
        player_input_dim=player_input_dim,
        device=device,
        x_pubs=x_pubs,
        y_pubs=y_pubs,
        patch_ids_pubs=patch_ids_pubs,
    )

    try:
        study.optimize(objective_fn, n_trials=args.n_trials)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Tuning study interrupted by user. Reviewing current results...[/bold yellow]")

    # --- Analyze the Pareto Front ---
    console.print("\n[bold green]=== Pareto Front (Best Trade-off Trials) ===[/bold green]")
    best_trials = study.best_trials
    for trial in best_trials:
        console.print(f"\n[bold]Trial {trial.number}:[/bold]")
        console.print(f"  Values (Top-5 MLM Acc, ROC-AUC): {trial.values}")
        console.print("  Best Parameters:")
        for k, v in trial.params.items():
            console.print(f"    {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1)
