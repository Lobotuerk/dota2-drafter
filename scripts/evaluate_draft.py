#!/usr/bin/env python3
"""Evaluate draft win probability using the trained Hierarchical Transformer model.

Receives 10 player IDs (5 Radiant + 5 Dire) and 10 hero picks (5 Radiant + 5 Dire),
loads the model and comfort matrices (copying the exact setup from interactive_draft.py),
and computes the predicted win probability for each side.

Usage:
    python scripts/evaluate_draft.py \
        --radiant_heroes "Anti-Mage,Crystal Maiden,Pudge,Tiny,Lina" \
        --dire_heroes "Axe,Invoker,Puck,Shadow Fiend,Slark" \
        --radiant_players "12345,0,0,0,0" \
        --dire_players "0,0,0,0,0"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dota2drafter.config import load_config
from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.processor.hero_indexer import HeroIndexer
from dota2drafter.search.state import DRAFT_SCHEDULE

console = Console()
logger = logging.getLogger(__name__)


def parse_args(config=None, args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate draft win probability using the trained Hierarchical Transformer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches and hero_indexer.json (default: data)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory for model checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit path to best_model.pt (default: <checkpoint_dir>/best_model.pt)",
    )
    parser.add_argument(
        "--rgcn_path",
        type=str,
        default="models/rgcn.pt",
        help="Path to RGCN weights (default: models/rgcn.pt)",
    )
    parser.add_argument(
        "--frozen_embeddings_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to frozen skip-gram/DGI embeddings (default: models/skip_gram_dgi.pt)",
    )
    parser.add_argument(
        "--comfort_path",
        type=str,
        default="data/player_comfort.pt",
        help="Path to player_comfort.pt (default: data/player_comfort.pt)",
    )
    parser.add_argument(
        "--hero_mapping",
        type=str,
        default="hero_mapping.json",
        help="Path to hero name mapping JSON (default: hero_mapping.json)",
    )
    parser.add_argument(
        "--radiant_players",
        type=str,
        default=None,
        help="Comma-separated list of 5 Radiant player account IDs (e.g. '12345,0,0,0,0')",
    )
    parser.add_argument(
        "--dire_players",
        type=str,
        default=None,
        help="Comma-separated list of 5 Dire player account IDs (e.g. '0,0,0,0,0')",
    )
    parser.add_argument(
        "--radiant_heroes",
        type=str,
        default=None,
        help="Comma-separated list of 5 Radiant hero names or IDs",
    )
    parser.add_argument(
        "--dire_heroes",
        type=str,
        default=None,
        help="Comma-separated list of 5 Dire hero names or IDs",
    )
    parser.add_argument(
        "--patch_id",
        type=int,
        default=22,
        help="Patch index for FiLM conditioning (default: 22)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device: "cpu" or "cuda" (auto-detect if None)',
    )

    d_model_default = config.model.d_model if config else 64
    nhead_default = config.model.nhead if config else 4
    num_layers_default = config.model.num_layers_transformer if config else 2
    dim_feedforward_default = config.model.dim_feedforward if config else 128
    dropout_default = config.model.dropout if config else 0.1
    wilson_threshold_default = config.graph.wilson_threshold if config else 0.50
    gamma_default = config.graph.gamma if config else 0.80

    parser.add_argument(
        "--d_model",
        type=int,
        default=d_model_default,
        help=f"Transformer d_model (default: {d_model_default})",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=nhead_default,
        help=f"Number of attention heads (default: {nhead_default})",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=num_layers_default,
        help=f"Number of transformer layers (default: {num_layers_default})",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=dim_feedforward_default,
        help=f"Feedforward dimension (default: {dim_feedforward_default})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=dropout_default,
        help=f"Dropout rate (default: {dropout_default})",
    )
    parser.add_argument(
        "--num_heroes",
        type=int,
        default=127,
        help="Number of heroes (default: 127)",
    )
    parser.add_argument(
        "--wilson_threshold",
        type=float,
        default=wilson_threshold_default,
        help=f"Wilson Score threshold for pruning edges (default: {wilson_threshold_default})",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=gamma_default,
        help=f"Decay factor per major patch (default: {gamma_default})",
    )
    return parser.parse_args(args)


def load_hero_indexer(data_dir: str) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json."""
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if not indexer_path.exists():
        raise FileNotFoundError(f"Hero indexer not found at: {indexer_path}")
    with open(indexer_path) as f:
        hero_data = json.load(f)
    heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
    indexer = HeroIndexer()
    indexer.build_mapping(heroes)
    return indexer


def load_hero_names(path: str, indexer: HeroIndexer) -> dict[int, str]:
    """Load hero name mapping from JSON file and map to contiguous hero indices."""
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Hero mapping file not found at: {mapping_path}")

    with open(mapping_path) as f:
        raw = json.load(f)

    hero_names: dict[int, str] = {}
    for api_id_str, name in raw.items():
        api_id = int(api_id_str)
        idx = indexer.map_hero_id(api_id)
        if idx is not None:
            hero_names[idx] = name

    return hero_names


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
        from dota2drafter.embeddings.data_extractor import DataExtractor
        from dota2drafter.embeddings.rgcn import HeroRGCN

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
        return h_gnn
    elif isinstance(h_gnn_loaded, dict) and "embedding.weight" in h_gnn_loaded:
        return h_gnn_loaded["embedding.weight"]
    return h_gnn_loaded


def build_comfort_tensor(
    comfort_map: dict[int, torch.Tensor],
    player_account_ids: list[int],
    player_input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a (1, 10, C) comfort tensor from player account IDs."""
    comfort_rows = []
    vocab_size = player_input_dim // 2
    default_comfort = torch.zeros(player_input_dim, device=device)
    default_comfort[vocab_size:] = 0.5  # default neutral Wilson score

    for account_id in player_account_ids:
        if account_id in comfort_map:
            t = comfort_map[account_id].to(device)
            if t.size(0) == player_input_dim:
                comfort_rows.append(t)
            elif t.size(0) < player_input_dim:
                padded = torch.zeros(player_input_dim, device=device)
                old_vocab = t.size(0) // 2
                padded[:old_vocab] = t[:old_vocab]
                padded[vocab_size : vocab_size + old_vocab] = t[old_vocab:]
                padded[vocab_size + old_vocab :] = 0.5
                comfort_rows.append(padded)
            else:
                comfort_rows.append(t[:player_input_dim])
        else:
            comfort_rows.append(default_comfort)

    return torch.stack(comfort_rows).unsqueeze(0)


def resolve_hero(
    hero_input: str | int,
    hero_names: dict[int, str],
    name_to_idx: dict[str, int],
    indexer: HeroIndexer,
) -> tuple[int, str]:
    """Resolve a hero string, API ID, or contiguous ID to (contiguous_idx, hero_name)."""
    input_str = str(hero_input).strip()
    if input_str.isdigit():
        val = int(input_str)
        # Check if direct contiguous index
        if val in hero_names:
            return val, hero_names[val]
        # Otherwise treat as API hero ID
        mapped = indexer.map_hero_id(val)
        if mapped is not None and mapped in hero_names:
            return mapped, hero_names[mapped]

    # Name matching
    cleaned = input_str.lower()
    if cleaned in name_to_idx:
        idx = name_to_idx[cleaned]
        return idx, hero_names[idx]

    # Partial match fallback
    matches = [
        (idx, name) for idx, name in hero_names.items()
        if cleaned in name.lower()
    ]
    if len(matches) == 1:
        return matches[0][0], matches[0][1]
    elif len(matches) > 1:
        match_names = ", ".join(name for _, name in matches)
        raise ValueError(f"Ambiguous hero '{input_str}'. Candidates: {match_names}")

    raise ValueError(f"Hero '{input_str}' could not be resolved.")


class PatchWrappedModel(torch.nn.Module):
    """Wraps MatchNetwork to automatically inject patch_tensor conditioning."""

    def __init__(self, model: MatchNetwork, patch_tensor: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.patch_tensor = patch_tensor

    def predict_proba(self, x_draft: torch.Tensor, player_comfort: torch.Tensor) -> torch.Tensor:
        expanded_patch = self.patch_tensor.expand(x_draft.size(0))
        return self.model.predict_proba(x_draft, player_comfort, patch_ids=expanded_patch)

    def forward(
        self, x_draft: torch.Tensor, player_comfort: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded_patch = self.patch_tensor.expand(x_draft.size(0))
        return self.model(x_draft, player_comfort, patch_ids=expanded_patch, *args, **kwargs)


def evaluate_draft(
    radiant_heroes: list[str | int],
    dire_heroes: list[str | int],
    radiant_players: list[int] | None = None,
    dire_players: list[int] | None = None,
    data_dir: str = "data",
    checkpoint_path: str = "checkpoints/best_model.pt",
    rgcn_path: str = "models/rgcn.pt",
    frozen_embeddings_path: str = "models/skip_gram_dgi.pt",
    comfort_path: str = "data/player_comfort.pt",
    hero_mapping_path: str = "hero_mapping.json",
    patch_id: int = 22,
    device: str | None = None,
) -> tuple[float, float]:
    """Evaluate win probability for a complete 5v5 draft.

    Returns:
        tuple of (radiant_win_probability, dire_win_probability)
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    hero_indexer = load_hero_indexer(data_dir)
    hero_names = load_hero_names(hero_mapping_path, hero_indexer)
    name_to_idx = {name.lower().strip(): idx for idx, name in hero_names.items()}

    # Resolve heroes
    r_resolved = [resolve_hero(h, hero_names, name_to_idx, hero_indexer) for h in radiant_heroes]
    d_resolved = [resolve_hero(h, hero_names, name_to_idx, hero_indexer) for h in dire_heroes]

    if len(r_resolved) != 5 or len(d_resolved) != 5:
        raise ValueError("Exactly 5 Radiant heroes and 5 Dire heroes are required.")

    # Players
    r_players = list(radiant_players or [0] * 5)
    d_players = list(dire_players or [0] * 5)
    while len(r_players) < 5:
        r_players.append(0)
    while len(d_players) < 5:
        d_players.append(0)
    all_players = r_players[:5] + d_players[:5]

    # Load comfort map
    c_path = Path(comfort_path)
    comfort_map = torch.load(c_path, weights_only=True) if c_path.exists() else {}
    if len(comfort_map) > 0:
        player_input_dim = next(iter(comfort_map.values())).size(0)
    else:
        player_input_dim = hero_indexer.get_contiguous_count() * 2

    comfort_tensor = build_comfort_tensor(comfort_map, all_players, player_input_dim, dev)

    # Load checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, weights_only=True, map_location=dev)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if (
        "match_network.joint_embedding.project.weight" in state_dict
        and "match_network.joint_embedding.project_policy.weight" not in state_dict
    ):
        state_dict["match_network.joint_embedding.project_policy.weight"] = state_dict[
            "match_network.joint_embedding.project.weight"
        ]
        state_dict["match_network.joint_embedding.project_policy.bias"] = state_dict[
            "match_network.joint_embedding.project.bias"
        ]
        state_dict["match_network.joint_embedding.project_value.weight"] = state_dict[
            "match_network.joint_embedding.project.weight"
        ].clone()
        state_dict["match_network.joint_embedding.project_value.bias"] = state_dict[
            "match_network.joint_embedding.project.bias"
        ].clone()
        del (
            state_dict["match_network.joint_embedding.project.weight"],
            state_dict["match_network.joint_embedding.project.bias"],
        )
    if "match_network.mlm_head.joint_embedding.project.weight" in state_dict:
        del (
            state_dict["match_network.mlm_head.joint_embedding.project.weight"],
            state_dict["match_network.mlm_head.joint_embedding.project.bias"],
        )

    # Infer architecture dimensions dynamically from checkpoint
    if "match_network.h_gnn" in state_dict:
        h_gnn = state_dict["match_network.h_gnn"]
        d_model = h_gnn.shape[1]
        num_heroes = h_gnn.shape[0] - 1
    else:
        h_gnn = load_h_gnn(
            Path(rgcn_path),
            Path(frozen_embeddings_path),
            hero_indexer.get_contiguous_count(),
            64,
            Path(data_dir),
            device=dev,
        )
        d_model = h_gnn.shape[1]
        num_heroes = h_gnn.shape[0] - 1

    if "player_network.mlp.0.weight" in state_dict:
        player_input_dim = state_dict["player_network.mlp.0.weight"].shape[1]

    dim_ff = 128
    if "match_network.transformer_decoder.layers.0.linear1.weight" in state_dict:
        dim_ff = state_dict["match_network.transformer_decoder.layers.0.linear1.weight"].shape[0]

    num_layers = 2
    layer_keys = [
        k for k in state_dict if k.startswith("match_network.transformer_decoder.layers.")
    ]
    if layer_keys:
        num_layers = max([int(k.split(".")[3]) for k in layer_keys]) + 1

    model = MatchNetwork(
        d_model=d_model,
        num_heroes=num_heroes,
        player_input_dim=player_input_dim,
        dim_feedforward=dim_ff,
        num_layers=num_layers,
        h_gnn=h_gnn,
    ).to(dev)

    model.load_state_dict(state_dict)
    model.eval()

    # Wrap model with patch conditioning
    patch_tensor = torch.tensor([patch_id], dtype=torch.long, device=dev)
    wrapped_model = PatchWrappedModel(model, patch_tensor)

    # Build draft tensor with picks placed according to standard Captains Mode schedule
    x_draft = torch.zeros(1, 24, 4, device=dev)
    r_iter = iter([idx for idx, _ in r_resolved])
    d_iter = iter([idx for idx, _ in d_resolved])

    for step, (action, team) in enumerate(DRAFT_SCHEDULE):
        if action == "pick":
            if team == 0:
                h_idx = next(r_iter)
                x_draft[0, step] = torch.tensor([1.0, 0.0, float(h_idx), float(step)], device=dev)
            else:
                h_idx = next(d_iter)
                x_draft[0, step] = torch.tensor([1.0, 1.0, float(h_idx), float(step)], device=dev)
        else:
            x_draft[0, step] = torch.tensor([0.0, float(team), -1.0, float(step)], device=dev)

    with torch.no_grad():
        radiant_win_prob = wrapped_model.predict_proba(x_draft, comfort_tensor).item()
        dire_win_prob = 1.0 - radiant_win_prob

    return radiant_win_prob, dire_win_prob


def main(args_list: list[str] | None = None) -> None:
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
            logger.warning("Could not load config from %s: %s", config_path, e)

    args = parse_args(config, args_list)

    # Resolve checkpoint path
    checkpoint_file = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(args.checkpoint_dir) / "best_model.pt"
    )

    # Collect inputs (from args or prompt interactively)
    if args.radiant_heroes:
        r_heroes_raw = [h.strip() for h in args.radiant_heroes.split(",") if h.strip()]
    else:
        console.print("[bold cyan]Enter 5 Radiant heroes (comma-separated):[/bold cyan]")
        r_input = console.input("Radiant Heroes: ")
        r_heroes_raw = [h.strip() for h in r_input.split(",") if h.strip()]

    if args.dire_heroes:
        d_heroes_raw = [h.strip() for h in args.dire_heroes.split(",") if h.strip()]
    else:
        console.print("[bold cyan]Enter 5 Dire heroes (comma-separated):[/bold cyan]")
        d_input = console.input("Dire Heroes: ")
        d_heroes_raw = [h.strip() for h in d_input.split(",") if h.strip()]

    r_players = (
        [int(p.strip()) for p in args.radiant_players.split(",") if p.strip()]
        if args.radiant_players
        else [0] * 5
    )
    d_players = (
        [int(p.strip()) for p in args.dire_players.split(",") if p.strip()]
        if args.dire_players
        else [0] * 5
    )

    console.print(f"[bold blue]Evaluating draft using checkpoint: {checkpoint_file}...[/bold blue]")

    radiant_prob, dire_prob = evaluate_draft(
        radiant_heroes=r_heroes_raw,
        dire_heroes=d_heroes_raw,
        radiant_players=r_players,
        dire_players=d_players,
        data_dir=args.data_dir,
        checkpoint_path=str(checkpoint_file),
        rgcn_path=args.rgcn_path,
        frozen_embeddings_path=args.frozen_embeddings_path,
        comfort_path=args.comfort_path,
        hero_mapping_path=args.hero_mapping,
        patch_id=args.patch_id,
        device=args.device,
    )

    # Render summary table
    table = Table(title="Draft Lineups", show_header=True, header_style="bold magenta")
    table.add_column("Slot", style="dim", width=6)
    table.add_column("Radiant Hero", style="bold green", width=25)
    table.add_column("Radiant Player", justify="right", width=16)
    table.add_column("Dire Hero", style="bold red", width=25)
    table.add_column("Dire Player", justify="right", width=16)

    for i in range(5):
        table.add_row(
            f"Pos {i + 1}",
            str(r_heroes_raw[i]).title() if i < len(r_heroes_raw) else "—",
            str(r_players[i]) if i < len(r_players) and r_players[i] != 0 else "Anonymous",
            str(d_heroes_raw[i]).title() if i < len(d_heroes_raw) else "—",
            str(d_players[i]) if i < len(d_players) and d_players[i] != 0 else "Anonymous",
        )

    console.print()
    console.print(table)
    console.print()

    # Probability summary panel
    r_pct = radiant_prob * 100.0
    d_pct = dire_prob * 100.0

    favored = "Radiant" if radiant_prob >= dire_prob else "Dire"
    color = "green" if radiant_prob >= dire_prob else "red"

    panel_content = (
        f"[bold green]Radiant Win Probability:[/bold green] [bold]{r_pct:.2f}%[/bold]\n"
        f"[bold red]Dire Win Probability:   [/bold red] [bold]{d_pct:.2f}%[/bold]\n\n"
        f"Model Favors: [bold {color}]{favored}[/bold {color}] "
        f"(+{abs(r_pct - d_pct):.2f}% advantage)"
    )
    panel = Panel(
        panel_content,
        title="[bold]Win Probability Prediction[/bold]",
        border_style=color,
    )
    console.print(panel)


if __name__ == "__main__":
    main()
