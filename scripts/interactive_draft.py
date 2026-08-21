#!/usr/bin/env python3
"""Interactive draft support tool with MCTS-based recommendations.

Provides a terminal UI for live draft decision support. The user selects
their team, provides a comfort matrix (or uses defaults), and then
enters a 24-step draft loop where the MCTS engine recommends picks/bans.

Usage::

    python scripts/interactive_draft.py \\
        --checkpoint_dir checkpoints \\
        --rgcn_path models/rgcn.pt \\
        --comfort_path data/player_comfort.pt \\
        --data_dir data \\
        --d_model 64 \\
        --num_heroes 124 \\
        --max_iterations 1000 \\
        --max_seconds 30

Prerequisites: run the training pipeline first to produce a model
checkpoint, RGCN embeddings, and player comfort map.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.processor.hero_indexer import HeroIndexer
from dota2drafter.search.mcts_agent import Dota2DraftAgent
from dota2drafter.search.state import DRAFT_SCHEDULE, DraftMove

logger = logging.getLogger(__name__)
console = Console()
torch.set_float32_matmul_precision('high')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive MCTS draft decision support tool.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory with .pt match batches and hero_indexer.json (default: data)",
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
    parser.add_argument(
        "--frozen_embeddings_path",
        type=str,
        default="models/skip_gram_dgi.pt",
        help="Path to frozen skip-gram/DGI embeddings (default: models/skip_gram_dgi.pt)",
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
        "--num_heroes", type=int, default=124, help="Number of heroes (default: 124)"
    )
    parser.add_argument(
        "--max_iterations", type=int, default=1000, help="Max MCTS iterations (default: 1000)"
    )
    parser.add_argument(
        "--max_seconds", type=float, default=30.0, help="Max MCTS search time (default: 30)"
    )
    parser.add_argument(
        "--max_candidates", type=int, default=20, help="Max candidate moves to evaluate per MCTS node (default: 20)"
    )
    parser.add_argument(
        "--top_n", type=int, default=5, help="Number of recommendations to show (default: 5)"
    )
    parser.add_argument(
        "--c_puct", type=float, default=1.414,
        help="PUCT exploration constant (default: 1.414)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="MCTS batch size for batched leaf evaluation (default: 64)",
    )
    parser.add_argument(
        "--num_search_threads", type=int, default=4,
        help="MCTS parallel search threads (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help='Device: "cpu" or "cuda" (auto-detect if None)'
    )
    parser.add_argument(
        "--hero_mapping",
        type=str,
        default="hero_mapping.json",
        help="Path to hero name mapping JSON (default: hero_mapping.json)",
    )
    return parser.parse_args()


def load_hero_indexer(data_dir: str) -> HeroIndexer:
    """Load HeroIndexer from data/hero_indexer.json if available."""
    indexer = HeroIndexer()
    indexer_path = Path(data_dir) / "hero_indexer.json"
    if indexer_path.exists():
        import json
        with open(indexer_path) as f:
            hero_data = json.load(f)
        heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
        indexer.build_mapping(heroes)
    return indexer


def load_hero_names(path: str, indexer: HeroIndexer) -> dict[int, str]:
    """Load hero name mapping from JSON file and map to contiguous hero indices.

    Args:
        path: Path to hero_mapping.json.
        indexer: The HeroIndexer mapping API ID to contiguous index.

    Returns:
        Dict mapping contiguous hero index to hero name.
    """
    import json

    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}

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
        hero_graph = extractor.build_hero_graph(batches)

        h_gnn = rgcn_model.get_embeddings(hero_graph)
        console.print(f"[bold green]Extracted raw H_GNN embeddings of shape {tuple(h_gnn.shape)} from loaded model state_dict.[/bold green]")
        return h_gnn
    return h_gnn_loaded


def build_comfort_tensor(
    comfort_map: dict[int, torch.Tensor],
    player_account_ids: list[int],
    player_input_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a (10, C) comfort tensor from player account IDs.

    Args:
        comfort_map: Dict mapping account_id to comfort vector.
        player_account_ids: List of 10 account IDs (5 Radiant + 5 Dire).
        player_input_dim: Dimension of comfort vectors.
        device: PyTorch device.

    Returns:
        Tensor of shape (10, player_input_dim).
    """
    rows: list[torch.Tensor] = []
    for account_id in player_account_ids:
        if account_id in comfort_map:
            rows.append(comfort_map[account_id])
        else:
            rows.append(torch.zeros(player_input_dim))
    return torch.stack(rows).to(device)


def display_recommendations(
    recommendations: list[tuple],
    step: int,
    hero_names: dict[int, str],
    top_n: int,
) -> None:
    """Display MCTS recommendations as a rich table.

    Args:
        recommendations: List of (move, visit_count, win_prob) tuples.
        step: Current draft step.
        hero_names: Mapping from hero index to name.
        top_n: Number of recommendations to show.
    """
    schedule_action, schedule_team = DRAFT_SCHEDULE[step]
    team_name = "Radiant" if schedule_team == 0 else "Dire"
    action_word = "pick" if schedule_action == "pick" else "ban"

    console.print()
    console.print(Panel(
        f"[bold]Step {step + 1}: {team_name} {action_word}[/bold]",
        title="[bold blue]MCTS Recommendations[/bold blue]",
        border_style="blue",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Hero", style="green", width=20)
    table.add_column("Action", width=8)
    table.add_column("Win Prob", style="yellow", width=10)
    table.add_column("Prior", style="yellow", width=10)
    table.add_column("Visits", style="cyan", width=8)

    for rank, rec in enumerate(recommendations[:top_n], 1):
        move = rec.move
        visit_count = rec.visit_count
        win_prob = rec.win_probability
        prior_prob = rec.prior_probability
        hero_name = hero_names.get(move.hero_id, f"Hero {move.hero_id}")
        action_word = "Pick" if move.is_pick else "Ban"
        table.add_row(
            str(rank),
            hero_name,
            action_word,
            f"{win_prob:.3f}",
            f"{prior_prob:.3f}",
            str(visit_count),
        )

    console.print(table)


def display_principal_variation(variation: list[DraftMove], hero_names: dict[int, str]) -> None:
    """Display the principal variation (expected draft plan).

    Args:
        variation: List of DraftMove objects.
        hero_names: Mapping from hero index to name.
    """
    if not variation:
        return

    console.print()
    console.print(Panel(
        "[bold]Expected Draft Plan (Principal Variation)[/bold]",
        title="[bold green]MCTS Principal Variation[/bold green]",
        border_style="green",
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", width=6)
    table.add_column("Team", width=10)
    table.add_column("Action", width=8)
    table.add_column("Hero", style="green", width=20)

    for move in variation:
        team_name = "Radiant" if move.team == 0 else "Dire"
        action_word = "Pick" if move.is_pick else "Ban"
        hero_name = hero_names.get(move.hero_id, f"Hero {move.hero_id}")
        table.add_row(
            str(move.step_index + 1),
            team_name,
            action_word,
            hero_name,
        )

    console.print(table)


def get_user_move(step: int, hero_names: dict[int, str]) -> DraftMove | None:
    """Prompt the user to input their draft action by entering a hero name.

    Args:
        step: Current draft step.
        hero_names: Mapping from contiguous hero index to name.

    Returns:
        DraftMove for the user's choice, or None to skip.
    """
    schedule_action, schedule_team = DRAFT_SCHEDULE[step]
    team_name = "Radiant" if schedule_team == 0 else "Dire"
    action_word = "pick" if schedule_action == "pick" else "ban"

    # Invert hero_names for case-insensitive lookup
    name_to_idx = {name.lower().strip(): idx for idx, name in hero_names.items()}

    while True:
        console.print()
        hero_input = console.input(
            f"[bold]Step {step + 1} - {team_name} {action_word}[/bold] "
            f"Enter hero name: "
        ).strip()

        if hero_input.lower() in ("q", "quit", "exit"):
            console.print("[bold yellow]Exiting draft...[/bold yellow]")
            sys.exit(0)

        if not hero_input:
            console.print("[bold red]Input cannot be empty. Please enter a hero name.[/bold red]")
            continue

        if hero_input.lower() == "skip":
            console.print("[bold red]Draft moves cannot be skipped in Captains Mode. Please enter a hero name.[/bold red]")
            continue

        # Try exact case-insensitive match
        match_idx = name_to_idx.get(hero_input.lower())

        # If not exact match, try partial match (e.g. "anti" matches "Anti-Mage")
        if match_idx is None:
            matches = [
                (idx, name) for idx, name in hero_names.items()
                if hero_input.lower() in name.lower()
            ]
            if len(matches) == 1:
                match_idx = matches[0][0]
                console.print(f"[dim]Auto-resolved to: {matches[0][1]}[/dim]")
            elif len(matches) > 1:
                console.print(f"[bold yellow]Multiple matches found:[/bold yellow] {', '.join(name for _, name in matches)}")
                console.print("Please be more specific.")
                continue

        if match_idx is not None:
            return DraftMove(
                hero_id=match_idx,
                is_pick=(schedule_action == "pick"),
                team=schedule_team,
                step_index=step,
            )

        console.print(f"[bold red]Hero '{hero_input}' not found. Please try again.[/bold red]")


def main() -> None:
    """Run the interactive draft tool."""
    args = parse_args()

    # Setup device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(f"[bold blue]Using device: {device}[/bold blue]")

    # Load checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_model.pt"
    if not best_checkpoint.exists():
        console.print(f"[bold red]Error:[/bold red] Best checkpoint not found: {best_checkpoint}")
        console.print("[bold yellow]Hint:[/bold yellow] Run training first with --mode train")
        sys.exit(1)

    # Load RGCN embeddings
    rgcn_path = Path(args.rgcn_path)
    if not rgcn_path.exists():
        console.print(f"[bold red]Error:[/bold red] RGCN model not found: {rgcn_path}")
        sys.exit(1)

    comfort_path = Path(args.comfort_path)
    if not comfort_path.exists():
        console.print(f"[bold red]Error:[/bold red] Comfort map not found: {comfort_path}")
        sys.exit(1)

    console.print("[bold blue]Loading RGCN embeddings...[/bold blue]")
    h_gnn = load_h_gnn(
        rgcn_path,
        Path(args.frozen_embeddings_path),
        args.num_heroes,
        args.d_model,
        Path(args.data_dir),
    )

    console.print("[bold blue]Loading comfort map...[/bold blue]")
    player_comfort_map = torch.load(args.comfort_path, weights_only=True)

    console.print("[bold blue]Loading hero indexer...[/bold blue]")
    hero_indexer = load_hero_indexer(args.data_dir)
    player_input_dim = hero_indexer.get_contiguous_count() if hero_indexer.get_contiguous_count() > 0 else args.num_heroes

    # Build model
    console.print("[bold blue]Loading model...[/bold blue]")
    model = MatchNetwork(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_heroes=args.num_heroes,
        player_input_dim=player_input_dim,
        h_gnn=h_gnn,
    ).to(device)

    # Load checkpoint into model
    checkpoint = torch.load(best_checkpoint, weights_only=True, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Load hero names
    hero_names = load_hero_names(args.hero_mapping, hero_indexer)

    # Team selection
    console.print()
    console.print(Panel(
        "[bold]Interactive MCTS Draft Support[/bold]\n\n"
        "This tool provides MCTS-based draft recommendations.\n"
        "You will be prompted at each step to either input your\n"
        "action or let the MCTS engine recommend one.\n",
        title="[bold cyan]Dota 2 Draft Assistant[/bold cyan]",
        border_style="cyan",
    ))

    console.print("Select your team:")
    console.print("  [bold]1[/bold] - Radiant (team 0)")
    console.print("  [bold]2[/bold] - Dire (team 1)")
    team_input = console.input("Enter team number (1 or 2): ").strip()
    active_team = 0 if team_input == "1" else 1
    team_name = "Radiant" if active_team == 0 else "Dire"
    console.print(f"[bold green]Active team: {team_name}[/bold green]")

    # Build comfort tensor from user input
    console.print()
    console.print("[bold]Player Account IDs[/bold]")
    console.print("Enter 10 account IDs (5 Radiant + 5 Dire), comma-separated:")
    console.print("  Format: r1,r2,r3,r4,r5,d1,d2,d3,d4,d5")
    console.print("  (Use 0 for unknown players to use default comfort vectors)")

    account_input = console.input("Account IDs: ").strip()
    account_ids = [int(x.strip()) for x in account_input.split(",") if x.strip()]

    # Pad to 10 if needed
    while len(account_ids) < 10:
        account_ids.append(0)

    comfort_tensor = build_comfort_tensor(
        player_comfort_map,
        account_ids[:10],
        player_input_dim,
        device,
    )

    # Set global pymcts threads to match agent
    import pymcts
    pymcts.set_rollout_threads(args.num_search_threads)
    
    # Initialize the draft agent
    
    # We pass the absolute latest patch ID (index 13 which corresponds to 7.41e)
    # to evaluate all games in the current meta.
    patch_tensor = torch.tensor([21], dtype=torch.long, device=device)
    
    # Wrap model to automatically inject the patch_tensor
    class PatchWrappedModel(torch.nn.Module):
        def __init__(self, m, p):
            super().__init__()
            self.model = m
            self.p = p
        def predict_proba(self, x_draft, player_comfort):
            expanded_patch = self.p.expand(x_draft.size(0))
            return self.model.predict_proba(x_draft, player_comfort, patch_ids=expanded_patch)
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)
            
    wrapped_model = PatchWrappedModel(model, patch_tensor)
    
    agent = Dota2DraftAgent(
        model=wrapped_model,
        comfort_matrix=comfort_tensor,
        active_team=active_team,
        max_iterations=args.max_iterations,
        max_seconds=args.max_seconds,
        c_puct=args.c_puct,
        batch_size=args.batch_size,
        num_search_threads=args.num_search_threads,
        top_n=args.top_n,
        hero_indexer=hero_indexer,
        max_candidates=args.max_candidates,
    )

    # Main draft loop
    console.print()
    console.print(Panel(
        f"[bold]Draft starting![/bold]\n"
        f"Team: {team_name} | "
        f"Iterations: {args.max_iterations} | "
        f"Time limit: {args.max_seconds}s",
        title="[bold yellow]Draft Session[/bold yellow]",
        border_style="yellow",
    ))

    total_steps = 24
    for step in range(total_steps):
        schedule_action, schedule_team = DRAFT_SCHEDULE[step]
        action_word = "pick" if schedule_action == "pick" else "ban"

        # Check whose turn it is
        is_my_turn = (schedule_team == active_team)

        if is_my_turn:
            # MCTS recommends
            console.print(f"\n[bold cyan]Step {step + 1}/{total_steps}[/bold cyan] "
                         f"[bold]{team_name} {action_word}[/bold] - MCTS searching...")

            start_time = time.time()
            recommendations = agent.search()
            elapsed = time.time() - start_time

            if recommendations:
                # Log the actual number of iterations completed
                actual_iterations = agent.agent.tree.root.visit_count if agent.agent.tree and agent.agent.tree.root else 0
                console.print(f"[dim]Search completed in {elapsed:.1f}s ({actual_iterations} iterations)[/dim]")

                display_recommendations(recommendations, step, hero_names, args.top_n)
                display_principal_variation(agent.get_principal_variation(), hero_names)

                # Ask user to confirm or pick from recommendations
                console.print()
                choice = console.input(
                    "[bold]Accept top recommendation? [y/n]: [/bold]"
                ).strip().lower()

                if choice != "y":
                    # Let user override
                    move = get_user_move(step, hero_names)
                    if move is None:
                        move = recommendations[0].move
                else:
                    move = recommendations[0].move

                hero_name = hero_names.get(move.hero_id, f"Hero {move.hero_id}")
                action_past = "Picked" if move.is_pick else "Banned"
                team_str = "Radiant" if move.team == 0 else "Dire"
                console.print(f"[bold green]{team_str} {action_past} {hero_name} (Step {step + 1})[/bold green]")
                agent.update_state(move)
            else:
                console.print("[bold red]No valid moves available.[/bold red]")
        else:
            # Opponent's turn - user inputs
            opponent_name = "Dire" if active_team == 0 else "Radiant"
            console.print(f"\n[bold cyan]Step {step + 1}/{total_steps}[/bold cyan] "
                         f"[bold]{opponent_name} {action_word}[/bold]")

            move = get_user_move(step, hero_names)
            if move is None:
                console.print("[bold yellow]Skipping opponent move.[/bold yellow]")
                continue

            hero_name = hero_names.get(move.hero_id, f"Hero {move.hero_id}")
            action_past = "Picked" if move.is_pick else "Banned"
            team_str = "Radiant" if move.team == 0 else "Dire"
            console.print(f"[bold green]{team_str} {action_past} {hero_name} (Step {step + 1})[/bold green]")
            agent.update_state(move)

    # Final summary
    console.print()
    console.print(Panel(
        "[bold]Draft Complete![/bold]\n\n"
        "The full draft tree has been explored.",
        title="[bold green]Session Complete[/bold green]",
        border_style="green",
    ))

    # Calculate and display final win probability
    if hasattr(agent, "state") and agent.state is not None:
        console.print()
        with console.status("[bold blue]Evaluating final draft win probability...[/bold blue]"):
            active_win_prob = agent.state.rollout()

        if active_team == 0:
            radiant_win_prob = active_win_prob
            dire_win_prob = 1.0 - active_win_prob
        else:
            radiant_win_prob = 1.0 - active_win_prob
            dire_win_prob = active_win_prob

        console.print(Panel(
            f"Active Team ([bold]{team_name}[/bold]) Win Probability: [bold green]{active_win_prob:.2%}[/bold green]\n"
            f"Radiant Win Probability: [bold cyan]{radiant_win_prob:.2%}[/bold cyan]\n"
            f"Dire Win Probability: [bold magenta]{dire_win_prob:.2%}[/bold magenta]",
            title="[bold gold3]Final Draft Win Probability[/bold gold3]",
            border_style="gold3",
        ))

    # Display final draft summary table
    if hasattr(agent, "state") and hasattr(agent.state, "actions") and agent.state.actions:
        console.print()
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Step", style="cyan", width=6)
        table.add_column("Team", width=10)
        table.add_column("Action", width=8)
        table.add_column("Hero", style="green", width=20)

        for move in agent.state.actions:
            team_name = "Radiant" if move.team == 0 else "Dire"
            action_word = "Pick" if move.is_pick else "Ban"
            hero_name = hero_names.get(move.hero_id, f"Hero {move.hero_id}")
            table.add_row(
                str(move.step_index + 1),
                team_name,
                action_word,
                hero_name,
            )

        console.print(Panel(table, title="[bold blue]Final Draft Summary[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    main()
