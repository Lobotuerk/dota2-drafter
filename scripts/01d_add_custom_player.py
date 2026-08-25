#!/usr/bin/env python3
"""Add a custom player and their manually specified hero stats to the comfort map.

Usage:
    python scripts/01d_add_custom_player.py
"""

import torch
import json
import math
from pathlib import Path
from dota2drafter.config import load_config
from rich.console import Console
from rich.prompt import Prompt, IntPrompt

console = Console()

def wilson_score(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.5
    p = wins / n
    denominator = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2))
    return (center - spread) / denominator

def main():
    comfort_path = Path("data/player_comfort.pt")
    if not comfort_path.exists():
        console.print("[red]player_comfort.pt not found![/red]")
        return
        
    comfort_map = torch.load(comfort_path, weights_only=True)
    
    # Load hero mapping
    with open("data/hero_indexer.json", "r") as f:
        hero_data = json.load(f)
        
    vocab_size = max([int(v) for v in hero_data.keys()])
    name_to_idx = {name.lower(): int(idx) - 1 for idx, name in hero_data.items()} # 0-based
    
    account_id = IntPrompt.ask("Enter a custom Account ID (e.g. 999999999)")
    
    # Start fresh or edit existing
    if account_id in comfort_map:
        console.print(f"[yellow]Player {account_id} already exists! Overwriting...[/yellow]")
        
    vector = torch.zeros(vocab_size * 2, dtype=torch.float32)
    vector[vocab_size:] = 0.5 # Default wilson
    
    total_games = IntPrompt.ask("Enter the total estimated matches played by this player (for affinity scaling)")
    
    console.print("\n[bold]Enter hero stats (type 'done' as Hero Name to finish)[/bold]")
    while True:
        hero = Prompt.ask("Hero Name").lower()
        if hero == 'done':
            break
            
        if hero not in name_to_idx:
            # Try partial match
            matches = [name for name in name_to_idx.keys() if hero in name]
            if not matches:
                console.print(f"[red]Could not find hero: {hero}[/red]")
                continue
            elif len(matches) > 1:
                console.print(f"[yellow]Multiple matches found: {matches}[/yellow]")
                continue
            hero = matches[0]
            
        hero_idx = name_to_idx[hero]
        games = IntPrompt.ask(f"Games played on {hero.title()}")
        wins = IntPrompt.ask(f"Wins on {hero.title()}")
        
        affinity = games / max(total_games, 1)
        w_score = wilson_score(wins, games)
        
        vector[hero_idx] = affinity
        vector[vocab_size + hero_idx] = w_score
        
        console.print(f"[green]Added {hero.title()} (Affinity: {affinity:.2f}, Wilson: {w_score:.2f})[/green]\n")
        
    comfort_map[account_id] = vector
    torch.save(comfort_map, comfort_path)
    console.print(f"\n[bold green]Saved player {account_id} to {comfort_path}[/bold green]")

if __name__ == '__main__':
    main()
