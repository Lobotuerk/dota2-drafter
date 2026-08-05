import time
import torch
import logging
from dota2drafter.models.match_network import MatchNetwork
from dota2drafter.search.mcts_agent import Dota2DraftAgent
from dota2drafter.processor.hero_indexer import HeroIndexer

# Disable logging
logging.getLogger().setLevel(logging.ERROR)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create real model (untrained is fine for perf test)
    d_model = 64
    num_heroes = 124
    h_gnn = torch.randn(num_heroes + 1, d_model)
    model = MatchNetwork(
        d_model=d_model,
        num_heroes=num_heroes,
        player_input_dim=num_heroes,
        h_gnn=h_gnn
    ).to(device)
    model.eval()

    indexer = HeroIndexer()
    heroes = [{"id": i, "playable": True} for i in range(1, num_heroes+1)]
    indexer.build_mapping(heroes)

    comfort = torch.zeros(10, num_heroes).to(device)

    agent = Dota2DraftAgent(
        model=model,
        comfort_matrix=comfort,
        active_team=0,
        max_iterations=100,
        max_seconds=10.0,
        hero_indexer=indexer
    )

    print("Running 100 iterations...")
    start = time.time()
    agent.search()
    elapsed = time.time() - start
    iters = agent.agent.tree.root.visit_count if agent.agent.tree and agent.agent.tree.root else 0

    print(f'Real model test: {iters} iterations in {elapsed:.3f}s -> {iters/max(elapsed, 0.001):.1f} iters/sec')

if __name__ == '__main__':
    run()
