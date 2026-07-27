import pytest
import torch
from dota2drafter.processor.hero_indexer import HeroIndexer
from dota2drafter.processor.draft_validator import DraftValidator
from dota2drafter.processor.tensor_transformer import TensorTransformer


def test_tensor_transformer_stratz():
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 30)])
    
    validator = DraftValidator()
    transformer = TensorTransformer(indexer, validator)
    
    # Mock valid STRATZ match details
    picks_bans = []
    for i in range(24):
        picks_bans.append({
            "type": "pick" if i % 2 == 0 else "ban",
            "hero": {"id": i + 1},
            "team": 0 if i % 4 < 2 else 1,
            "order": i
        })
        
    match_data = {
        "id": "7512345678",
        "radiantWin": True,
        "draft": {
            "picksBans": picks_bans
        }
    }
    
    processed = transformer.transform(match_data, source="stratz")
    assert processed is not None
    assert processed.match_id == "7512345678"
    assert isinstance(processed.x_tensor, torch.Tensor)
    assert isinstance(processed.y_tensor, torch.Tensor)
    
    # x shape should be (24, 3)
    assert processed.x_tensor.shape == (24, 3)
    # y shape should be (1,)
    assert processed.y_tensor.shape == (1,)
    assert processed.y_tensor.item() == 1.0
    
    # Check values at first step (is_pick=1.0, team=0.0, hero_id mapped correctly)
    assert processed.x_tensor[0].tolist() == [1.0, 0.0, float(indexer.map_hero_id(1))]


def test_tensor_transformer_opendota():
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 30)])
    
    validator = DraftValidator()
    transformer = TensorTransformer(indexer, validator)
    
    # Mock valid OpenDota match details
    picks_bans = []
    for i in range(24):
        picks_bans.append({
            "is_pick": i % 2 == 0,
            "hero_id": i + 1,
            "team": 0 if i % 4 < 2 else 1,
            "order": i
        })
        
    match_data = {
        "match_id": 7512345678,
        "radiant_win": False,
        "game_mode": 2,
        "picks_bans": picks_bans
    }
    
    processed = transformer.transform(match_data, source="opendota")
    assert processed is not None
    assert processed.match_id == "7512345678"
    assert processed.x_tensor.shape == (24, 3)
    assert processed.y_tensor.shape == (1,)
    assert processed.y_tensor.item() == 0.0
