import torch

from dota2drafter.processor.draft_validator import DraftValidator
from dota2drafter.processor.hero_indexer import HeroIndexer
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

    # x shape should be (24, 4) - new format with step_index
    assert processed.x_tensor.shape == (24, 4)
    # y shape should be (1,)
    assert processed.y_tensor.shape == (1,)
    assert processed.y_tensor.item() == 1.0

    # Check values at first step (is_pick=1.0, team=0.0, hero_id mapped correctly, step_index=0)
    assert processed.x_tensor[0].tolist() == [1.0, 0.0, float(indexer.map_hero_id(1)), 0.0]

    # Check step_index is correctly set
    assert processed.x_tensor[10, 3].item() == 10.0
    assert processed.x_tensor[23, 3].item() == 23.0

    # Player fields should be zero-padded lists when no player data in payload
    assert processed.radiant_players == [0, 0, 0, 0, 0]
    assert processed.dire_players == [0, 0, 0, 0, 0]


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
    assert processed.x_tensor.shape == (24, 4)
    assert processed.y_tensor.shape == (1,)
    assert processed.y_tensor.item() == 0.0


def test_tensor_transformer_stratz_with_players():
    """Test STRATZ player extraction."""
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 30)])

    validator = DraftValidator()
    transformer = TensorTransformer(indexer, validator)

    picks_bans = []
    for i in range(24):
        picks_bans.append({
            "type": "pick" if i % 2 == 0 else "ban",
            "hero": {"id": i + 1},
            "team": 0 if i % 4 < 2 else 1,
            "order": i
        })

    match_data = {
        "id": "7512345679",
        "radiantWin": True,
        "draft": {
            "picksBans": picks_bans
        },
        "players": [
            {"accountid": 1001, "team": 1},  # Radiant
            {"accountid": 1002, "team": 1},
            {"accountid": 1003, "team": 1},
            {"accountid": 1004, "team": 1},
            {"accountid": 1005, "team": 1},
            {"accountid": 2001, "team": 2},  # Dire
            {"accountid": 2002, "team": 2},
            {"accountid": 2003, "team": 2},
            {"accountid": 2004, "team": 2},
            {"accountid": 2005, "team": 2},
        ]
    }

    processed = transformer.transform(match_data, source="stratz")
    assert processed is not None
    assert processed.radiant_players == [1001, 1002, 1003, 1004, 1005]
    assert processed.dire_players == [2001, 2002, 2003, 2004, 2005]


def test_tensor_transformer_opendota_with_players():
    """Test OpenDota player extraction."""
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 30)])

    validator = DraftValidator()
    transformer = TensorTransformer(indexer, validator)

    picks_bans = []
    for i in range(24):
        picks_bans.append({
            "is_pick": i % 2 == 0,
            "hero_id": i + 1,
            "team": 0 if i % 4 < 2 else 1,
        })

    # player_slot < 128 = Radiant, >= 128 = Dire
    match_data = {
        "match_id": 7512345679,
        "radiant_win": True,
        "game_mode": 2,
        "picks_bans": picks_bans,
        "players": [
            {"account_id": 1001, "player_slot": 0},   # Radiant
            {"account_id": 1002, "player_slot": 1},
            {"account_id": 1003, "player_slot": 2},
            {"account_id": 1004, "player_slot": 3},
            {"account_id": 1005, "player_slot": 4},
            {"account_id": 2001, "player_slot": 128},  # Dire
            {"account_id": 2002, "player_slot": 129},
            {"account_id": 2003, "player_slot": 130},
            {"account_id": 2004, "player_slot": 131},
            {"account_id": 2005, "player_slot": 132},
        ]
    }

    processed = transformer.transform(match_data, source="opendota")
    assert processed is not None
    assert processed.radiant_players == [1001, 1002, 1003, 1004, 1005]
    assert processed.dire_players == [2001, 2002, 2003, 2004, 2005]


def test_tensor_transformer_batch_shape():
    """Test that stacked batch produces correct (N, 24, 4) shape."""
    indexer = HeroIndexer()
    indexer.build_mapping([{"id": i, "playable": True} for i in range(1, 30)])

    validator = DraftValidator()
    transformer = TensorTransformer(indexer, validator)

    match_data = {
        "id": "7512345680",
        "radiantWin": True,
        "draft": {
            "picksBans": [
                {
                    "type": "pick",
                    "hero": {"id": i + 1},
                    "team": 0 if i % 4 < 2 else 1,
                    "order": i,
                }
                for i in range(24)
            ]
        }
    }

    processed = transformer.transform(match_data, source="stratz")
    assert processed is not None
    assert processed.x_tensor.shape == (24, 4)
    assert processed.x_tensor.dtype == torch.float32
    assert processed.y_tensor.shape == (1,)
    assert processed.y_tensor.dtype == torch.float32


def test_get_patch_id():
    from dota2drafter.processor.tensor_transformer import get_patch_id

    # Check that a timestamp before 7.37 (first patch) returns the first patch's ID (0)
    assert get_patch_id(1722383000) == 0

    # Check exact patch start timestamps
    assert get_patch_id(1722384000) == 0  # 7.37
    assert get_patch_id(1785369600) == 21 # 7.41e
    assert get_patch_id(1789430400) == 22 # 7.41f

    # Check times inside/between patches
    assert get_patch_id(1789430400 + 3600) == 22 # 1 hour after 7.41f starts
    assert get_patch_id(1785369600 + 86400) == 21 # 1 day after 7.41e starts

    # Check that None/empty timestamp returns the absolute latest patch ID (22)
    assert get_patch_id(None) == 22

