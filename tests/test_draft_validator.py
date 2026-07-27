import pytest
from dota2drafter.processor.draft_validator import DraftValidator


def test_validate_opendota():
    validator = DraftValidator()
    
    # Valid payload
    valid_payload = {
        "match_id": 12345,
        "radiant_win": True,
        "game_mode": 2,  # Captains Mode
        "picks_bans": [{"is_pick": True, "hero_id": i, "team": 0, "order": i} for i in range(24)]
    }
    assert validator.validate(valid_payload, source="opendota") is True
    
    # Missing radiant win
    invalid_payload = valid_payload.copy()
    del invalid_payload["radiant_win"]
    assert validator.validate(invalid_payload, source="opendota") is False
    
    # Invalid game mode
    invalid_payload = valid_payload.copy()
    invalid_payload["game_mode"] = 1  # All Pick
    assert validator.validate(invalid_payload, source="opendota") is False
    
    # Incomplete picks_bans (less than 24 steps)
    invalid_payload = valid_payload.copy()
    invalid_payload["picks_bans"] = invalid_payload["picks_bans"][:23]
    assert validator.validate(invalid_payload, source="opendota") is False


def test_validate_stratz():
    validator = DraftValidator()
    
    # Valid payload
    valid_payload = {
        "id": "12345",
        "draft": {
            "picksBans": [{"type": "pick", "hero": {"id": i}, "team": 0, "order": i} for i in range(24)]
        }
    }
    assert validator.validate(valid_payload, source="stratz") is True
    
    # Invalid draft length
    invalid_payload = {
        "id": "12345",
        "draft": {
            "picksBans": [{"type": "pick", "hero": {"id": i}, "team": 0, "order": i} for i in range(23)]
        }
    }
    assert validator.validate(invalid_payload, source="stratz") is False
