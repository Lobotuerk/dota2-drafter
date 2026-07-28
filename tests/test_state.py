import pytest
from pathlib import Path
from dota2drafter.state import StateDatabase, MatchStatus


def test_database_init(tmp_path):
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)
    assert db_path.exists()
    
    # Should get empty pending matches and stats on a fresh DB
    assert state_db.get_pending_matches() == []
    assert state_db.get_pending_count() == 0
    assert state_db.get_stats() == {}


def test_insert_league(tmp_path):
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)
    
    state_db.insert_league("123", "DreamLeague", 1)
    
    # Verify we can reference it
    state_db.upsert_matches([("10001", "pending", "123")])
    pending = state_db.get_pending_matches()
    assert len(pending) == 1
    assert pending[0] == ("10001", "123")
    assert state_db.get_pending_count() == 1


def test_match_status_transitions(tmp_path):
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)
    
    state_db.insert_league("123", "DreamLeague", 1)
    
    matches = [
        ("10001", "pending", "123"),
        ("10002", "pending", "123"),
        ("10003", "pending", "123"),
    ]
    state_db.upsert_matches(matches)
    
    assert state_db.get_pending_count() == 3
    
    # Mark completed
    state_db.mark_completed("10001", radiant_win=True)
    # Mark failed
    state_db.mark_failed("10002", "Timeout error")
    # Mark invalid
    state_db.mark_invalid("10003")
    
    assert state_db.get_pending_count() == 0
    
    stats = state_db.get_stats()
    assert stats["completed"] == 1
    assert stats["failed"] == 1
    assert stats["invalid"] == 1
