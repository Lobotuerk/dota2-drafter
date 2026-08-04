"""Tests for StateDatabase INSERT OR IGNORE behavior."""

import pytest
from pathlib import Path
from dota2drafter.state import StateDatabase


def test_upsert_matches_ignores_existing(tmp_path):
    """Verify that upsert_matches uses INSERT OR IGNORE and does not overwrite existing matches."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1)

    # Insert a match with status pending
    state_db.upsert_matches([("10001", "pending", "123")])
    assert state_db.get_pending_count() == 1

    # Attempt to upsert the same match with a different status (completed)
    state_db.upsert_matches([("10001", "completed", "123")])

    # The match should still be pending because the update was ignored
    assert state_db.get_pending_count() == 1
    stats = state_db.get_stats()
    assert stats.get("completed", 0) == 0
    assert stats.get("pending", 0) == 1


def test_upsert_matches_inserts_new(tmp_path):
    """Verify that upsert_matches inserts new matches correctly."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1)

    state_db.upsert_matches([("10001", "pending", "123")])

    assert state_db.get_pending_count() == 1
    pending = state_db.get_pending_matches()
    assert pending[0] == ("10001", "123")


def test_upsert_matches_batch_ignores(tmp_path):
    """Verify that batch upsert_matches ignores existing matches and keeps their status."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1)

    # Insert multiple matches
    state_db.upsert_matches([
        ("10001", "pending", "123"),
        ("10002", "pending", "123"),
    ])
    assert state_db.get_pending_count() == 2

    # Attempt to update one of them to failed
    state_db.upsert_matches([("10001", "failed", "123")])

    # Should still have 2 pending, 0 failed
    stats = state_db.get_stats()
    assert stats.get("pending", 0) == 2
    assert stats.get("failed", 0) == 0
