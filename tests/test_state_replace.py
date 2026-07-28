"""Tests for StateDatabase INSERT OR REPLACE behavior."""

import pytest
from pathlib import Path
from dota2drafter.state import StateDatabase


def test_upsert_matches_replaces_existing(tmp_path):
    """Verify that upsert_matches uses INSERT OR REPLACE to update existing matches."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1, "7.35")

    # Insert a match
    state_db.upsert_matches([("10001", "pending", "123")])
    assert state_db.get_pending_count() == 1

    # Update the same match with a different status
    state_db.upsert_matches([("10001", "completed", "123")])

    # The match should now be completed, not still pending
    assert state_db.get_pending_count() == 0
    stats = state_db.get_stats()
    assert stats["completed"] == 1


def test_upsert_matches_inserts_new(tmp_path):
    """Verify that upsert_matches inserts new matches correctly."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1, "7.35")

    state_db.upsert_matches([("10001", "pending", "123")])

    assert state_db.get_pending_count() == 1
    pending = state_db.get_pending_matches()
    assert pending[0] == ("10001", "123")


def test_upsert_matches_batch_replaces(tmp_path):
    """Verify that batch upsert_matches replaces existing matches."""
    db_path = tmp_path / "test_state.db"
    state_db = StateDatabase(db_path)

    state_db.insert_league("123", "DreamLeague", 1, "7.35")

    # Insert multiple matches
    state_db.upsert_matches([
        ("10001", "pending", "123"),
        ("10002", "pending", "123"),
    ])
    assert state_db.get_pending_count() == 2

    # Update one of them
    state_db.upsert_matches([("10001", "failed", "123")])

    # Should have 1 pending, 1 failed
    stats = state_db.get_stats()
    assert stats["pending"] == 1
    assert stats["failed"] == 1
