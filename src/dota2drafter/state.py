"""State management using SQLite for incremental pipeline execution."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Generator

from dota2drafter.config import PipelineConfig


class MatchStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


class StateDatabase:
    """SQLite-backed state tracker for pipeline progress."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leagues (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    tier INTEGER NOT NULL,
                    patch TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    league_id TEXT,
                    radiant_win INTEGER,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (league_id) REFERENCES leagues(id)
                )
            """)
            conn.commit()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_league(self, league_id: str, name: str, tier: int, patch: str) -> None:
        """Insert or update a league record."""
        with self._connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO leagues (id, name, tier, patch)
                   VALUES (?, ?, ?, ?)""",
                (league_id, name, tier, patch),
            )

    def upsert_matches(self, matches: list[tuple[str, str, str | None]]) -> None:
        """Insert or update multiple match records.

        Each tuple is (match_id, status, league_id).
        """
        with self._connection() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO matches (match_id, status, league_id)
                   VALUES (?, ?, ?)""",
                matches,
            )

    def get_pending_matches(self, limit: int = 100) -> list[tuple[str, str]]:
        """Get pending matches for processing.

        Returns list of (match_id, league_id) tuples.
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT match_id, league_id FROM matches
                   WHERE status = 'pending'
                   ORDER BY created_at
                   LIMIT ?""",
                (limit,),
            )
            return [(row["match_id"], row["league_id"]) for row in cursor.fetchall()]

    def mark_completed(self, match_id: str, radiant_win: bool | None = None) -> None:
        """Mark a match as completed."""
        with self._connection() as conn:
            updates = {"status": "completed", "updated_at": "CURRENT_TIMESTAMP"}
            if radiant_win is not None:
                updates["radiant_win"] = 1 if radiant_win else 0
            sets = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [match_id]
            conn.execute(
                f"UPDATE matches SET {sets} WHERE match_id = ?",
                values,
            )

    def mark_failed(self, match_id: str, error: str) -> None:
        """Mark a match as failed with an error message."""
        with self._connection() as conn:
            conn.execute(
                """UPDATE matches SET status = 'failed', error = ?,
                   updated_at = CURRENT_TIMESTAMP WHERE match_id = ?""",
                (error, match_id),
            )

    def mark_invalid(self, match_id: str) -> None:
        """Mark a match as invalid (failed validation)."""
        with self._connection() as conn:
            conn.execute(
                """UPDATE matches SET status = 'invalid',
                   updated_at = CURRENT_TIMESTAMP WHERE match_id = ?""",
                (match_id,),
            )

    def get_stats(self) -> dict[str, int]:
        """Get processing statistics."""
        with self._connection() as conn:
            cursor = conn.execute(
                """SELECT status, COUNT(*) as count FROM matches GROUP BY status"""
            )
            return {row["status"]: row["count"] for row in cursor.fetchall()}

    def get_pending_count(self) -> int:
        """Get the count of pending matches."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as cnt FROM matches WHERE status = 'pending'"
            )
            return cursor.fetchone()["cnt"]
