# Technical Specification: AUT-11 - Date-based League Filtering

## Overview
Replaces the `patch` based filtering with a date-based filtering mechanism (`cutoff_date`) for discovering Dota 2 leagues and their matches. This ensures we don't rely on potentially broken Stratz API patch IDs.

## 1. Configuration Changes
**Files**: `src/dota2drafter/config.py`, `config.yaml`
- Remove `patch: str` from `PipelineConfig`.
- Add `cutoff_date: str = "2026-06-04"` to `PipelineConfig` (representing an ISO 8601 date string).
- Update `load_config` to read `cutoff_date` from `config.yaml` instead of `patch`.

## 2. League Discovery & Date Filtering
**Files**: `src/dota2drafter/api/stratz_client.py`, `src/dota2drafter/discovery/league_mapper.py`
- **`fetch_leagues` Signature**: Remove the `patch: str` argument. Add `cutoff_date: str` argument.
- **GraphQL Params**: Remove `patchIds` from `request_params`.
- **Date Conversion**: Convert `cutoff_date` into a Unix timestamp (seconds) inside `fetch_leagues`. You can use `datetime.fromisoformat(cutoff_date).timestamp()`.
- **Filtering Logic**:
  - Remove the existing 14-day `ended` heuristic.
  - Filter out any league whose `lastMatchDate` is `None` or strictly `< cutoff_timestamp`.
  - Only return leagues that had matches on or after the `cutoff_date`.
- **`LeagueMapper.discover_leagues`**: Remove `self._config.patch` references. Pass `self._config.cutoff_date` to `fetch_leagues()`. Remove `patch` when calling `self._state_db.insert_league()`.

## 3. Match Discovery & Date Filtering
**Files**: `src/dota2drafter/api/stratz_client.py`, `src/dota2drafter/discovery/match_finder.py`
- **`fetch_matches_by_league` Signature**: Remove the `patch: str` argument. Add `cutoff_date: str` argument.
- **GraphQL Params**: Remove `patchIds`.
- **Date Filter**: Add a date filter to `request_params`. Convert `cutoff_date` to a Unix timestamp. The intent is to fetch matches that occurred *after* the `cutoff_date`. Note: The orchestrator discussed an `endDateTime` filter, but mathematically we need a `startDateTime` parameter (set to `cutoff_timestamp`) to restrict matches to those played after the patch release date. Add `startDateTime = int(cutoff_timestamp)` to `request_params` (or `endDateTime` if Stratz API specifically semantics demand it, but standard is `startDateTime`).
- **`MatchFinder.find_matches_for_league`**: Update to pass `self._config.cutoff_date` to `fetch_matches_by_league()` instead of `patch`.

## 4. Hero Discovery
**Files**: `src/dota2drafter/api/stratz_client.py`, `src/dota2drafter/main.py`
- **`fetch_heroes` Signature**: Drop the unused `patch` argument from `fetch_heroes()` in both the STRATZ client.
- **`main.py`**: Update `await stratz_client.fetch_heroes(config.patch)` to `await stratz_client.fetch_heroes()`.

## 5. State Database Cleanup
**File**: `src/dota2drafter/state.py`
- **`insert_league`**: Remove the `patch` parameter and remove it from the `INSERT OR REPLACE INTO leagues` query.
- **`_init_db` Schema**: Remove `patch TEXT NOT NULL` from the `CREATE TABLE leagues` schema.
- **Migration**: Add an `ALTER TABLE leagues DROP COLUMN patch` inside a `try/except sqlite3.OperationalError` block to migrate existing state databases.

## Execution Plan
The `SDD-Implementer` will checkout this branch, apply these changes across the configuration, API client, SQLite state database, and the pipeline orchestrator (`main.py`). The implementer should verify tests and ensure that the pipeline starts correctly with the new config.
