# Technical Specification: Switch to OpenDota API

## Objective
The STRATZ API is missing data and not functioning properly for recent matches. We need to switch our primary data ingestion source from STRATZ to the OpenDota API, specifically for discovering leagues, matches, and fetching the hero roster. STRATZ will remain as a fallback where applicable.

## Reference Implementation
The user has already created a reference implementation in the `opendota_api` branch (`origin/opendota_api`). It has been reviewed and tested. The Implementer should replicate the exact changes from the `opendota_api` branch into the current feature branch.

## Required Changes

### 1. `src/dota2drafter/api/opendota_client.py`
- Add an `asyncio.Lock()` and a `_last_request_time` variable to the `OpenDotaClient`.
- Implement rate limiting inside the `_get` method: enforce a 1.1s delay between consecutive OpenDota API calls to respect their 60 requests/minute limit.
- Add `fetch_leagues()`, `fetch_recent_pro_matches(less_than_match_id)`, and `fetch_league_matches(league_id)` methods.

### 2. `src/dota2drafter/discovery/league_mapper.py`
- Refactor `LeagueMapper.__init__` to accept `opendota_client` alongside `stratz_client`.
- Update `discover_leagues` to discover leagues exclusively via `OpenDotaClient`.
- Iterate over `fetch_recent_pro_matches` to gather all active leagues within the `cutoff_date`.
- Map OpenDota string tiers ("premium", "professional") to integer tiers (1, 2).
- Apply Liquipedia string-matching rules to correct any misclassified tiers.

### 3. `src/dota2drafter/discovery/match_finder.py`
- Refactor `MatchFinder.__init__` to accept `opendota_client`.
- Update `find_matches_for_league` to fetch match IDs exclusively via OpenDota (`fetch_league_matches`), applying the config `cutoff_date` filtering.

### 4. `src/dota2drafter/main.py`
- Step 1 (Heroes): Attempt to fetch the hero roster from `opendota_client` first. If it fails, fall back to `stratz_client`. Mark all OpenDota heroes as `playable: True`.
- Step 2 (Leagues): Pass `opendota_client` into `LeagueMapper`.
- Step 3 (Matches): Pass `opendota_client` into `MatchFinder`. Add a guard to skip match discovery if no new leagues were found.
- Step 4 (Processing): Keep trying STRATZ first for detailed match data, but update fallback logging to be clearer. Ensure `radiantWin` parsing checks `radiant_win` (snake_case) as returned by OpenDota.

### 5. `config.yaml`
- Update `cutoff_date` to `2026-05-06`.

### 6. Tests (`tests/test_main.py`)
- Update mock initializations in tests to match the new `LeagueMapper` and `MatchFinder` signatures.
- Add tests confirming OpenDota fallback behavior for `fetch_heroes`.
- Add tests to ensure `LeagueMapper` successfully discovers leagues via OpenDota mock data.

## Implementation Steps
For the SDD-Implementer:
1. `git checkout origin/opendota_api -- src/ tests/ config.yaml` (or manually apply the diff from PR #14).
2. Verify tests pass (`pytest`).
3. Commit the changes and open a PR against `main`.