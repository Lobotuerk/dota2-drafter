### 📋 Technical Specification

#### 1. Architectural Strategy & Design Philosophy
This update splits the previously monolithic data gathering pipeline into two distinct phases to support human-in-the-loop validation of leagues, eliminating noise caused by automated heuristic assumptions.
- **Deep Modules:** The gathering core (e.g. `MatchFinder`, `DatasetBuilder`, `StateDatabase`) remains encapsulated and unaware of the JSON human-review step.
- **Clear Boundaries:** The data schema acts as the interface (`data/leagues.json`). The discovery pipeline is the sole writer (appending unreviewed entries), and the match-gather pipeline is purely a reader (consuming only approved ones).
- **Clean Exception Paths:** Re-batching corrupted `.pt` datasets handles unexpected partial-state scenarios cleanly in a dedicated maintenance script, rather than bloating the main gather loop.

#### 2. Scripts Refactoring
- **Create `scripts/01a_gather_leagues.py` (Discovery Phase):**
  - Loads configuration (`config.yaml`).
  - Calls `run_discovery_pipeline(config)` inside `main.py`.
- **Create `scripts/01b_gather_matches.py` (Gather Phase):**
  - Loads configuration (`config.yaml`).
  - Calls `run_match_gather_pipeline(config)` inside `main.py`.
- **Create `scripts/01e_cleanup_unapproved.py` (Validation Script):**
  - Opens `data/leagues.json` and loads all approved league IDs (`review == true`).
  - Scans `state.db` and flags any `completed` matches linked to an unapproved `league_id`.
  - Removes unapproved matches from `state.db`.
  - Iterates over all `.pt` batches in `data/`, removes tensors corresponding to unapproved match IDs, and re-chunks the data uniformly (batch size = 1000) using `DatasetBuilder`.
- **Retire and Rename Existing Scripts:**
  - Delete `scripts/01_gather_data.py`.
  - Move `scripts/01b_build_comfort.py` -> `scripts/01c_build_comfort.py`.
  - Move `scripts/01c_add_custom_player.py` -> `scripts/01d_add_custom_player.py`.

#### 3. Pipeline Modifications (`src/dota2drafter/main.py`)
- **Deprecate `run_pipeline`**, splitting it into two distinct functions:
  - **`run_discovery_pipeline(config: PipelineConfig)`:**
    1. Re-use hero mapping initialization (Step 1).
    2. Invoke `league_mapper.discover_leagues()`.
    3. Read `data/leagues.json` (if it exists).
    4. For any newly discovered league not in the JSON, append a new dictionary containing keys: `id`, `name`, `tier`, `review: false`, `start_date`, and `end_date`.
    5. Save the updated list back to `data/leagues.json`.
  - **`run_match_gather_pipeline(config: PipelineConfig)`:**
    1. Read `data/leagues.json`.
    2. Filter to extract only the leagues where `review == true`.
    3. Instantiate `MatchFinder` and pass this approved league list to `find_all_matches`.
    4. Maintain the current behavior for Steps 4 & 5: concurrently fetch, transform, save to `state.db`, and build `.pt` dataset batches.

#### 4. League Mapper Refactoring (`src/dota2drafter/discovery/league_mapper.py`)
- **Remove Substring Filtering:** Delete `LIQUIPEDIA_TIER_1_NAMES`, `LIQUIPEDIA_TIER_2_NAMES`, and the `_is_tier_match` heuristic.
- **Update `discover_leagues()`:**
  - Query OpenDota pro matches.
  - Collect *all* leagues having matches with `start_time >= cutoff_date`.
  - Skip tier inference/filtering. Simply extract the league `id`, `name`, API-provided `tier`, and infer `start_date`/`end_date` from the match boundary timestamps observed during the query (or default them if unavailable).
  - Return a raw list of league dictionaries that the orchestrating script will merge into the JSON structure.

#### 5. Data Contracts
- **JSON Location:** `data/leagues.json`
- **Schema per League Entry:**
  ```json
  {
      "id": 12345,
      "name": "The International",
      "tier": 1,
      "review": false,
      "start_date": "2026-05-15",
      "end_date": "2026-06-01"
  }
  ```
