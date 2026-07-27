# Technical Specification: Dota 2 Draft Ingestion Pipeline

## Overview
This specification details the design for a Python-based data ingestion pipeline that fetches, validates, and transforms Dota 2 Captains Mode draft sequences into PyTorch-ready tensor datasets (`.pt`). The pipeline aggregates historical match data from the STRATZ GraphQL and OpenDota REST APIs, processes the 24-step pick/ban sequences, maps heroes to contiguous indices dynamically per patch, and supports resumable, concurrent execution.

## Architecture & Modules

### 1. Configuration & State Management
- **Config Loader (`config.yaml` / `.env`):** Defines parameters such as the target `patch` (e.g., "7.35"), tournament tiers (1 and 2), concurrent worker limits, rate limit thresholds, and the output directory. The `STRATZ_API_KEY` is loaded securely via `.env` (which must be gitignored).
- **State Database (`state.db`):** A local SQLite database to track progress. It will store:
  - `leagues`: Discovered league IDs and metadata.
  - `matches`: Match IDs with their processing status (`pending`, `completed`, `failed`, `invalid`). This ensures the pipeline is incremental and resumable across runs.

### 2. API Clients (`api/`)
- **STRATZ Client (`stratz_client.py`):** An asynchronous GraphQL client (using `aiohttp` or `httpx`). Responsible for:
  - Fetching tier 1 & 2 leagues.
  - Querying all match IDs associated with those leagues.
  - Fetching detailed match information (specifically the draft sequence) when necessary.
- **OpenDota Client (`opendota_client.py`):** An asynchronous REST client. Responsible for:
  - Fetching the active hero pool for the target patch to build the dynamic hero index mapping.
  - Acting as a fallback for match detail payloads if STRATZ lacks specific draft granularities.
- **Resilience:** Both clients implement exponential backoff, retry logic for 429/5xx errors, and concurrency limiters (e.g., `asyncio.Semaphore`).

### 3. Discovery Engine (`discovery/`)
- **League Mapper:** Interfaces with the APIs (and potentially a scraper for Liquipedia if APIs lack clear tier labels) to resolve tier 1 and tier 2 tournament names to actual League IDs for the specified patch.
- **Match Finder:** Iterates through the resolved League IDs to collect all associated Match IDs and populates the `state.db`.

### 4. Data Processing (`processor/`)
- **Hero Indexer:** Queries the API to determine the total active hero universe `K` for the patch. It generates a bidirectional mapping between the sparse API `hero_id` integers and a contiguous integer space `h ∈ {1, ..., K}`.
- **Draft Validator:** Inspects raw match payloads. It discards matches that:
  - Are not Captains Mode.
  - Do not have exactly 24 steps in their `picks_bans` array.
- **Tensor Transformer:** Converts a valid 24-step sequence into numerical arrays. 
  - Each step is encoded as `[is_pick, team, hero_index]` where `is_pick` is 1 for pick / 0 for ban, `team` is 0 for Radiant / 1 for Dire, and `hero_index` is the contiguous integer.
  - Includes match metadata (e.g., `match_id`, `radiant_win`).

### 5. Dataset Builder (`dataset/`)
- Aggregates processed match arrays.
- Converts arrays to `torch.Tensor`.
- Saves the dataset natively as `.pt` files using `torch.save()`. Can save in chunks to prevent memory bloat during massive historical fetches.

## Execution Flow

1. **Initialization:** Load configuration and ensure `.env` is present. Initialize the SQLite state DB.
2. **Hero Mapping:** Fetch the patch's hero list and create the contiguous index map.
3. **League Discovery:** Discover all tier 1 & 2 leagues for the patch; store in DB.
4. **Match Discovery:** Fetch all match IDs for these leagues; insert as `pending` in DB.
5. **Concurrent Fetch & Process (The Event Loop):**
   - Pull batches of `pending` match IDs from the DB.
   - Fetch match details asynchronously.
   - Validate and transform the payload.
   - Update the DB status to `completed` or `invalid`.
   - Append processed tensors to a buffer.
6. **Flush to Disk:** Periodically, and at the end of the run, flush the tensor buffer to `.pt` files.

## Data Schema (PyTorch Tensor Structure)
A single processed match in the dataset will consist of:
- `x`: A `(24, 3)` tensor representing the sequential picks and bans.
- `y`: A `(1,)` tensor representing `radiant_win` (1 if Radiant won, 0 otherwise), useful for downstream predictive modeling.
- `match_id`: For traceability.
