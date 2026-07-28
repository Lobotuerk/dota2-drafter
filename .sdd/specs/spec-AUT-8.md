### 📋 Technical Specification

## 1. Context and Scope
This document outlines the design for resolving issues identified in PR #7 (`practical_fixes`) for the `Auto-drafter` project. The orchestrator identified several regressions and architectural concerns that block proper pipeline execution and model training.

## 2. Proposed Implementation

### 2.1 API & Data Retrieval (`src/dota2drafter/api/stratz_client.py`)
- **Patch Filtering:** The PR removed `patch` from the GraphQL queries (`LEAGUES_QUERY`, `MATCHES_BY_LEAGUE_QUERY`). We will restore patch filtering by injecting the `patchIds: [$patch]` argument into the `request` variable payload for these queries according to the STRATZ GraphQL schema.
- **Connection Pooling & Retry Jitter:** The `StratzClient` currently instantiates a new `aiohttp.ClientSession` on every `_graphql` call. The design requires initializing a persistent `ClientSession` as a class member and adding an async `close()` method. We will also add jitter to the `tenacity` retry configuration (e.g., `wait_random_exponential`) to prevent simultaneous worker retries on `429 Too Many Requests`.
- **Top-level Imports:** Ensure `aiohttp` and `asyncio` are exclusively imported at the top of the file.

### 2.2 State Management (`src/dota2drafter/state.py`)
- **Transient Error Handling:** PR #7 changed `upsert_matches` to use `INSERT OR IGNORE`. This creates data gaps because transiently failed matches are ignored in subsequent runs. We will revert `upsert_matches` to use `INSERT OR REPLACE` (or `INSERT ... ON CONFLICT(match_id) DO UPDATE SET status=excluded.status WHERE matches.status = 'failed'`). The ended-league filtering introduced in PR #7 already prevents unnecessary API limits, so reverting this behavior is safe and correct.

### 2.3 Model Architecture (`src/dota2drafter/models/match_network.py`)
- **Device Alignment:** The `JointEmbedding.forward` method uses `h_gnn_dev = self.h_gnn.to(hero_indices.device)`, causing an unnecessary tensor copy on every forward pass. Since `self.h_gnn` is a registered buffer, it natively aligns with the module's device. We will remove the `.to()` cast and directly compute `hero_embeds = self.h_gnn[hero_indices]`.

### 2.4 Pretraining & Extraction (`src/dota2drafter/embeddings/pretrainer.py` & `data_extractor.py`)
- **Double Batch Loading:** `train_embeddings` instantiates a `temp_extractor`, loads batches into memory to find `max_hero_idx`, and then initializes another `extractor`. The design requires resolving `max_hero_idx` more gracefully. The solution is to remove `temp_extractor`, let `load_batches` parse the batches once as a standalone utility or class method, or infer `max_hero_idx` from the batches before instantiating the extractor.
- **Node Count Mismatch:** Ensure that `build_hero_graph` always returns `num_nodes=self._num_heroes + 1` across all return paths, ensuring index `0` (padding) is properly accounted for in the downstream GNN.

### 2.5 Code Cleanup & Testing (`src/dota2drafter/dataset/builder.py`, `tests/`)
- **Dead Code:** Remove the unused `state_db` parameter from the `DatasetBuilder.__init__` signature.
- **Model Loading:** Update `load_h_gnn` to accept `model_path` as an argument instead of hardcoding the path inside the function.
- **Test Fixes:**
  - Fix the `return_return` typo in `tests/test_main.py` (change to `return_value`).
  - Introduce new tests verifying the ended-league filtering, connection pooling logic, and temporary-error recovery.

## 3. Execution Plan for SDD-Implementer
1. Start with `src/dota2drafter/api/stratz_client.py`: Refactor `ClientSession` to be persistent, restore patch filtering, and update retry strategies.
2. Update `src/dota2drafter/state.py`: Revert to `INSERT OR REPLACE` or targeted updates for transient errors.
3. Fix the tensor device alignment in `src/dota2drafter/models/match_network.py`.
4. Refactor `pretrainer.py` to eliminate `temp_extractor`.
5. Clean up `builder.py` and `tests/test_main.py`.
6. Write missing unit tests and verify the entire test suite passes (`pytest`).