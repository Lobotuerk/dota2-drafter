### 📋 Technical Specification

#### 1. Dataset Generation: `TensorTransformer` and `DatasetBuilder`
**Objective:** Propagate the per-player hero pick from raw API match payloads down to the `.pt` batches.

- **`TensorTransformer._extract_stratz_players` & `_extract_opendota_players`**
  Modify both methods to extract both `account_id` and the mapped `hero_id`. 
  - If `account_id` is missing (anonymous player), use `0`. 
  - If `hero_id` is missing or fails to map, use `-1`.
  - Return `tuple[list[int], list[int]]` representing `(account_ids, hero_ids)`.
  - Ensure both lists are exactly 5 elements long by zero-padding `(0, -1)` if the raw payload contains fewer players for a team.
- **`ProcessedMatch` Dataclass**
  Add two fields: `radiant_heroes: list[int] = field(default_factory=list)` and `dire_heroes: list[int] = field(default_factory=list)`.
- **`TensorTransformer.transform_stratz` & `transform_opendota`**
  Update the instantiation of `ProcessedMatch` to include `radiant_heroes` and `dire_heroes` using the unpacked tuples from the extraction functions.
- **`DatasetBuilder._flush`**
  Gather `m.radiant_heroes` and `m.dire_heroes` from the `_buffer` into lists, and save them under the keys `"radiant_heroes"` and `"dire_heroes"` in the saved `.pt` dictionary.

#### 2. Comfort Matrix Generation: `scripts/01b_build_comfort.py`
**Objective:** Compute a per-player comfort vector based on historical wins and losses using the newly extracted `radiant_heroes` and `dire_heroes`.

- **Computation Logic:**
  Instead of assigning zero/random tensors, iterate through all match batches (`radiant_players`, `dire_players`, `radiant_heroes`, `dire_heroes`, and `y`). Maintain an intermediate dictionary counting wins and losses for each `(account_id, hero_id)` pair. Ignore anonymous players (`account_id == 0`) and unmapped heroes (`hero_id == -1`).
  - **Raw Score:** For each player p and hero h, `W_p(h) - L_p(h)` (where a win adds 1, a loss subtracts 1).
- **Normalization:**
  Normalize the comfort vector for each player using **L2 normalization**: divide the vector by `max(1.0, L2_norm)`. This bounds the influence of extreme hero spammers while maintaining relative preference magnitudes and ensures unplayed heroes remain exactly `0.0` (neutral).
- **Implementation Update:**
  The script must determine `vocab_size` (the number of heroes). Update the arguments to accept `--vocab_size` (defaulting to 124) or load it dynamically from `data/hero_indexer.json`. Initialize each player's comfort tensor to shape `(vocab_size,)`.

#### 3. Model Architecture: `PlayerComfortNetwork`
**Objective:** Adapt the pipeline downstream models to ingest the new feature dimension (`C = num_heroes`).

- **`scripts/04_train_transformer.py`**
  Update the instantiation of `PlayerComfortNetwork` and `PlayerComfortDataset`. Fetch `vocab_size` from the `HeroIndexer` instance and use it to replace the hardcoded `player_input_dim=10`.
  - The `PlayerComfortNetwork` input changes from `(B, 10, 10)` to `(B, 10, vocab_size)`.
  - The `PlayerComfortEmbedding` output remains size `d_model` (e.g., `(B, 10, d_model)`), preserving downstream compatibility without architectural rewrites.
- **`PlayerComfortDataset._build_player_comfort`**
  With `radiant_players` and `dire_players` strictly padded to length 5 in `DatasetBuilder`, this function will cleanly produce a `(10, C)` tensor. For anonymous accounts (`account_id == 0`), it will securely yield a zero vector.
