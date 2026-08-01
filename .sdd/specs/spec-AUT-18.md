### 📋 Technical Specification

#### 1. Overview
A new script `scripts/01c_add_custom_player.py` will be created to allow adding or updating custom player profiles in the player comfort tensor map (`data/player_comfort.pt`). This allows the drafter to prioritize a provided list of "comfort heroes" when recommending picks for that custom player ID.

#### 2. CLI Interface
The script should be executable via the command line with the following arguments:
- `--id` (int, required): The numeric Steam account ID of the custom player.
- `--heroes` (str, required): A comma-separated list of hero names (e.g., `Pudge,Anti-Mage`).
- `--comfort` (str, default: `data/player_comfort.pt`): Path to the existing player comfort PyTorch file.
- `--hero_mapping` (str, default: `hero_mapping.json`): Path to the mapping from API IDs to hero names.
- `--hero_indexer` (str, default: `data/hero_indexer.json`): Path to the indexer mapping API IDs to contiguous tensor indices.
- `--force` (flag): If provided, allows overwriting an existing `id` in the comfort map. If the `id` exists and this flag is not provided, the script must abort.

#### 3. Execution Flow
1. **Load Hero Mapping**: Read `hero_mapping.json` and invert the dictionary to create a `name_to_api_id` mapping. Standardize names (e.g., strip whitespace) to ensure robust lookups.
2. **Load Hero Indexer**: Read `hero_indexer.json`. Instantiate `src.dota2drafter.processor.hero_indexer.HeroIndexer` and build the mapping:
   ```python
   indexer = HeroIndexer()
   with open(args.hero_indexer) as f:
       hero_data = json.load(f)
   heroes = [{"id": int(api_id), "playable": True} for api_id in hero_data.keys()]
   indexer.build_mapping(heroes)
   ```
3. **Parse Target Heroes**:
   Split the `--heroes` argument by comma. For each hero name:
   - Look up the `api_id` in `name_to_api_id`. Raise a `ValueError` with a clean error message if the name is not found.
   - Look up the contiguous index `idx = indexer.map_hero_id(int(api_id))`. Raise a `ValueError` if it returns `None`.
   - Collect these contiguous indices.
4. **Load Comfort Map**:
   - Load the dictionary mapping via `comfort_map = torch.load(args.comfort, weights_only=True)`.
   - Check if `args.id` exists in `comfort_map`. If it does and `--force` is false, raise a `ValueError`.
5. **Determine Tensor Dimension (`vocab_size`)**:
   - To ensure absolute dimension compatibility with `torch.stack` during interactive drafting, derive `vocab_size` from an existing tensor in `comfort_map` (e.g., `vocab_size = next(iter(comfort_map.values())).size(0)`).
   - If `comfort_map` is empty, fallback to `vocab_size = indexer.get_contiguous_count()`.
6. **Construct the Vector**:
   - Initialize `raw_vector = torch.zeros(vocab_size, dtype=torch.float32)`.
   - For each collected `idx`, assign `raw_vector[idx] = 1.0`. Wait, to prevent index errors, the script should dynamically ensure `vocab_size` is at least `max(idx) + 1` if it needs to expand, or safely raise an error if an index exceeds `vocab_size`.
   - Apply L2 Normalization: `l2_norm = raw_vector.norm().item()`, then `normalized_vector = raw_vector / max(1.0, l2_norm)`. This matches the behavior of `01b_build_comfort.py`.
7. **Save**:
   - Update `comfort_map[args.id] = normalized_vector`.
   - Write back to disk with `torch.save(comfort_map, args.comfort)`.

#### 4. Design Philosophy Considerations
- **High Modular Depth / Simple Interfaces**: The script isolates file parsing and vector math. It leverages the existing `HeroIndexer` rather than re-inventing index generation, ensuring index logic remains consistent with the rest of the application.
- **Clean Exception Paths**: Input validation happens eagerly (e.g., validating all hero names before modifying the tensor) so that the script fails fast and explicitly without leaving intermediate state. Overwriting requires an explicit `--force` flag.
