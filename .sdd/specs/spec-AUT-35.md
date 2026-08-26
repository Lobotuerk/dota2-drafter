# Technical Specification: AUT-35 Relational Graph Improvement

This specification outlines the transition of the hero interaction graph construction from raw win-rate and count-based pruning to a **Patch-Weighted Wilson Score** model. It removes the deprecated `build_hero_graph` method, refactors `build_pruned_hero_graph` to support configurable thresholds, updates all caller scripts, and designs a comprehensive testing strategy to verify the correctness of the new graph-building behavior.

---

## 1. File Structure Changes

The following files will be created or modified by the implementer agent:

- **Modified production files**:
  - `src/dota2drafter/embeddings/data_extractor.py`: Remove `build_hero_graph()`; completely rewrite `build_pruned_hero_graph()`; add helper `_wilson_score_eff()`.
  - `src/dota2drafter/config.py`: Add `GraphConfig` dataclass and integrate it into `PipelineConfig`.
  - `config.yaml`: Add configurable fields for `graph.wilson_threshold` and `graph.gamma`.
- **Modified script and pretrainer files**:
  - `scripts/03_train_rgcn.py`: Replace `build_hero_graph()` with `build_pruned_hero_graph()`.
  - `scripts/04_train_transformer.py`: Replace `build_pruned_hero_graph()` counts-percentile logic with the new signature/config.
  - `scripts/06_test_rgcn.py`: Replace `build_pruned_hero_graph()` percentile calls with the new signature/config.
  - `scripts/interactive_draft.py`: Replace `build_hero_graph()` with `build_pruned_hero_graph()`.
  - `src/dota2drafter/embeddings/pretrainer.py`: Replace `build_pruned_hero_graph()` call.
  - `src/dota2drafter/embeddings/train_rgcn.py`: Replace `build_pruned_hero_graph()` call.
- **Modified/Updated test files**:
  - `tests/test_embeddings/test_data_extractor.py`: Re-write/align all test functions that depend on the old signatures. Add dedicated test cases for patch-weighted Wilson scoring.

---

## 2. Interfaces, Signatures & Configs

### A. Configuration: `config.yaml`
Add a new nested `graph` section under `config.yaml` to make the new model parameters easily tuneable:
```yaml
graph:
  wilson_threshold: 0.50
  gamma: 0.80
```

### B. Config Class: `src/dota2drafter/config.py`
Introduce a new dataclass `GraphConfig` and add it as a field in `PipelineConfig`:
```python
@dataclass
class GraphConfig:
    wilson_threshold: float = 0.50
    gamma: float = 0.80

@dataclass
class PipelineConfig:
    # Existing fields...
    graph: GraphConfig = field(default_factory=lambda: GraphConfig())
```
Ensure that `load_config` properly parses the nested `graph` parameters from the raw yaml:
```python
def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    # ...
    raw_graph = raw.get("graph", {})
    graph_conf = GraphConfig(
        wilson_threshold=raw_graph.get("wilson_threshold", 0.50),
        gamma=raw_graph.get("gamma", 0.80),
    )
    # ...
```

### C. Signature of `build_pruned_hero_graph` in `data_extractor.py`
Remove `build_hero_graph()`. Update the signature of `build_pruned_hero_graph()` to accept the new configuration parameters. Keep `percentile_keep` with a default `None` as a deprecated fallback (for backward-compatibility with untuned scripts) but ignore it internally:
```python
def build_pruned_hero_graph(
    self,
    batches: list[dict[str, Any]],
    wilson_threshold: float = 0.50,
    gamma: float = 0.80,
    percentile_keep: float | None = None,
) -> Data:
```

---

## 3. Detailed Algorithmic Design

The new graph construction workflow follows a rigorous **discount-before-Wilson** order of operations:

### Step 1: Current Patch Discovery
Identify the maximum patch ID across all batches to serve as $P_{\text{current}}$:
```python
max_patch_id = 0
has_patches = False
for batch in batches:
    p_ids = batch.get("patch_ids")
    if p_ids is not None and p_ids.numel() > 0:
        has_patches = True
        batch_max = int(p_ids.max().item())
        if batch_max > max_patch_id:
            max_patch_id = batch_max

P_current = max_patch_id if has_patches else 0
```

### Step 2: Weighted Statistics Accumulation
For each match index `match_idx` in batch $B$, retrieve the patch ID:
```python
p_ids = batch.get("patch_ids")
p_id = int(p_ids[match_idx].item()) if (has_patches and p_ids is not None and match_idx < len(p_ids)) else P_current
```
Compute major patch distance:
$$\Delta P = \max(0, P_{\text{current}} - p_id)$$
Compute match weight:
$$w_m = \gamma^{\Delta P}$$

Accumulate the following floating-point variables for each pair/key in the respective relation types:
- $N_w = \sum w_m$ (weighted count)
- $W_w = \sum y_m \cdot w_m$ (weighted wins)
- $S_w = \sum w_m^2$ (sum of squared weights)
- `total_count` (unweighted total observation count, used solely for the hard floor)

#### Outcome $y_m \in \{0, 1\}$ definition per relation:
1. **Synergy (type 0)** (Undirected pairs co-picked on the same team):
   - On Radiant: $y_m = \text{radiant\_win}$
   - On Dire: $y_m = 1 - \text{radiant\_win}$
2. **Antagonist (type 1)** (Directed pair $u \rightarrow v$ meaning $u$ counter-picks $v$, where $u$ is on the winning team, $v$ on the losing team):
   - $u$ on Radiant, $v$ on Dire: $y_m = \text{radiant\_win}$
   - $u$ on Dire, $v$ on Radiant: $y_m = 1 - \text{radiant\_win}$
3. **Required Bans (type 2)** (Directed pair $X \rightarrow Y$ meaning picked hero $X$ requires $Y$ banned):
   - $X$ on Radiant: $y_m = \text{radiant\_win}$
   - $X$ on Dire: $y_m = 1 - \text{radiant\_win}$

### Step 3: Effective Wilson Score Calculation
For each unique relationship key that satisfies the unweighted hard count threshold:
- **Synergy (type 0)**: `total_count >= 5`
- **Antagonist (type 1)**: `total_count >= 3`
- **Required Bans (type 2)**: `total_count >= 3`

Calculate effective sample size and weighted proportion:
$$n_{\text{eff}} = \frac{N_w^2}{S_w}$$
$$\hat{p} = \frac{W_w}{N_w}$$

Pass these to the standard Wilson lower bound formula (using $z = 1.96$ for 95% confidence level):
```python
def _wilson_score_eff(p_hat: float, n_eff: float, z: float = 1.96) -> float:
    if n_eff <= 0:
        return 0.5
    # Clamp p_hat to ensure numerical safety under sqrt
    p_hat = max(0.0, min(1.0, p_hat))
    denominator = 1 + z**2 / n_eff
    center = p_hat + z**2 / (2 * n_eff)
    spread = z * math.sqrt((p_hat * (1 - p_hat) / n_eff) + z**2 / (4 * n_eff**2))
    return (center - spread) / denominator
```

### Step 4: Pruning & Graph Output
- Prune (exclude) any edge where the computed Wilson Score $\le \text{wilson\_threshold}$ (default `0.50`).
- For surviving edges, assign:
  $$\text{edge\_weight} = \text{Wilson Score}$$
- Build and return the PyG `Data` object. Ensure synergy edges are added bidirectionally: if $(h_i, h_j)$ survives, both $h_i \rightarrow h_j$ and $h_j \rightarrow h_i$ are added to the edge list with the identical Wilson score weight.

---

## 4. Edge Cases & Fallbacks

- **Zero Variance / Constant Wins**: Clamping $\hat{p}$ to $[0.0, 1.0]$ prevents domain errors during `math.sqrt`.
- **No Batches / Empty Dataset**: If there are no matches or no edges survive, return an empty PyG `Data` object containing long tensor shapes `(2, 0)` for `edge_index` and `(0, 1)` for `edge_weight` to prevent runtime crashes.
- **Missing `patch_ids`**: If `batch.get("patch_ids")` is missing or `None`, default `p_id` to $P_{\text{current}}$ (yielding $\Delta P = 0$ and $w_m = 1.0$). This ensures backward compatibility with older dataset formats.

---

## 5. Testing & Verification Strategy

The implementer must update the test suite in `tests/test_embeddings/test_data_extractor.py` to cover:

1. **Test Signature compatibility**: Align existing tests to use `build_pruned_hero_graph` instead of `build_hero_graph`.
2. **Test Patch Discounting Calculation**:
   - Construct a batch with 2 matches on different patch IDs (e.g. Patch 21 and Patch 19).
   - Set $\gamma = 0.80$. Verify that the match on Patch 19 gets decaying weight $w_m = 0.64$ and correctly scales the accumulated $N_w, W_w, S_w$ values.
3. **Test Effective Sample Size $n_{\text{eff}}$**:
   - Assert that $n_{\text{eff}} \le N_{\text{total}}$ when unequal patch games exist, matching standard effective sample size theory.
4. **Test Threshold Pruning**:
   - Validate that edges with a computed Wilson lower bound $\le 0.50$ are pruned.
   - Validate that edges with a score $> 0.50$ are kept and their `edge_weight` equals the exact Wilson Score.
