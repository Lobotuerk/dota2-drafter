# Technical Specification: AUT-14 Embedding Usage Scripts

## Overview
This specification details the design for two new scripts, `scripts/05_test_skip_gram.py` and `scripts/06_test_rgcn.py`, which will test the Skip-Gram + DGI embeddings and query the RGCN training graph respectively. Both scripts accept a hero name, resolve the corresponding ID via `hero_mapping.json`, and output the top 5 closest/most relevant heroes based on the respective models and relations.

## Architectural Design & Philosophy
Following our Design Philosophy:
- **Clean Interfaces**: The scripts act as CLI entry points, relying on standard `argparse` for parameters.
- **Deep Modules**: The existing `load_frozen_embeddings` and `DataExtractor` will do the heavy lifting of model loading and graph building. The scripts will only coordinate the loading, inference, and result formatting.
- **Error Handling**: Missing models, missing data directories, or invalid hero names will be gracefully caught and reported using clear error messages via `rich.console`.

## 1. Skip-Gram Test Script (`scripts/05_test_skip_gram.py`)
**Purpose**: Prints the 5 closest heroes to a given hero based on cosine similarity in the Skip-Gram + DGI embedding space.

### Workflow
1. **Argument Parsing**: Accept `--hero_name` (e.g., "Anti-Mage"), `--model_path` (default `models/skip_gram_dgi.pt`), `--mapping_file` (default `hero_mapping.json`), `--embed_dim` (default 64), and `--num_heroes` (default 124).
2. **Hero Resolution**:
   - Load `hero_mapping.json` (maps ID string -> Name string).
   - Create a reverse mapping (Name -> ID). Case-insensitive matching is recommended for UX.
   - If the hero is not found, print an error and exit.
3. **Model Loading**:
   - Use `load_frozen_embeddings` from `dota2drafter.embeddings.pretrainer`.
4. **Similarity Calculation**:
   - Extract the target hero's embedding vector.
   - Compute cosine similarities against all heroes in `embeddings.weight` using `torch.nn.functional.cosine_similarity`.
   - Sort descending, exclude the query hero itself.
5. **Output Formatting**:
   - Print the top 5 closest heroes and their similarity scores using a formatted `rich` output.

## 2. RGCN Test Script (`scripts/06_test_rgcn.py`)
**Purpose**: Prints the top 5 synergies, top 5 heroes best against, and top 5 heroes to ban for a given hero based on the raw relation edges of the RGCN graph.

### Workflow
1. **Argument Parsing**: Accept `--hero_name`, `--data_dir` (default `data`), `--mapping_file` (default `hero_mapping.json`), and `--num_heroes` (default 124).
2. **Hero Resolution**: Same as Script 1.
3. **Graph Building**:
   - Initialize `DataExtractor(num_heroes=num_heroes)`.
   - Load batches: `batches = extractor.load_batches(data_dir)`.
   - Build graph: `hero_graph = extractor.build_hero_graph(batches)`.
4. **Edge Extraction & Ranking**:
   - For **5 Best Synergy** (`SYNERGY` edge type 0): Filter `edge_index` where `source == hero_id` and `edge_type == 0`. Sort by `edge_weight` descending. Take the top 5 target node IDs.
   - For **5 Heroes Best Against** (`ANTAGONIST` edge type 1): Filter `edge_index` where `source == hero_id` and `edge_type == 1`. Sort by `edge_weight` descending. Take the top 5 target node IDs.
   - For **5 Things to Ban** (`BANNED_AGAINST` edge type 2): Filter `edge_index` where `source == hero_id` and `edge_type == 2`. Sort by `edge_weight` descending. Take the top 5 target node IDs.
5. **Output Formatting**:
   - For each category, map the target IDs back to hero names using the parsed mapping.
   - Print the top 5 heroes and their corresponding edge weights/scores using `rich` console.

## Error Handling
- Invalid `hero_name`: Gracefully exit with a message listing acceptable names or indicating the name was not found.
- Missing files (`models/skip_gram_dgi.pt`, `data/`, `hero_mapping.json`): Exit gracefully with instructions on how to train or obtain the data.

## Files to Create/Modify
- `scripts/05_test_skip_gram.py` (Create)
- `scripts/06_test_rgcn.py` (Create)
