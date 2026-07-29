# dota2drafter.embeddings

Embedding pre-training and RGCN training parameters.

## train_embeddings (Skip-Gram + DGI)

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Directory containing draft batch `.pt` files |
| `output_file` | *(required)* | Path to save the final embedding weights |
| `embed_dim` | `64` | Dimension of the embedding space |
| `skip_gram_epochs` | `10` | Number of Skip-Gram training epochs |
| `dgi_epochs` | `20` | Number of DGI training epochs |
| `skip_gram_lr` | `1e-2` | Learning rate for the Skip-Gram optimizer |
| `dgi_lr` | `1e-2` | Learning rate for the DGI optimizer |
| `batch_size` | `256` | Batch size for the Skip-Gram DataLoader |
| `device` | `None` (auto) | Device to train on (`"cpu"` or `"cuda"`) |

Returns the path to the saved embedding weights. The output tensor is padded at index 0 (heroes are 1-indexed).

## train_rgcn (HeroRGCN)

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Directory containing draft batch `.pt` files |
| `frozen_embeddings_path` | *(required)* | Path to saved DGI/Skip-Gram embeddings (`.pt` tensor) |
| `output_file` | *(required)* | Path to save the trained RGCN model weights |
| `d_model` | `64` | Embedding dimension |
| `num_relations` | `3` | Number of edge types in the multi-relational hero graph |
| `rgcn_epochs` | `20` | Number of RGCN training epochs |
| `learning_rate` | `1e-2` | Learning rate for the optimizer |
| `device` | `None` (auto) | Device to train on (`"cpu"` or `"cuda"`) |
| `hidden_dim` | `None` (== d_model) | Hidden dimension for RGCN layers |
| `num_layers` | `2` | Number of RGCN layers (1-2 recommended) |

The function trains the HeroRGCN by maximizing mutual information between node embeddings and the global graph summary.

## Edge Types

| Constant | Value | Description |
|---|---|---|
| `SYNERGY` | `0` | Co-picked-Radiant / Co-picked-Dire (undirected) |
| `ANTAGONIST` | `1` | Mechanical counter-picks (directed) |
| `BANNED_AGAINST` | `2` | Banned-Against correlations (directed) |
