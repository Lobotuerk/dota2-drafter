# dota2drafter.training

Match Network and Transformer training parameters.

## MatchNetwork

| Parameter | Default | Description |
|---|---|---|
| `d_model` | `128` | Transformer embedding dimension |
| `nhead` | `4` | Number of attention heads |
| `num_layers` | `4` | Number of TransformerDecoderLayer blocks |
| `dim_feedforward` | `256` | Feedforward dimension in decoder layers |
| `dropout` | `0.1` | Dropout rate |
| `num_heroes` | `120` | Number of heroes K |
| `player_input_dim` | `10` | Number of input features per player comfort vector |
| `h_gnn` | `None` | Frozen RGCN hero embeddings of shape `(K+1, d_model)` |

## TrainingConfig

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `1e-3` | Learning rate for the Adam optimizer |
| `num_epochs` | `50` | Maximum number of training epochs |
| `batch_size` | `64` | Batch size for training and validation |
| `val_split` | `0.2` | Fraction of data reserved for validation |
| `device` | `"cpu"` | Device to train on (`"cpu"` or `"cuda"`) |
| `checkpoint_dir` | `"./checkpoints"` | Directory for saving best model checkpoints |
| `patience` | `10` | Early stopping patience |
| `min_delta` | `1e-4` | Minimum change to qualify as an improvement |

## TrainingMetrics

Tracks training and validation metrics across epochs:

| Field | Type | Description |
|---|---|---|
| `train_losses` | `list[float]` | Per-epoch training loss |
| `val_losses` | `list[float]` | Per-epoch validation loss |
| `val_accuracies` | `list[float]` | Per-epoch validation accuracy |
| `val_auc_scores` | `list[float]` | Per-epoch validation ROC-AUC |
| `best_epoch` | `int` | Epoch with best validation loss |
| `best_val_loss` | `float` | Best validation loss value |

## PlayerComfortDataset

PyTorch Dataset that yields `(x_draft, player_matrices, y)` from raw data. Dynamically looks up historical player matrices given account IDs.

| Parameter | Default | Description |
|---|---|---|
| `x_drafts` | *(required)* | List of draft sequence tensors, each `(24, 4)` |
| `y_labels` | *(required)* | List of label tensors, each `(1,)` |
| `radiant_players` | *(required)* | List of Radiant player account ID lists (5 IDs each) |
| `dire_players` | *(required)* | List of Dire player account ID lists (5 IDs each) |
| `player_comfort_map` | `None` | Optional mapping of `account_id -> comfort tensor (10, C)` |
| `player_input_dim` | `10` | C, number of features per player comfort vector |
