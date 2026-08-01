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
| `player_input_dim` | `127` | Number of input features per player comfort vector (defaults to 127 to match total hero count) |
| `h_gnn` | `None` | Frozen RGCN hero embeddings of shape `(K+1, d_model)` |

## TrainingConfig

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `1e-4` | Learning rate for the AdamW optimizer (with CosineAnnealingLR and 1e-2 weight_decay) |
| `lr_backbone` | `None` | Optional lower learning rate applied to the pre-trained Transformer backbone (e.g. `1e-5`) |
| `lr_head` | `None` | Optional standard learning rate applied to the linear head (e.g. `1e-3`) |
| `step_loss_gamma` | `0.0` | Gamma power parameter for scaling classification loss based on draft completeness (t/24)^gamma |
| `num_epochs` | `50` | Maximum number of training epochs |
| `batch_size` | `64` | Batch size for training and validation |
| `val_split` | `0.2` | Fraction of data reserved for validation |
| `device` | `"cpu"` | Device to train on (`"cpu"` or `"cuda"`) |
| `checkpoint_dir` | `"./checkpoints"` | Directory for saving best model checkpoints |
| `patience` | `10` | Early stopping patience |
| `min_delta` | `1e-4` | Minimum change to qualify as an improvement |
| `label_smoothing_eps` | `0.15` | Label smoothing epsilon value (configurable via CLI as `--label_smoothing_eps`) |

## TrainingMetrics

Tracks training and validation metrics across epochs:

| Field | Type | Description |
|---|---|---|
| `train_losses` | `list[float]` | Per-epoch training loss |
| `val_losses` | `list[float]` | Per-epoch validation loss |
| `val_accuracies` | `list[float]` | Per-epoch validation accuracy |
| `val_auc_scores` | `list[float]` | Per-epoch validation ROC-AUC |
| `best_epoch` | `int` | Epoch with best validation loss |
| `best_roc_auc` | `float` | Best validation loss value |

## PlayerComfortDataset

PyTorch Dataset that yields `(x_draft, player_matrices, y)` from raw data. Dynamically looks up historical player matrices given account IDs.

| Parameter | Default | Description |
|---|---|---|
| `x_drafts` | *(required)* | List of draft sequence tensors, each `(24, 4)` |
| `y_labels` | *(required)* | List of label tensors, each `(1,)` |
| `radiant_players` | *(required)* | List of Radiant player account ID lists (5 IDs each) |
| `dire_players` | *(required)* | List of Dire player account ID lists (5 IDs each) |
| `player_comfort_map` | `None` | Optional mapping of `account_id -> comfort tensor (10, C)` |
| `player_input_dim` | `127` | C, number of features per player comfort vector (defaults to 127 to match total hero count) |
| `augment` | `False` | If True, applies fused permutation (64 combinations) and prefix truncation (6 stages) on top of each other, yielding a 448x dataset expansion for robust generalization |
