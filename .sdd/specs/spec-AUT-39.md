# Technical Specification: AW-MLM Implementation (AUT-39)

## 1. Overview
This specification details the implementation of the Advantage-Weighted Masked Language Modeling (AW-MLM) objective for the `dota2-drafter` project. The current standard Masked Language Modeling (MLM) loss function in `transformer_trainer.py` will be upgraded to weight the cross-entropy elements by an advantage factor $w_{b,t}$, derived from the predicted win probability (the "frozen value network" proxy). This adapts the gradient landscape to penalize blunders while aggressively optimizing for high-advantage drafting behavior.

## 2. Configuration Updates
In `src/dota2drafter/training/transformer_trainer.py` (and/or `config.py`), update `TrainingConfig` to include the hyperparameters for the AW-MLM temperature annealing and clipping.

```python
    # AW-MLM Parameters
    aw_tau_start: float = 0.15
    aw_tau_end: float = 0.08
    aw_tau_decay_epochs: int = 50
    aw_clip_min: float = 0.1
    aw_clip_max: float = 10.0
```

## 3. Dynamic Advantage Calculation
In the `train()` method of `transformer_trainer.py` (`src/dota2drafter/training/transformer_trainer.py`), we must calculate the signed marginal advantage for every step in the sequence before computing the MLM loss. 

To achieve this efficiently without modifying the `MatchNetwork` architecture, we will leverage PyTorch's `expand` and `reshape` to compute the value of all prefix subsets of the current batch in a single forward pass through the frozen `SetTransformerHead`.

### Algorithm
1. Inside the training batch loop, use a `with torch.no_grad():` block to compute advantages.
2. For a batch `x_batch` of shape `(B, seq_len, 4)`, clone and expand it to create a prefix mask for each step $t \in [0, 23]$.
3. For a given prefix slice $t$, mask out all subsequent steps ($> t$) by setting their hero index (column 2) to `-1.0`.
4. Reshape this prefix tensor to `(B * seq_len, seq_len, 4)` and pass it through the joint embedding and `set_transformer_head`.
5. Apply the `sigmoid` activation to get the win probability $V(s_t)$ for the Radiant team. Reshape back to `(B, seq_len)`.
6. Compute the marginal advantage $V(s_t) - V(s_{t-1})$. For $t=0$, use a prior probability of `0.5` for $V(s_{-1})$.
7. Sign the advantage based on the acting team: if the team at step $t$ is Dire (`1.0`), multiply the advantage by `-1` (since Dire's advantage is the decrease in Radiant's win probability).

## 4. Temperature Annealing & Weight Generation
Outside the batch loop, calculate the current AW-MLM temperature $\tau$ based on the epoch, similar to the existing slot attention tau annealing:
```python
        aw_tau_decay_rate = (self.config.aw_tau_end / self.config.aw_tau_start) ** (
            1.0 / self.config.aw_tau_decay_epochs
        )
```
Inside the batch loop:
```python
        current_aw_tau = self.config.aw_tau_start * (
            aw_tau_decay_rate ** min(epoch - 1, self.config.aw_tau_decay_epochs)
        )
```

Convert the advantages $A_{b,t}$ to weights $w_{b,t}$:
```python
        w_t = torch.exp(A_t / current_aw_tau)
        w_t = torch.clamp(w_t, min=self.config.aw_clip_min, max=self.config.aw_clip_max)
```

## 5. Replacing the MLM Objective
Locate the existing Step-Weighted Cross Entropy logic in `transformer_trainer.py`:
```python
        # Step-Weighted Cross Entropy:
        # w_t = 0.5 + 1.0 * (t / 23) => 0.5 at Step 0, 1.5 at Last Pick
        seq_len = mlm_logits.size(1)
        t_idx = torch.arange(seq_len, device=self.device).float()
        step_weights = 0.5 + 1.0 * (t_idx / max(1.0, float(seq_len - 1)))
```

Replace `step_weights.unsqueeze(0)` with the dynamically computed `w_t` (which already has shape `(B, seq_len)`).

```python
        # AW-MLM Loss:
        weighted_ce = ce_elements * w_t
        valid_elements = (ntp_labels != -1).sum().float()
        mlm_loss = weighted_ce.sum() / torch.clamp(valid_elements, min=1.0)
```

## 6. Implementation Notes
- Ensure all advantage calculations are strictly within `torch.no_grad()` to prevent memory leaks and gradient bleeding into the value network during the MLM optimization phase.
- The `valid_elements` and `ntp_labels != -1` masks correctly handle any padded sequences at the end of partial drafts.
- Use `self.model.match_network.joint_embedding.get_pure_hero_embeddings` with `for_value=True` to get the correct inputs for the `set_transformer_head`.
