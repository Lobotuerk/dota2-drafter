# MCTS Node Evaluation & Backpropagation Flow - AUT-31

This layout document provides a Mermaid visualization of the draft state evaluation flow, detailing the uniform evaluation path of pick/ban states and how the neural network output propagates to inform the MCTS search.

```mermaid
graph TD
    %% Define styles
    classDef stateNode fill:#f9f,stroke:#333,stroke-width:2px;
    classDef nnNode fill:#bbf,stroke:#333,stroke-width:2px;
    classDef valNode fill:#bfb,stroke:#333,stroke-width:2px;

    subgraph "MCTS Search Engine"
        A[Adversarial Selection Phase] -->|Select Action| B(DraftState Node)
    end

    subgraph "Unified DraftState Evaluation"
        B -->|MCTS evaluate_batch| C{Terminal Check}
        C -->|t >= 24| D[Return value, empty priors []]
        C -->|t < 24| E[Build Unified Tensor: B, 24, 4]
        E --> F[Run Neural Network Forward Pass]
    end

    subgraph "Neural Network Inference"
        F --> G[Sigmoid Logits: win_prob]
        F --> H[MLM Logits: policy_logits]
    end

    subgraph "Reward & Prior Extraction"
        G -->|Active Team perspective| I[Value: radiant_win or 1.0 - radiant_win]
        H -->|Extract mlm_logits for step_idx| J[Raw Policy Logits]
        J -->|Apply used/invalid hero mask| K[Masked Logits]
        K -->|Filter to top K candidates| L[Top-K Masked Logits]
        L -->|Compute Softmax| M[Priors Distribution]
    end

    I --> N[Return Value and Priors to MCTS]
    M --> N
    N -->|Backpropagate Reward| O[Update Q-value and Visited Counts]

    class B,C stateNode;
    class F,G,H nnNode;
    class I,M,N valNode;
```
