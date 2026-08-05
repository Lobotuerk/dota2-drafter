# MCTS Tree Graph Layout

```mermaid
graph TD
    A[Root: Current State] -->|Pick Radiant| B(Child: Hero X)
    A -->|Ban Dire| C(Child: Hero Y)
    A -->|Pick Radiant| D(Child: Hero Z)
    
    B -->|Visits: 300, Prob: 55%| E[...]
    C -->|Visits: 10, Prob: 45%| F[...]
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style B fill:#bbf,stroke:#333,stroke-width:2px
```

*Note: This task is purely backend algorithmic optimization. This diagram illustrates the memory structure of the MCTS tree being reused across turns, rather than a UI layout.*