# Interactive Draft CLI Mockup

```text
================================================================================
                           DOTA 2 ADVERSARIAL DRAFTER
================================================================================
Team: RADIANT (First Pick)     Opponent: DIRE
--------------------------------------------------------------------------------
Step 01 / 24 | Action: BAN | Team: RADIANT
[?] Enter your ban (Hero ID or Name): 
```

_If MCTS is calculating:_

```text
Step 01 / 24 | Action: BAN | Team: RADIANT
[⏳] Calculating MCTS recommendations for 30s... (3402 nodes expanded)

⭐ RECOMMENDED ACTIONS
╭──────────────┬────────┬──────────┬────────────┬─────────────╮
│ Hero         │ Action │ Win Prob │ Prior Prob │ Visit Count │
├──────────────┼────────┼──────────┼────────────┼─────────────┤
│ 103 (Chen)   │ BAN    │  54.2%   │   0.051    │    1240     │
│  89 (Naga)   │ BAN    │  53.8%   │   0.048    │     980     │
│  15 (Razor)  │ BAN    │  53.1%   │   0.031    │     810     │
╰──────────────┴────────┴──────────┴────────────┴─────────────╯

📈 PRINCIPAL VARIATION (Expected Draft Plan)
  1. Radiant BAN Chen
  2. Dire    BAN Enigma
  3. Radiant BAN Naga Siren
  ...

[?] Select Hero to BAN (default: Chen):
```
