# Dota 2 Draft Ingestion Pipeline

A Python-based data ingestion pipeline that fetches, validates, and transforms
Dota 2 Captains Mode draft sequences into PyTorch-ready tensor datasets.

## Setup

1. Copy `.env.example` to `.env` and fill in your STRATZ API key:
   ```bash
   cp .env.example .env
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

3. Configure the pipeline in `config.yaml` (target patch, output directory, etc.)

## Usage

Run the pipeline:
```bash
dota2-drafter
```

## Output

The pipeline generates `.pt` files in the configured output directory.
Each file contains a batch of processed matches with:
- `x`: (N, 24, 3) tensor of draft sequences
- `y`: (N,) tensor of radiant_win labels
- `match_ids`: list of match IDs for traceability
