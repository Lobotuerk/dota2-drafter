# dota2drafter

Ingestion pipeline configuration parameters.

## PipelineConfig

| Parameter | Default | Description |
|---|---|---|
| `patch` | `"7.35"` | Dota patch version to filter matches |
| `tiers` | `[1, 2]` | Tournament tiers to include |

## StratzConfig

| Parameter | Default | Description |
|---|---|---|
| `api_key` | *(required)* | STRATZ API key (supports `${VAR}` env var expansion) |
| `base_url` | `"https://api.stratz.com/v1"` | STRATZ API base URL |
| `max_retries` | `5` | Retry attempts for API failures |
| `retry_delay` | `2.0` | Delay between retries (seconds) |

## OpenDotaConfig

| Parameter | Default | Description |
|---|---|---|
| `base_url` | `"https://api.opendota.com/api"` | OpenDota API base URL |
| `max_retries` | `5` | Retry attempts for API failures |
| `retry_delay` | `2.0` | Delay between retries (seconds) |

## ConcurrencyConfig

| Parameter | Default | Description |
|---|---|---|
| `max_workers` | `10` | Maximum concurrent workers |
| `max_connections_per_host` | `5` | Max connections per host |
| `rate_limit_per_second` | `5` | Rate limit per second |

## OutputConfig

| Parameter | Default | Description |
|---|---|---|
| `directory` | `"./data"` | Where `.pt` batch files are saved |
| `chunk_size` | `1000` | Number of matches per `.pt` file |

## StateConfig

| Parameter | Default | Description |
|---|---|---|
| `database_path` | `"./state.db"` | SQLite database for tracking progress |
