"""Configuration loader for the Dota 2 draft ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass
class StratzConfig:
    api_key: str
    base_url: str = "https://api.stratz.com/v1"
    max_retries: int = 5
    retry_delay: float = 2.0


@dataclass
class OpenDotaConfig:
    base_url: str = "https://api.opendota.com/api"
    max_retries: int = 5
    retry_delay: float = 2.0


@dataclass
class ConcurrencyConfig:
    max_workers: int = 10
    max_connections_per_host: int = 5
    rate_limit_per_second: int = 5


@dataclass
class OutputConfig:
    directory: str = "./data"
    chunk_size: int = 1000


@dataclass
class StateConfig:
    database_path: str = "./state.db"


@dataclass
class GraphConfig:
    wilson_threshold: float = 0.50
    gamma: float = 0.80


@dataclass
class PipelineConfig:
    cutoff_date: str = "2026-06-04"
    tiers: list[int] = field(default_factory=lambda: [1, 2])
    stratz: StratzConfig = field(default_factory=lambda: StratzConfig(api_key=""))
    opendota: OpenDotaConfig = field(default_factory=lambda: OpenDotaConfig())
    concurrency: ConcurrencyConfig = field(default_factory=lambda: ConcurrencyConfig())
    output: OutputConfig = field(default_factory=lambda: OutputConfig())
    state: StateConfig = field(default_factory=lambda: StateConfig())
    graph: GraphConfig = field(default_factory=lambda: GraphConfig())


def _resolve_env_vars(value: str) -> str:
    """Resolve ${VAR} patterns in a string value."""
    result = value
    var_name = result.split("${", 1)[1].split("}", 1)[0] if "${" in result else ""
    if var_name:
        result = os.environ.get(var_name, value)
    return result


def _load_stratz(data: dict[str, Any], config: PipelineConfig) -> StratzConfig:
    api_key = data.get("api_key", "")
    api_key = _resolve_env_vars(api_key)
    return StratzConfig(
        api_key=api_key,
        base_url=data.get("base_url", config.stratz.base_url),
        max_retries=data.get("max_retries", config.stratz.max_retries),
        retry_delay=data.get("retry_delay", config.stratz.retry_delay),
    )


def _load_opendota(data: dict[str, Any], config: PipelineConfig) -> OpenDotaConfig:
    return OpenDotaConfig(
        base_url=data.get("base_url", config.opendota.base_url),
        max_retries=data.get("max_retries", config.opendota.max_retries),
        retry_delay=data.get("retry_delay", config.opendota.retry_delay),
    )


def _load_concurrency(data: dict[str, Any], config: PipelineConfig) -> ConcurrencyConfig:
    return ConcurrencyConfig(
        max_workers=data.get("max_workers", config.concurrency.max_workers),
        max_connections_per_host=data.get("max_connections_per_host", config.concurrency.max_connections_per_host),
        rate_limit_per_second=data.get("rate_limit_per_second", config.concurrency.rate_limit_per_second),
    )


def _load_output(data: dict[str, Any], config: PipelineConfig) -> OutputConfig:
    return OutputConfig(
        directory=data.get("directory", config.output.directory),
        chunk_size=data.get("chunk_size", config.output.chunk_size),
    )


def _load_state(data: dict[str, Any], config: PipelineConfig) -> StateConfig:
    return StateConfig(
        database_path=data.get("database_path", config.state.database_path),
    )


def _load_graph(data: dict[str, Any], config: PipelineConfig) -> GraphConfig:
    return GraphConfig(
        wilson_threshold=data.get("wilson_threshold", config.graph.wilson_threshold),
        gamma=data.get("gamma", config.graph.gamma),
    )


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base = PipelineConfig()
    resolved = PipelineConfig(
        cutoff_date=raw.get("cutoff_date", base.cutoff_date),
        tiers=raw.get("tiers", base.tiers),
        stratz=_load_stratz(raw.get("stratz", {}), base),
        opendota=_load_opendota(raw.get("opendota", {}), base),
        concurrency=_load_concurrency(raw.get("concurrency", {}), base),
        output=_load_output(raw.get("output", {}), base),
        state=_load_state(raw.get("state", {}), base),
        graph=_load_graph(raw.get("graph", {}), base),
    )
    return resolved
