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
class ModelConfig:
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 128
    num_layers_rgcn: int = 2
    num_layers_transformer: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    step_loss_gamma: float = 0.0
    label_smoothing_eps: float = 0.15
    augment: int | str | bool = 0
    batch_size: int = 16
    skip_gram_lr: float = 1e-2
    dgi_lr: float = 1e-2
    rgcn_lr: float = 1.5e-3
    checkpoint_metric: str = "val_auc"


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
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig())


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
        max_connections_per_host=data.get(
            "max_connections_per_host", config.concurrency.max_connections_per_host
        ),
        rate_limit_per_second=data.get(
            "rate_limit_per_second", config.concurrency.rate_limit_per_second
        ),
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


def _load_model(data: dict[str, Any], config: PipelineConfig) -> ModelConfig:
    return ModelConfig(
        d_model=data.get("d_model", config.model.d_model),
        nhead=data.get("nhead", config.model.nhead),
        dim_feedforward=data.get("dim_feedforward", config.model.dim_feedforward),
        num_layers_rgcn=data.get("num_layers_rgcn", config.model.num_layers_rgcn),
        num_layers_transformer=data.get(
            "num_layers_transformer", config.model.num_layers_transformer
        ),
        dropout=data.get("dropout", config.model.dropout),
    )


def _load_training(data: dict[str, Any], config: PipelineConfig) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=data.get("learning_rate", config.training.learning_rate),
        step_loss_gamma=data.get("step_loss_gamma", config.training.step_loss_gamma),
        label_smoothing_eps=data.get("label_smoothing_eps", config.training.label_smoothing_eps),
        augment=data.get("augment", config.training.augment),
        batch_size=data.get("batch_size", config.training.batch_size),
        skip_gram_lr=data.get("skip_gram_lr", config.training.skip_gram_lr),
        dgi_lr=data.get("dgi_lr", config.training.dgi_lr),
        rgcn_lr=data.get("rgcn_lr", config.training.rgcn_lr),
        checkpoint_metric=data.get("checkpoint_metric", config.training.checkpoint_metric),
    )


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    """Load pipeline configuration from a YAML file."""
    load_dotenv()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
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
        model=_load_model(raw.get("model", {}), base),
        training=_load_training(raw.get("training", {}), base),
    )
    return resolved
