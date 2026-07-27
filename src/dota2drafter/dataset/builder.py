"""Dataset builder - aggregates processed matches and saves as .pt files."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from dota2drafter.config import OutputConfig
from dota2drafter.processor.tensor_transformer import ProcessedMatch

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Aggregates processed matches and saves them as PyTorch .pt files."""

    def __init__(self, config: OutputConfig) -> None:
        self._config = config
        self._output_dir = Path(config.directory)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[ProcessedMatch] = []
        self._batch_count: int = 0

    def add(self, processed: ProcessedMatch) -> None:
        """Add a processed match to the buffer."""
        self._buffer.append(processed)

        if len(self._buffer) >= self._config.chunk_size:
            self._flush()

    def flush(self) -> None:
        """Flush any remaining matches in the buffer."""
        if self._buffer:
            self._flush()

    def _flush(self) -> None:
        """Write the current buffer to a .pt file."""
        if not self._buffer:
            return

        match_ids = [m.match_id for m in self._buffer]
        x_tensors = [m.x_tensor for m in self._buffer]
        y_tensors = [m.y_tensor for m in self._buffer]

        # Stack into batched tensors
        x_batch = torch.stack(x_tensors)  # (N, 24, 3)
        y_batch = torch.cat(y_tensors)  # (N,)

        dataset: dict[str, Any] = {
            "x": x_batch,
            "y": y_batch,
            "match_ids": match_ids,
        }

        self._batch_count += 1
        output_path = self._output_dir / f"drafts_batch_{self._batch_count:05d}.pt"
        torch.save(dataset, str(output_path))

        logger.info(
            "Saved batch %d: %d matches -> %s",
            self._batch_count,
            len(self._buffer),
            output_path,
        )

        self._buffer.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get dataset builder statistics."""
        return {
            "batches_saved": self._batch_count,
            "buffer_size": len(self._buffer),
            "output_dir": str(self._output_dir),
        }
