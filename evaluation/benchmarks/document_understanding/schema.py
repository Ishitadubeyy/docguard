"""Evaluation result schema for document understanding benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class EvaluationResult:
    """Single metric result for a document understanding evaluation run."""

    model: str
    dataset: str
    task: str
    metric: str
    score: float | None
    latency_seconds: float | None = None
    memory_mb: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSample:
    """Ground-truth sample for document understanding evaluation."""

    sample_id: str
    document_type: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)
    tables: list[Any] = field(default_factory=list)
    corrected_text: str | None = None


@dataclass
class BenchmarkRunSummary:
    """Aggregated benchmark metrics for one model/dataset/task."""

    model: str
    dataset: str
    task: str
    results: list[EvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "task": self.task,
            "results": [result.to_dict() for result in self.results],
        }
