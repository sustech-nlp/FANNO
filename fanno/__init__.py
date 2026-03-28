"""FANNO: Free ANNOtator pipeline for synthetic instruction generation."""

from fanno.config import (
    EvaluatorConfig,
    FannoConfig,
    FileConfig,
    InferenceConfig,
    MetricsConfig,
    PipelineConfig,
)

__version__ = "0.2.0"


def __getattr__(name: str):
    """Lazy imports for heavy modules that depend on vllm/torch."""
    if name == "FannoPipeline":
        from fanno.pipeline import FannoPipeline
        return FannoPipeline
    if name == "run_pipeline":
        from fanno.pipeline import run_pipeline
        return run_pipeline
    raise AttributeError(f"module 'fanno' has no attribute {name!r}")


__all__ = [
    "FannoConfig",
    "InferenceConfig",
    "PipelineConfig",
    "FileConfig",
    "EvaluatorConfig",
    "MetricsConfig",
    "FannoPipeline",
    "run_pipeline",
]
