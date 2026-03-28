from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import os
import random
import yaml

PROJECT_ROOT = Path(os.getenv("FANNO_HOME", Path.cwd()))
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_suffix(".yaml")

random.seed(42)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class InferenceConfig:
    """Inference and generation related settings."""

    model_name_or_path: str = os.getenv("FANNO_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    backend: str = os.getenv("FANNO_BACKEND", "vllm")  # ["vllm", "azure"]
    azure_tenant_id: str = os.getenv("FANNO_AZURE_TENANT", "72f988bf-86f1-41af-91ab-2d7cd011db47")
    azure_api_version: str = os.getenv("FANNO_AZURE_API_VERSION", "2024-12-01-preview")
    azure_max_retries: int = int(os.getenv("FANNO_AZURE_MAX_RETRIES", 5))
    tensor_parallel_size: int = int(os.getenv("FANNO_TP_SIZE", 1))
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 1024
    max_model_len: Optional[int] = None
    stop: Optional[Sequence[str]] = None
    skip_special_tokens: bool = True
    gpu_memory_utilization: float = float(os.getenv("FANNO_GPU_UTIL", 0.9))
    dtype: str = os.getenv("FANNO_DTYPE", "auto")
    seed: int = 42


@dataclass
class PipelineConfig:
    """High level pipeline knobs."""

    seed_docs_num: int = 50
    window_size: int = 500
    limit_size: int = 5000
    diversity_samples: int = 3
    seed_gen_strategy: str = "tagging"
    ins_aug_strategy: str = "ucb"
    instruction_quality_strategy: str = "combined"
    response_strategy: str = "basic"  # ["basic","temperature_sweep","dynamic_temperature","aspect","iterative"]
    think_diff_strategy: str = "ucb"  # ["ucb","random"]


@dataclass
class FileConfig:
    """All I/O related paths."""

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    unlabeled_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "unlabel_data.jsonl")
    com_unlabeled_data_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "unlabel_data_com.jsonl")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    run_name: str = os.getenv("FANNO_RUN_NAME", "fanno-run")

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    @property
    def seed_path(self) -> Path:
        return self.run_dir / "initial_seed.jsonl"

    @property
    def final_data_path(self) -> Path:
        return self.run_dir / "final_data.jsonl"


@dataclass
class EvaluatorConfig:
    """Filtering and scoring settings."""

    min_community_size: int = 1
    threshold: float = 0.8
    words_num: int = 0  # 0 = use full instruction text (was 4, causing coarse diversity filtering)
    device: str = os.getenv("FANNO_DEVICE", "cuda")
    batch_size: int = 64
    encode_model_path: str = os.getenv("FANNO_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    enable_llm_filter: bool = False


@dataclass
class MetricsConfig:
    """Configuration for instruction value metrics."""

    perplexity_model: Optional[str] = None  # falls back to inference model when None
    max_ppl_tokens: int = 1024
    ifd_prompt_temperature: float = 0.3
    ifd_scale: float = 5.0  # expected max score from the scorer prompt


@dataclass
class FannoConfig:
    """Root config object combining all sub-configs."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    files: FileConfig = field(default_factory=FileConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "FannoConfig":
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            return cls()

        with config_path.open("r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        def _update(dataclass_obj, values: Dict[str, Any]):
            for key, value in values.items():
                if hasattr(dataclass_obj, key):
                    setattr(dataclass_obj, key, value)
            return dataclass_obj

        cfg = cls()
        if "inference" in raw:
            cfg.inference = _update(cfg.inference, raw["inference"])
        if "pipeline" in raw:
            cfg.pipeline = _update(cfg.pipeline, raw["pipeline"])
        if "files" in raw:
            file_vals = raw["files"]
            if "data_dir" in file_vals:
                file_vals["data_dir"] = Path(file_vals["data_dir"])
            if "unlabeled_data_path" in file_vals:
                file_vals["unlabeled_data_path"] = Path(file_vals["unlabeled_data_path"])
            if "com_unlabeled_data_path" in file_vals:
                file_vals["com_unlabeled_data_path"] = Path(file_vals["com_unlabeled_data_path"])
            if "output_dir" in file_vals:
                file_vals["output_dir"] = Path(file_vals["output_dir"])
            cfg.files = _update(cfg.files, file_vals)
        if "evaluator" in raw:
            cfg.evaluator = _update(cfg.evaluator, raw["evaluator"])
        if "metrics" in raw:
            cfg.metrics = _update(cfg.metrics, raw["metrics"])
        return cfg


__all__ = [
    "FannoConfig",
    "InferenceConfig",
    "PipelineConfig",
    "FileConfig",
    "EvaluatorConfig",
    "MetricsConfig",
]
