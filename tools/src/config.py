import os
from dataclasses import dataclass
from typing import Optional

DEFAULT_INPUT_PATH = "/home/v-hezhu2/FANNO-Tool-Dev/data/unlabel_data.jsonl"
DEFAULT_OUTPUT_PATH = "/home/v-hezhu2/FANNO-Tool-Dev/synthetic_data.jsonl"
DEFAULT_TARGET_CONVERSATIONS = 1000
DEFAULT_MAX_TURNS = 10
DEFAULT_MIN_QUALITY_SCORE = 7
DEFAULT_MODEL = os.environ.get("STORY_MODEL", "gpt-4o")
DEFAULT_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "72f988bf-86f1-41af-91ab-2d7cd011db47")
DEFAULT_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")
DEFAULT_MAX_RETRIES = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_COMPLETION_TOKENS = 2000
DEFAULT_PARALLEL_COMPLETION_TOKENS = 512
DEFAULT_INFERENCE_WORKERS = 8

ALLOWED_ROLES = {"human", "gpt", "function_call", "observation"}
LOGIC_PATTERNS = {
    "smooth": "The conversation completes smoothly with all tool calls succeeding",
    "partial_failure": "Encounters partial failures requiring additional user information or alternative solutions",
    "error_recovery": "Encounters errors and the assistant tries alternative approaches",
    "escalation": "The issue cannot be resolved and eventually transfers to human agent",
    "user_change_mind": "User changes requirements midway and assistant adapts strategy",
    "multi_goal": "User has multiple goals that need to be completed step by step",
}


@dataclass
class ScenarioConfig:
    num_tools: Optional[int] = None
    num_turns: Optional[int] = None
    logic_pattern: Optional[str] = None
    domain_hint: Optional[str] = None


@dataclass
class DatasetConfig:
    input_path: str = DEFAULT_INPUT_PATH
    output_path: str = DEFAULT_OUTPUT_PATH
    target_conversations: int = DEFAULT_TARGET_CONVERSATIONS
    max_turns: int = DEFAULT_MAX_TURNS
    min_quality_score: int = DEFAULT_MIN_QUALITY_SCORE
    seed: Optional[int] = None
    model: str = DEFAULT_MODEL


@dataclass
class InferenceConfig:
    model: str = DEFAULT_MODEL
    tenant_id: str = DEFAULT_TENANT_ID
    api_version: str = DEFAULT_API_VERSION
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    parallel_max_tokens: int = DEFAULT_PARALLEL_COMPLETION_TOKENS
    workers: int = DEFAULT_INFERENCE_WORKERS


@dataclass(frozen=True)
class LLMCallParams:
    temperature: float
    max_tokens: Optional[int] = None


LLM_CALL_PARAMS = {
    "scenario_generation": LLMCallParams(temperature=0.8),
    "quality_scenario": LLMCallParams(temperature=0.1),
    "quality_conversation": LLMCallParams(temperature=0.3),
    "world_model": LLMCallParams(temperature=0.7),
    "user_simulator": LLMCallParams(temperature=0.9),
    "initial_query": LLMCallParams(temperature=0.9),
    "decide_action": LLMCallParams(temperature=0.7),
    "function_call": LLMCallParams(temperature=0.7),
    "gpt_response": LLMCallParams(temperature=0.8),
    "completion_check": LLMCallParams(temperature=0.3),
}


__all__ = [
    "ScenarioConfig",
    "DatasetConfig",
    "InferenceConfig",
    "LLMCallParams",
    "LLM_CALL_PARAMS",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_TARGET_CONVERSATIONS",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MIN_QUALITY_SCORE",
    "DEFAULT_MODEL",
    "DEFAULT_TENANT_ID",
    "DEFAULT_API_VERSION",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_COMPLETION_TOKENS",
    "DEFAULT_PARALLEL_COMPLETION_TOKENS",
    "DEFAULT_INFERENCE_WORKERS",
    "LOGIC_PATTERNS",
    "ALLOWED_ROLES",
]
