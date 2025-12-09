from __future__ import annotations

from typing import List, Sequence

from fanno.config import FannoConfig
from fanno.strategies.base import ResponseStrategy
from fanno.strategies.diversity import (
    AspectDiversityStrategy,
    IterativeBoostStrategy,
    TemperatureDiversityStrategy,
)
from fanno.template.response_template import q2a
from fanno.inference import run_inference


class BasicResponseStrategy(ResponseStrategy):
    """Single response generation."""

    def __init__(self, config: FannoConfig) -> None:
        super().__init__(name="basic_response")
        self.config = config

    def generate(self, instructions: Sequence[str]) -> List[List[str]]:
        prompts = [q2a(instr) for instr in instructions]
        responses = run_inference(prompts, config=self.config.inference, template_type="direct")
        return [[resp] for resp in responses]


class DiversityResponseStrategy(ResponseStrategy):
    """Wraps DiversityBench-inspired response samplers."""

    def __init__(self, config: FannoConfig, strategy_type: str) -> None:
        super().__init__(name=f"diversity_response:{strategy_type}")
        self.config = config
        self.strategy_type = strategy_type
        self._temperature_strategy = TemperatureDiversityStrategy(config.inference)
        self._aspect_strategy = AspectDiversityStrategy(config.inference)
        self._iterative_strategy = IterativeBoostStrategy(config.inference)

    def generate(self, instructions: Sequence[str]) -> List[List[str]]:
        if self.strategy_type == "temperature_sweep":
            return self._temperature_strategy.generate_temperature_sweep(instructions)
        if self.strategy_type == "dynamic_temperature":
            return self._temperature_strategy.generate_dynamic_temperature(instructions)
        if self.strategy_type == "aspect":
            return self._aspect_strategy.generate(instructions)
        if self.strategy_type == "iterative":
            return self._iterative_strategy.generate_iterative_responses(instructions)
        raise ValueError(f"Unknown response strategy: {self.strategy_type}")


def build_response_strategy(config: FannoConfig) -> ResponseStrategy:
    strategy = config.pipeline.response_strategy
    if strategy in {"none", "basic"}:
        return BasicResponseStrategy(config)
    return DiversityResponseStrategy(config, strategy_type=strategy)


__all__ = ["BasicResponseStrategy", "DiversityResponseStrategy", "build_response_strategy"]
