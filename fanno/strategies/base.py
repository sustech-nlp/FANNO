from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence


class BaseStrategy(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


class SeedGenerationStrategy(BaseStrategy):
    @abstractmethod
    def generate(self, docs: Sequence[str]) -> List[Dict[str, Any]]:
        ...


class InstructionAugmentationStrategy(BaseStrategy):
    @abstractmethod
    def generate(self, docs: Sequence[str], seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...


class ThinkDifferentStrategy(BaseStrategy):
    @abstractmethod
    def build_prompts(self, docs: Sequence[str], seeds: List[Dict[str, Any]]) -> List[str]:
        ...


class InstructionQualityStrategy(BaseStrategy):
    @abstractmethod
    def evaluate(self, new_data: List[Dict[str, Any]], old_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...


class ResponseStrategy(BaseStrategy):
    @abstractmethod
    def generate(self, instructions: Sequence[str]) -> List[List[str]]:
        ...
