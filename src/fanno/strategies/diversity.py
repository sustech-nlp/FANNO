from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, List, Sequence

from fanno.config import InferenceConfig
from fanno.inference import run_inference


# Aspect-based diversification -------------------------------------------------
@dataclass
class AspectStrategyConfig:
    num_responses: int = 5
    top_p: float = 0.95
    max_tokens: int = 1024


class AspectDiversityStrategy:
    """Generate diverse answers by first sampling angles, then expanding them."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        strategy_config: AspectStrategyConfig | None = None,
        inference_fn: Callable[..., List[str]] = run_inference,
    ) -> None:
        self.inference_config = inference_config
        self.config = strategy_config or AspectStrategyConfig()
        self._inference_fn = inference_fn

    def _angle_prompt(self, instruction: str) -> str:
        return (
            f"Generate {self.config.num_responses} different angles or perspectives to answer the following question. "
            "Each angle should be brief (1-2 sentences) and semantically distinct from others. "
            'Number them exactly like "1. ", "2. ", "3. " etc.\n\n'
            f"Question: {instruction}\n\nDifferent angles to answer:"
        )

    def _expansion_prompt(self, instruction: str, angle: str) -> str:
        return (
            "Based on the following angle, provide a comprehensive response to the question. "
            "Expand the angle into a full, detailed answer.\n\n"
            f"Question: {instruction}\n\nAngle: {angle}\n\nFull response:"
        )

    def _parse_angles(self, angles_text: str) -> List[str]:
        angles: List[str] = []
        pattern = r"(\d+)\.\s+(.*?)(?=\n\d+\.\s|$)"
        matches = re.findall(pattern, angles_text, re.DOTALL)
        for _, text in matches:
            cleaned = text.strip()
            if cleaned:
                angles.append(cleaned)

        if angles:
            return angles[: self.config.num_responses]

        # Fallback when numbering is missing
        lines = angles_text.strip().split("\n")
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r"^\d+\.\s+", line):
                if current:
                    angles.append(current.strip())
                current = re.sub(r"^\d+\.\s+", "", line)
            else:
                current += f" {line}"
        if current:
            angles.append(current.strip())
        return angles[: self.config.num_responses]

    def _cfg(self, *, temperature: float = 0.0, max_tokens: int | None = None) -> InferenceConfig:
        return replace(
            self.inference_config,
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
        )

    def generate(self, instructions: Sequence[str]) -> List[List[str]]:
        angle_prompts = [self._angle_prompt(instr) for instr in instructions]
        angle_texts = self._inference_fn(angle_prompts, config=self._cfg(temperature=0.2), template_type="direct")

        all_angles = [self._parse_angles(text) for text in angle_texts]

        expansion_prompts: List[str] = []
        prompt_owner: List[int] = []
        for idx, instruction in enumerate(instructions):
            for angle in all_angles[idx]:
                expansion_prompts.append(self._expansion_prompt(instruction, angle))
                prompt_owner.append(idx)

        expanded_responses = self._inference_fn(expansion_prompts, config=self._cfg(), template_type="direct")

        organized: List[List[str]] = [[] for _ in instructions]
        for response, owner in zip(expanded_responses, prompt_owner):
            organized[owner].append(response)
        return organized


# Temperature sampling diversification ----------------------------------------
@dataclass
class TemperatureStrategyConfig:
    num_samples: int = 5
    temperatures: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 5.0)
    top_p: float = 0.95
    max_tokens: int = 1024
    dynamic_high_temp: float = 2.5
    dynamic_low_temp: float = 0.5
    dynamic_prefix_tokens: int = 30


class TemperatureDiversityStrategy:
    """Generate diverse answers using fixed or dynamic temperature sweeps."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        strategy_config: TemperatureStrategyConfig | None = None,
        inference_fn: Callable[..., List[str]] = run_inference,
    ) -> None:
        self.inference_config = inference_config
        self.config = strategy_config or TemperatureStrategyConfig()
        self._inference_fn = inference_fn

    def _base_prompt(self, instruction: str) -> str:
        return f"Please provide a detailed response to the following question:\n\nQuestion: {instruction}\n\nResponse:"

    def _cfg(self, *, temperature: float, max_tokens: int | None = None) -> InferenceConfig:
        return replace(
            self.inference_config,
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
        )

    def generate_temperature_sweep(self, instructions: Sequence[str]) -> List[List[str]]:
        prompts = [self._base_prompt(instr) for instr in instructions]
        bucket: List[List[str]] = [[] for _ in instructions]

        temperatures = list(self.config.temperatures)
        if len(temperatures) < self.config.num_samples:
            temperatures.extend([temperatures[-1]] * (self.config.num_samples - len(temperatures)))

        for temp in temperatures[: self.config.num_samples]:
            responses = self._inference_fn(prompts, config=self._cfg(temperature=temp), template_type="direct")
            for idx, response in enumerate(responses):
                bucket[idx].append(response)
        return bucket

    def generate_dynamic_temperature(self, instructions: Sequence[str]) -> List[List[str]]:
        prompts = [self._base_prompt(instr) for instr in instructions]
        bucket: List[List[str]] = [[] for _ in instructions]

        for _ in range(self.config.num_samples):
            prefixes = self._inference_fn(
                prompts,
                config=self._cfg(
                    temperature=self.config.dynamic_high_temp,
                    max_tokens=self.config.dynamic_prefix_tokens,
                ),
                template_type="direct",
            )
            continuation_prompts = [f"{prompt}\n{prefix}" for prompt, prefix in zip(prompts, prefixes)]
            completions = self._inference_fn(
                continuation_prompts,
                config=self._cfg(
                    temperature=self.config.dynamic_low_temp,
                    max_tokens=self.config.max_tokens,
                ),
                template_type="direct",
            )
            for idx, completion in enumerate(completions):
                bucket[idx].append(completion)

        return bucket


# Iterative boost diversification ---------------------------------------------
@dataclass
class IterativeStrategyConfig:
    num_iterations: int = 5
    temperature: float = 0.0
    top_p: float = 0.95
    max_initial_tokens: int = 1024
    max_compress_tokens: int = 512
    max_enhance_tokens: int = 1024


class IterativeBoostStrategy:
    """Iteratively generates, compresses, diversifies, and enhances responses."""

    def __init__(
        self,
        inference_config: InferenceConfig,
        strategy_config: IterativeStrategyConfig | None = None,
        inference_fn: Callable[..., List[str]] = run_inference,
    ) -> None:
        self.inference_config = inference_config
        self.config = strategy_config or IterativeStrategyConfig()
        self._inference_fn = inference_fn

    def _initial_prompt(self, instruction: str) -> str:
        return f"Please provide a detailed response to the following question:\n\nQuestion: {instruction}\n\nResponse:"

    def _compress_prompt(self, instruction: str, response: str) -> str:
        return (
            "Compress the following response to the given question, preserving only the key information. "
            "Keep it concise but maintain the core concepts.\n\n"
            f"Question: {instruction}\n\nOriginal response: {response}\n\nCompressed response:"
        )

    def _diverse_prompt(self, instruction: str, previous_compressed: Sequence[str], iteration: int) -> str:
        context = "\n".join(f"Response {idx + 1}: {resp}" for idx, resp in enumerate(previous_compressed))
        return (
            f"Generate a completely different response (Response #{iteration + 1}) to the question below.\n"
            "Your response should have semantically distinct content from ALL previous responses in the context.\n"
            "Aim for a unique perspective, approach, or framework that hasn't been covered yet.\n\n"
            f"Question: {instruction}\n\nPrevious responses to avoid or differentiate from:\n{context}\n\n"
            f"New diverse response #{iteration + 1}:"
        )

    def _enhancement_prompt(self, instruction: str, diverse_response: str) -> str:
        return (
            "Enhance the following response to the question. Identify areas for improvement and make the response more "
            "detailed, informative, and comprehensive. Build upon the existing perspective.\n\n"
            f"Question: {instruction}\n\nCurrent response: {diverse_response}\n\nEnhanced response:"
        )

    def _cfg(self, *, temperature: float | None = None, max_tokens: int | None = None) -> InferenceConfig:
        return replace(
            self.inference_config,
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_initial_tokens,
        )

    def _run(self, prompts: Sequence[str], *, max_tokens: int | None = None) -> List[str]:
        return self._inference_fn(prompts, config=self._cfg(max_tokens=max_tokens), template_type="direct")

    def generate_iterative_responses(self, instructions: Sequence[str]) -> List[List[str]]:
        if not instructions:
            return []

        all_responses: List[List[str]] = [[] for _ in instructions]
        compressed_history: List[List[str]] = [[] for _ in instructions]

        initial_prompts = [self._initial_prompt(instr) for instr in instructions]
        initial_responses = self._run(initial_prompts, max_tokens=self.config.max_initial_tokens)
        for idx, response in enumerate(initial_responses):
            all_responses[idx].append(response)

        initial_compressed = self._run(
            [self._compress_prompt(instructions[idx], response) for idx, response in enumerate(initial_responses)],
            max_tokens=self.config.max_compress_tokens,
        )
        for idx, compressed in enumerate(initial_compressed):
            compressed_history[idx].append(compressed)

        for iteration in range(1, self.config.num_iterations):
            diverse_responses = self._run(
                [self._diverse_prompt(instructions[idx], compressed_history[idx], iteration) for idx in range(len(instructions))]
            )
            enhanced_responses = self._run(
                [self._enhancement_prompt(instructions[idx], diverse_responses[idx]) for idx in range(len(instructions))],
                max_tokens=self.config.max_enhance_tokens,
            )

            for idx, enhanced in enumerate(enhanced_responses):
                all_responses[idx].append(enhanced)

            if iteration < self.config.num_iterations - 1:
                compressed_batch = self._run(
                    [self._compress_prompt(instructions[idx], enhanced_responses[idx]) for idx in range(len(instructions))],
                    max_tokens=self.config.max_compress_tokens,
                )
                for idx, compressed in enumerate(compressed_batch):
                    compressed_history[idx].append(compressed)

        return all_responses


__all__ = [
    "AspectDiversityStrategy",
    "AspectStrategyConfig",
    "TemperatureDiversityStrategy",
    "TemperatureStrategyConfig",
    "IterativeBoostStrategy",
    "IterativeStrategyConfig",
]
