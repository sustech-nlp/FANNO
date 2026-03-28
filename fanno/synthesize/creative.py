"""Creative writing data synthesis."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from loguru import logger

from fanno.synthesize.base import BaseSynthesizer
from fanno.synthesize.prompts import (
    CREATIVE_INSTRUCTION_PROMPT,
    ANSWER_PROMPT,
)

WRITING_TYPES = [
    "a short story", "a persuasive essay", "a poem", "a technical tutorial",
    "a product review", "a news article", "a personal reflection",
    "a how-to guide", "a dialogue scene", "a letter of recommendation",
    "a research summary", "a speech", "a blog post", "an analysis",
    "a comparison essay", "a descriptive passage", "a debate argument",
]


class CreativeSynthesizer(BaseSynthesizer):
    """Synthesize creative writing instruction/response pairs."""

    def generate(self, num_samples: int = 500, **kwargs) -> List[Dict[str, Any]]:
        """Generate creative writing tasks and responses."""
        logger.info(f"Generating {num_samples} creative writing samples")

        # Step 1: Generate diverse writing instructions
        prompts = [
            CREATIVE_INSTRUCTION_PROMPT.format(type=random.choice(WRITING_TYPES))
            for _ in range(num_samples)
        ]
        instructions = self.api_client.batch_chat(prompts, max_tokens=256)

        # Step 2: Generate responses
        valid_instructions = [i.strip() for i in instructions if len(i.strip().split()) >= 5]
        answer_prompts = [ANSWER_PROMPT.format(instruction=instr) for instr in valid_instructions]
        answers = self.api_client.batch_chat(answer_prompts, max_tokens=1024)

        data: List[Dict[str, Any]] = []
        for instr, answer in zip(valid_instructions, answers):
            if answer.strip() and len(answer.strip().split()) >= 20:
                data.append({
                    "instruction": instr,
                    "input": "",
                    "output": answer.strip(),
                    "category": "creative",
                    "source": "fanno-synthesized",
                })

        logger.info(f"Generated {len(data)} creative writing samples")
        return data

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate creative writing samples."""
        return [
            item for item in data
            if len(item.get("output", "").split()) >= 20
            and len(item.get("instruction", "").split()) >= 5
        ]


__all__ = ["CreativeSynthesizer"]
