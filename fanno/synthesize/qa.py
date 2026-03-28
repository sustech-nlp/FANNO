"""QA synthesis: complex, code, math, and reasoning questions."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from loguru import logger

from fanno.synthesize.base import BaseSynthesizer
from fanno.synthesize.prompts import (
    QA_GENERAL_PROMPT,
    QA_CODE_PROMPT,
    QA_MATH_PROMPT,
    QA_REASONING_PROMPT,
    ANSWER_PROMPT,
)


# Topic pools for diverse generation
GENERAL_TOPICS = [
    "artificial intelligence", "climate change", "quantum computing",
    "gene editing", "space exploration", "cybersecurity", "renewable energy",
    "blockchain technology", "autonomous vehicles", "mental health",
    "economic policy", "education reform", "healthcare systems",
    "urban planning", "food science", "marine biology", "astrophysics",
    "materials science", "cognitive psychology", "international relations",
    "data privacy", "cultural anthropology", "bioethics", "game theory",
    "network science", "computational linguistics", "robotics",
    "environmental engineering", "behavioral economics", "neuroscience",
]

CODE_CONCEPTS = [
    ("Python", "data structures", "medium"),
    ("Python", "algorithms", "hard"),
    ("Python", "object-oriented programming", "medium"),
    ("Python", "functional programming", "medium"),
    ("Python", "concurrency and parallelism", "hard"),
    ("JavaScript", "async/await patterns", "medium"),
    ("JavaScript", "DOM manipulation", "easy"),
    ("SQL", "complex queries and joins", "medium"),
    ("SQL", "window functions", "hard"),
    ("Python", "dynamic programming", "hard"),
    ("Python", "graph algorithms", "hard"),
    ("Python", "string manipulation", "easy"),
    ("Python", "tree traversal", "medium"),
    ("Python", "sorting algorithms", "medium"),
]

MATH_TOPICS = [
    ("algebra", "easy"), ("calculus", "medium"), ("probability", "medium"),
    ("number theory", "hard"), ("combinatorics", "medium"),
    ("linear algebra", "medium"), ("statistics", "easy"),
    ("geometry", "medium"), ("optimization", "hard"),
    ("discrete mathematics", "medium"),
]

REASONING_DOMAINS = [
    "science and technology", "business and economics", "law and ethics",
    "philosophy and logic", "social sciences", "environmental studies",
    "medicine and health", "engineering design", "policy analysis",
]


class QASynthesizer(BaseSynthesizer):
    """Synthesize QA pairs across multiple categories."""

    CATEGORIES = ["general", "code", "math", "reasoning"]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        workers: int = 8,
        categories: Optional[List[str]] = None,
    ) -> None:
        super().__init__(model=model, workers=workers)
        self.categories = categories or self.CATEGORIES

    def _generate_prompts(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate instruction prompts for all categories."""
        prompts: List[Dict[str, str]] = []
        per_category = max(1, num_samples // len(self.categories))

        for category in self.categories:
            for _ in range(per_category):
                if category == "general":
                    topic = random.choice(GENERAL_TOPICS)
                    prompt = QA_GENERAL_PROMPT.format(topic=topic)
                elif category == "code":
                    lang, concept, diff = random.choice(CODE_CONCEPTS)
                    prompt = QA_CODE_PROMPT.format(
                        language=lang, concept=concept, difficulty=diff
                    )
                elif category == "math":
                    topic, diff = random.choice(MATH_TOPICS)
                    prompt = QA_MATH_PROMPT.format(topic=topic, difficulty=diff)
                elif category == "reasoning":
                    domain = random.choice(REASONING_DOMAINS)
                    prompt = QA_REASONING_PROMPT.format(domain=domain)
                else:
                    continue
                prompts.append({"prompt": prompt, "category": category})

        random.shuffle(prompts)
        return prompts[:num_samples]

    def generate(self, num_samples: int = 1000, **kwargs) -> List[Dict[str, Any]]:
        """Generate QA pairs.

        1. Generate instruction prompts across categories
        2. Call LLM to generate instructions
        3. Call LLM to generate answers
        4. Return instruction/output pairs
        """
        logger.info(f"Generating {num_samples} QA pairs across {self.categories}")

        # Step 1: Generate instructions
        prompt_items = self._generate_prompts(num_samples)
        prompts = [item["prompt"] for item in prompt_items]
        categories = [item["category"] for item in prompt_items]

        instructions = self.api_client.batch_chat(prompts, max_tokens=256)

        # Clean instructions
        cleaned_instructions: List[str] = []
        valid_categories: List[str] = []
        for instr, cat in zip(instructions, categories):
            instr = instr.strip().strip('"').strip("'")
            if len(instr.split()) >= 5:
                cleaned_instructions.append(instr)
                valid_categories.append(cat)

        # Step 2: Generate answers
        answer_prompts = [ANSWER_PROMPT.format(instruction=instr) for instr in cleaned_instructions]
        answers = self.api_client.batch_chat(answer_prompts, max_tokens=1024)

        # Step 3: Assemble
        data: List[Dict[str, Any]] = []
        for instr, answer, cat in zip(cleaned_instructions, answers, valid_categories):
            if answer.strip():
                data.append({
                    "instruction": instr,
                    "input": "",
                    "output": answer.strip(),
                    "category": cat,
                    "source": "fanno-synthesized",
                })

        logger.info(f"Generated {len(data)}/{num_samples} valid QA pairs")
        return data

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate QA pairs with basic quality checks."""
        valid: List[Dict[str, Any]] = []
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            # Basic quality checks
            if len(instruction.split()) < 5:
                continue
            if len(output.split()) < 10:
                continue
            if output.lower().startswith("i cannot") or output.lower().startswith("i'm sorry"):
                continue
            valid.append(item)
        return valid


__all__ = ["QASynthesizer"]
