"""Multi-turn dialog data synthesis."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from loguru import logger

from fanno.synthesize.base import BaseSynthesizer
from fanno.synthesize.prompts import (
    DIALOG_SCENARIO_PROMPT,
    DIALOG_USER_TURN_PROMPT,
    DIALOG_ASSISTANT_TURN_PROMPT,
)

DIALOG_TOPICS = [
    "debugging a Python application", "planning a vacation",
    "learning a new programming language", "understanding machine learning",
    "cooking a complex recipe", "financial planning and investing",
    "home renovation project", "career advice and job searching",
    "writing a research paper", "setting up a home network",
    "fitness and nutrition planning", "starting a small business",
    "learning a musical instrument", "gardening tips and techniques",
    "photography techniques", "language learning strategies",
]


class DialogSynthesizer(BaseSynthesizer):
    """Synthesize multi-turn conversation data."""

    def generate(
        self,
        num_samples: int = 500,
        min_turns: int = 2,
        max_turns: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate multi-turn dialog data.

        Each dialog has 2-5 user-assistant turn pairs.
        Output in ShareGPT format.
        """
        logger.info(f"Generating {num_samples} multi-turn dialogs")

        data: List[Dict[str, Any]] = []

        # Process in batches for efficiency
        batch_size = min(50, num_samples)

        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_count = batch_end - batch_start

            # Generate initial user messages
            topics = [random.choice(DIALOG_TOPICS) for _ in range(batch_count)]
            num_turns_list = [random.randint(min_turns, max_turns) for _ in range(batch_count)]

            # Generate first scenario
            scenario_prompts = [
                DIALOG_SCENARIO_PROMPT.format(
                    topic=topic,
                    complexity=random.choice(["simple", "moderate", "complex"]),
                    num_turns=nt,
                )
                for topic, nt in zip(topics, num_turns_list)
            ]
            first_messages = self.api_client.batch_chat(scenario_prompts, max_tokens=512)

            # Build conversations turn by turn
            for idx in range(batch_count):
                conversations: List[Dict[str, str]] = []
                first_msg = first_messages[idx].strip()

                # Parse first turn or use as single user message
                if "User:" in first_msg and "Assistant:" in first_msg:
                    # Parse structured conversation
                    turns = first_msg.split("User:")
                    for turn in turns[1:]:  # skip empty first
                        if "Assistant:" in turn:
                            user_part, asst_part = turn.split("Assistant:", 1)
                            conversations.append({"from": "human", "value": user_part.strip()})
                            conversations.append({"from": "gpt", "value": asst_part.strip()})
                else:
                    # Use as first user message, generate response
                    conversations.append({"from": "human", "value": first_msg})

                # Generate additional turns if needed
                target_turns = num_turns_list[idx]
                current_pairs = len(conversations) // 2

                while current_pairs < target_turns:
                    history = self._format_history(conversations)

                    # Generate assistant response if last message is from user
                    if conversations and conversations[-1]["from"] == "human":
                        asst_prompts = [DIALOG_ASSISTANT_TURN_PROMPT.format(history=history)]
                        asst_responses = self.api_client.batch_chat(asst_prompts, max_tokens=512)
                        conversations.append({"from": "gpt", "value": asst_responses[0].strip()})

                    # Generate next user message
                    if current_pairs < target_turns - 1 or len(conversations) % 2 == 0:
                        history = self._format_history(conversations)
                        user_prompts = [DIALOG_USER_TURN_PROMPT.format(history=history)]
                        user_responses = self.api_client.batch_chat(user_prompts, max_tokens=256)
                        conversations.append({"from": "human", "value": user_responses[0].strip()})

                    current_pairs = len(conversations) // 2

                # Ensure conversation ends with assistant
                if conversations and conversations[-1]["from"] == "human":
                    history = self._format_history(conversations)
                    asst_prompts = [DIALOG_ASSISTANT_TURN_PROMPT.format(history=history)]
                    asst_responses = self.api_client.batch_chat(asst_prompts, max_tokens=512)
                    conversations.append({"from": "gpt", "value": asst_responses[0].strip()})

                if len(conversations) >= 4:  # at least 2 turns
                    data.append({
                        "conversations": conversations,
                        "topic": topics[idx],
                        "num_turns": len(conversations) // 2,
                        "source": "fanno-synthesized",
                        "category": "dialog",
                    })

            logger.info(f"Generated {len(data)}/{num_samples} dialogs so far")

        logger.info(f"Generated {len(data)} multi-turn dialogs total")
        return data

    def _format_history(self, conversations: List[Dict[str, str]]) -> str:
        """Format conversation history as text."""
        lines: List[str] = []
        for msg in conversations:
            role = "User" if msg["from"] == "human" else "Assistant"
            lines.append(f"{role}: {msg['value']}")
        return "\n\n".join(lines)

    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate dialog samples."""
        valid: List[Dict[str, Any]] = []
        for item in data:
            convs = item.get("conversations", [])
            if len(convs) < 4:
                continue
            # Check all turns have content
            if all(msg.get("value", "").strip() for msg in convs):
                valid.append(item)
        return valid


__all__ = ["DialogSynthesizer"]
