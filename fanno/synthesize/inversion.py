"""Trajectory inversion: generate new scenarios from existing trajectories."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from fanno.synthesize.prompts import (
    INVERSION_EXTRACT_PROMPT,
    INVERSION_GENERATE_PROMPT,
)


class TrajectoryInverter:
    """Generate new scenarios from existing agent trajectories.

    Analyzes completed trajectories to extract patterns and create
    new, related but distinct task scenarios.
    """

    def __init__(self, model: str = "gpt-4o-mini", workers: int = 8) -> None:
        self.model = model
        self.workers = workers
        self._api_client = None

    @property
    def api_client(self):
        if self._api_client is None:
            from fanno.api.client import AzureAPIClient
            self._api_client = AzureAPIClient(
                model_name=self.model, workers=self.workers
            )
        return self._api_client

    def extract_trajectory_text(self, trajectory: Dict[str, Any]) -> str:
        """Convert trajectory to readable text format.

        Handles both FANNO-Tools format (conversations with function_call/observation)
        and OpenAI function-calling format (messages with function_call).
        """
        lines: List[str] = []

        # Handle OpenAI message format
        if "messages" in trajectory:
            for msg in trajectory["messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if msg.get("function_call"):
                    fc = msg["function_call"]
                    name = fc.get("name", "unknown")
                    args = fc.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    lines.append(f"[{role}] Called {name}({json.dumps(args)})")
                elif msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        lines.append(f"[{role}] Called {fn.get('name', '?')}({fn.get('arguments', '{}')})")
                elif role == "tool":
                    lines.append(f"[tool:{msg.get('name', '?')}] {content[:200]}")
                else:
                    lines.append(f"[{role}] {content[:300]}")

        # Handle FANNO-Tools conversations format
        elif "conversations" in trajectory:
            for msg in trajectory["conversations"]:
                role = msg.get("role", msg.get("from", "unknown"))
                content = msg.get("content", msg.get("value", ""))
                if msg.get("function_call"):
                    fc = msg["function_call"]
                    lines.append(f"[{role}] Called {fc.get('name', '?')}({fc.get('arguments', '{}')})")
                elif role in ("observation", "tool"):
                    lines.append(f"[observation] {content[:200]}")
                else:
                    lines.append(f"[{role}] {content[:300]}")

        return "\n".join(lines)

    def extract_key_decisions(self, trajectory: Dict[str, Any]) -> List[str]:
        """Extract key decision points from a trajectory.

        Args:
            trajectory: A trajectory dict (OpenAI or FANNO-Tools format).

        Returns:
            List of decision point descriptions.
        """
        traj_text = self.extract_trajectory_text(trajectory)
        prompt = INVERSION_EXTRACT_PROMPT.format(trajectory=traj_text[:2000])
        responses = self.api_client.batch_chat([prompt], max_tokens=512)
        response = responses[0] if responses else ""

        # Parse numbered decisions
        decisions: List[str] = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                decisions.append(line.lstrip("0123456789.-) "))

        return decisions

    def invert(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a new scenario from an existing trajectory.

        Args:
            trajectory: A completed trajectory.

        Returns:
            New scenario dict with 'task', 'tools', 'metadata'.
        """
        traj_text = self.extract_trajectory_text(trajectory)

        # Summarize the original trajectory
        summary = traj_text[:1500]

        prompt = INVERSION_GENERATE_PROMPT.format(summary=summary)
        responses = self.api_client.batch_chat([prompt], max_tokens=512)
        new_scenario_text = responses[0] if responses else ""

        # Extract tools from original trajectory
        original_tools = trajectory.get("tools", [])
        if not original_tools and "metadata" in trajectory:
            # Try to reconstruct from metadata
            tool_names = trajectory["metadata"].get("tool_names", [])
            from fanno.synthesize.agent import TOOL_LIBRARY
            original_tools = [TOOL_LIBRARY[n] for n in tool_names if n in TOOL_LIBRARY]

        return {
            "task": new_scenario_text.strip(),
            "tools": original_tools,
            "metadata": {
                "source": "trajectory_inversion",
                "original_task": trajectory.get("metadata", {}).get("task", ""),
                "original_role": trajectory.get("metadata", {}).get("role", ""),
            },
        }

    def batch_invert(
        self,
        trajectories: List[Dict[str, Any]],
        num_per_trajectory: int = 2,
    ) -> List[Dict[str, Any]]:
        """Generate multiple new scenarios from each trajectory.

        Args:
            trajectories: List of completed trajectories.
            num_per_trajectory: Number of inversions per trajectory.

        Returns:
            List of new scenario dicts.
        """
        logger.info(
            f"Inverting {len(trajectories)} trajectories "
            f"({num_per_trajectory} each = {len(trajectories) * num_per_trajectory} total)"
        )

        results: List[Dict[str, Any]] = []
        for i, trajectory in enumerate(trajectories):
            for j in range(num_per_trajectory):
                try:
                    new_scenario = self.invert(trajectory)
                    results.append(new_scenario)
                except Exception as e:
                    logger.warning(f"Failed to invert trajectory {i}, attempt {j}: {e}")

            if (i + 1) % 10 == 0:
                logger.info(f"Inverted {i + 1}/{len(trajectories)} trajectories")

        logger.info(f"Generated {len(results)} new scenarios from inversions")
        return results


__all__ = ["TrajectoryInverter"]
