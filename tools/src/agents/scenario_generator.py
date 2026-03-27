import json
from typing import Dict

from src.config import LOGIC_PATTERNS, LLM_CALL_PARAMS, ScenarioConfig
from src.prompt_templates import build_scenario_prompt
from src.utils import call_gpt


class ScenarioGenerator:
    """
    Generates conversation scenarios with system prompts and tool definitions.
    Uses LLM to creatively design scenarios based on seed documents.
    """

    def generate(self, seed_data: Dict, config: ScenarioConfig) -> Dict:
        prompt = build_scenario_prompt(seed_data.get("doc", ""), config, LOGIC_PATTERNS)
        params = LLM_CALL_PARAMS["scenario_generation"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        return self._parse_scenario_response(response)

    def _parse_scenario_response(self, response: str) -> Dict:
        """Parse and validate LLM response"""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            data = json.loads(response.strip())

            # Validate required fields
            assert "system" in data, "Missing 'system' field"
            assert "tools" in data, "Missing 'tools' field"
            assert "meta" in data, "Missing 'meta' field"
            assert isinstance(data["tools"], list), "'tools' must be a list"
            assert len(data["system"]) >= 100, "System prompt too short"

            # Validate tool format
            for tool in data["tools"]:
                assert "name" in tool, f"Tool missing 'name': {tool}"
                assert "description" in tool, f"Tool missing 'description': {tool}"
                assert "parameters" in tool, f"Tool missing 'parameters': {tool}"
                assert tool["parameters"].get("type") == "object", (
                    f"Tool parameters type must be 'object': {tool.get('name')}"
                )

            return data

        except (json.JSONDecodeError, AssertionError) as e:
            raise ValueError(f"Failed to parse scenario response: {e}\n{response[:500]}...")


__all__ = ["ScenarioGenerator"]
