import json
import random

from src.config import LLM_CALL_PARAMS
from src.utils import call_gpt
from src.prompt_templates import build_execution_prompt


class WorldModel:
    """
    Simulates tool execution with observation diversity and minimal assumptions.
    """

    def __init__(self):
        self.execution_history = []
        self.status_distribution = {"success": 0, "partial_failure": 0, "error": 0}
        self.expected_diversity = []
        self.conversation_outline = []
        self.current_phase_index = 0

    def initialize(self, scenario_meta: dict):
        self.expected_diversity = scenario_meta.get("expected_observation_diversity", [])
        self.conversation_outline = scenario_meta.get("conversation_outline", [])
        self.current_phase_index = 0
        self.execution_history = []
        self.status_distribution = {"success": 0, "partial_failure": 0, "error": 0}

    def execute(self, function_call: dict, system_prompt: str, tools: list, conversation_history: list) -> dict:
        target_status = self._determine_target_status()
        prompt = build_execution_prompt(
            function_call,
            system_prompt,
            tools,
            conversation_history,
            self.execution_history,
            target_status=target_status,
        )
        params = LLM_CALL_PARAMS["world_model"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        observation = self._parse_execution_response(response, function_call, target_status)
        actual_status = observation.get("status", "success")
        self.status_distribution[actual_status] = self.status_distribution.get(actual_status, 0) + 1
        self.execution_history.append(
            {
                "function_call": function_call,
                "observation": observation,
                "status": actual_status,
            }
        )
        return observation

    def _determine_target_status(self) -> str:
        total = sum(self.status_distribution.values())
        if total == 0:
            return "success"
        current_ratios = {
            status: (count / total if total else 0) for status, count in self.status_distribution.items()
        }
        target_ratios = {"success": 0.7, "partial_failure": 0.2, "error": 0.1}
        deficits = {status: target_ratios[status] - current_ratios.get(status, 0) for status in target_ratios}
        if total >= 5 and current_ratios.get("success", 0) == 1.0:
            return "partial_failure"
        max_deficit_status = max(deficits, key=deficits.get)
        if deficits[max_deficit_status] > 0.1:
            return max_deficit_status
        return random.choices(list(target_ratios.keys()), weights=list(target_ratios.values()))[0]

    def _parse_execution_response(self, response: str, function_call: dict, target_status: str | None) -> dict:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        try:
            observation = json.loads(response.strip())
            if observation.get("status") not in {"success", "partial_failure", "error"}:
                observation["status"] = target_status or "success"
            if "data" not in observation:
                observation["data"] = {}
            return observation
        except json.JSONDecodeError:
            return {
                "status": target_status or "error",
                "error": f"Failed to execute {function_call.get('name')}",
                "data": {},
            }

    def get_diversity_report(self) -> dict:
        total = sum(self.status_distribution.values())
        if total == 0:
            return {"has_diversity": False, "distribution": {}}
        ratios = {s: c / total for s, c in self.status_distribution.items()}
        non_zero = sum(1 for c in self.status_distribution.values() if c > 0)
        return {
            "has_diversity": non_zero >= 2,
            "distribution": self.status_distribution,
            "ratios": ratios,
            "unique_statuses": non_zero,
        }
