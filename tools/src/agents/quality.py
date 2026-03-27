import json

from src.config import LLM_CALL_PARAMS
from src.prompt_templates import build_conversation_evaluation_prompt, build_evaluation_prompt
from src.utils import call_gpt, cosine_similarity, hash_embedding


class QualityEvaluator:
    """
    Evaluates the quality of generated scenarios using LLM-based scoring.
    """

    def evaluate(self, scenario, tools=None, conversation_history=None):
        tools = tools or scenario.get("tools", [])
        if conversation_history:
            prompt = build_conversation_evaluation_prompt(scenario, tools, conversation_history)
            params = LLM_CALL_PARAMS["quality_conversation"]
            response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
            return self._parse_conversation_evaluation_response(response)
        prompt = build_evaluation_prompt(scenario, tools, conversation_history or [])
        params = LLM_CALL_PARAMS["quality_scenario"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        return self._parse_evaluation_response(response)

    def _parse_evaluation_response(self, response: str) -> int:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            data = json.loads(response.strip())
            score = data.get("score", 0)
            return min(max(int(score), 0), 10)
        except (json.JSONDecodeError, ValueError, KeyError):
            return 5

    def _parse_conversation_evaluation_response(self, response: str) -> int:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        try:
            data = json.loads(response.strip())
            if str(data.get("verdict", "")).upper() == "REJECT":
                return 0
            score = data.get("score", 0)
            return min(max(int(score), 0), 10)
        except (json.JSONDecodeError, ValueError, KeyError):
            return 0

    def format_conversation(self, history):
        lines = []
        for i, turn in enumerate(history):
            role = turn.get("from")
            value = turn.get("value", "")
            if role == "function_call":
                try:
                    fc = json.loads(value)
                    lines.append(f"Turn {i+1} [TOOL]: {fc.get('name', 'unknown')}")
                except Exception:
                    lines.append(f"Turn {i+1} [TOOL]: (parse error)")
            elif role == "observation":
                try:
                    obs = json.loads(value)
                    lines.append(f"Turn {i+1} [RESULT]: status={obs.get('status', 'unknown')}")
                except Exception:
                    lines.append(f"Turn {i+1} [RESULT]: (parse error)")
            else:
                preview = value[:80] + "..." if len(value) > 80 else value
                lines.append(f"Turn {i+1} [{role.upper()}]: {preview}")
        return "\n".join(lines)


class ScenarioDiversityTracker:
    def __init__(self, dim=64):
        self.dim = dim
        self.vectors = []
        self.similarities = []

    def add(self, text):
        vec = hash_embedding(text, self.dim)
        if self.vectors:
            sims = [cosine_similarity(vec, v) for v in self.vectors]
            self.similarities.append(sum(sims) / len(sims))
        self.vectors.append(vec)

    def average_similarity(self):
        if not self.similarities:
            return 0.0
        return sum(self.similarities) / len(self.similarities)


class MetricsTracker:
    def __init__(self):
        self.accepted = 0
        self.rejected = 0
        self.total_length = 0
        self.total_tool_calls = 0
        self.success_tool_calls = 0
        self.diversity = ScenarioDiversityTracker()

    def record_evaluation(self, score, threshold):
        if score >= threshold:
            self.accepted += 1
        else:
            self.rejected += 1

    def record_conversation(self, record, scenario):
        self.total_length += len(record.get("conversations", []))
        for item in record.get("conversations", []):
            role = item.get("from")
            if role == "function_call":
                self.total_tool_calls += 1
            if role == "observation":
                try:
                    obs = json.loads(item.get("value", "{}"))
                    if obs.get("status") == "success":
                        self.success_tool_calls += 1
                except json.JSONDecodeError:
                    continue
        self.diversity.add(scenario.get("system", ""))

    def summary(self):
        total = self.accepted + self.rejected
        acceptance_rate = self.accepted / total if total else 0.0
        avg_len = self.total_length / self.accepted if self.accepted else 0.0
        success_rate = (
            self.success_tool_calls / self.total_tool_calls if self.total_tool_calls else 0.0
        )
        return {
            "acceptance_rate": round(acceptance_rate, 3),
            "avg_conversation_length": round(avg_len, 2),
            "tool_call_success_rate": round(success_rate, 3),
            "scenario_avg_similarity": round(self.diversity.average_similarity(), 3),
        }


__all__ = ["QualityEvaluator", "ScenarioDiversityTracker", "MetricsTracker"]
