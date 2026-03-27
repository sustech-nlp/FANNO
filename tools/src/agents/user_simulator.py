from src.config import LLM_CALL_PARAMS
from src.prompt_templates import build_user_prompt
from src.utils import call_gpt


class UserSimulator:
    """
    Generates realistic user responses based on conversation context.
    """

    def generate_response(self, system_prompt: str, conversation_history: list, goal: str, world_state: dict = None) -> str:
        prompt = build_user_prompt(system_prompt, conversation_history, goal, world_state or {})
        params = LLM_CALL_PARAMS["user_simulator"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        return self._parse_user_response(response)

    def _parse_user_response(self, response: str) -> str:
        response = response.strip()
        prefixes = ["User:", "USER:", "Response:", "RESPONSE:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        if response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        return response


__all__ = ["UserSimulator"]
