import json
from typing import Dict, List, Tuple

from src.config import LLM_CALL_PARAMS
from src.prompt_templates import (
    build_completion_check_prompt,
    build_decide_action_prompt,
    build_function_call_prompt,
    build_gpt_response_prompt,
    build_initial_query_prompt,
)
from src.utils import call_gpt
from .world_model import WorldModel
from .user_simulator import UserSimulator


class MultiTurnGenerator:
    """
    Orchestrates the generation of complete multi-turn conversations with anti-hallucination and coverage tracking.
    """

    def __init__(self):
        self.world_model = WorldModel()
        self.user_simulator = UserSimulator()

    def generate(self, scenario: Dict, num_turns: int = 10) -> Dict:
        system = scenario["system"]
        tools = scenario["tools"]
        meta = scenario.get("meta", {})

        initial_query = self._generate_initial_query(system, meta)
        conversations = [{"from": "human", "value": initial_query}]

        # initialize world model with meta for diversity hints
        self.world_model.initialize(meta)

        tools_used = set()
        tools_available = [t["name"] for t in tools]
        min_turns = self._parse_min_turns(meta.get("estimated_turns", "6-10"))

        for turn in range(num_turns):
            try:
                should_call_tool, direct_response = self._decide_next_action(
                    system, tools, conversations, tools_used, tools_available
                )
                if should_call_tool:
                    function_call = self._generate_function_call(system, tools, conversations, meta)
                    conversations.append({"from": "function_call", "value": json.dumps(function_call, ensure_ascii=False)})
                    tools_used.add(function_call.get("name"))

                    observation = self.world_model.execute(function_call, system, tools, conversations)
                    conversations.append({"from": "observation", "value": json.dumps(observation, ensure_ascii=False)})

                    gpt_response = self._generate_gpt_response(system, tools, conversations)
                    conversations.append({"from": "gpt", "value": gpt_response})
                else:
                    conversations.append({"from": "gpt", "value": direct_response})

                should_end, reason = self._should_end_conversation(
                    conversations, meta, turn, min_turns, tools_used, tools_available
                )
                if should_end:
                    break

                user_response = self.user_simulator.generate_response(
                    system_prompt=system,
                    conversation_history=conversations,
                    goal=meta.get("expected_user_goal", ""),
                    world_state=self.world_model.get_diversity_report(),
                )
                conversations.append({"from": "human", "value": user_response})
            except Exception as e:
                print(f"Error in turn {turn}: {e}")
                break

        return {
            "system": system,
            "tools": json.dumps(tools, ensure_ascii=False),
            "conversations": conversations,
        }

    def _generate_initial_query(self, system: str, meta: Dict) -> str:
        prompt = build_initial_query_prompt(system, meta)
        params = LLM_CALL_PARAMS["initial_query"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        return response.strip()

    def _decide_next_action(
        self,
        system: str,
        tools: List[Dict],
        conversations: List[Dict],
        tools_used: set,
        tools_available: List[str],
    ) -> Tuple[bool, str]:
        usage_rate = len(tools_used) / len(tools_available) if tools_available else 0
        prompt = f"""
Decide the next action for the assistant.

SYSTEM:
{system}

AVAILABLE TOOLS:
{json.dumps([{"name": t["name"], "description": t["description"]} for t in tools], indent=2)}

TOOLS ALREADY USED: {list(tools_used)}
TOOL USAGE RATE: {usage_rate:.1%}

CONVERSATION:
{self._format_conversations(conversations[-6:])}

DECISION GUIDANCE:
- If usage rate < 50% and conversation allows, prefer calling a new tool
- If the last turn was observation, usually respond to user
- If user asks a question, respond or clarify before calling tools
- If user confirms an action, call the appropriate tool

Output as JSON:
{{
  "action": "call_tool" | "respond_directly",
  "reason": "Brief explanation",
  "response": "Direct response text (only if respond_directly)"
}}
"""
        try:
            params = LLM_CALL_PARAMS["decide_action"]
            response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
            result = json.loads(self._extract_json(response))
            should_call = result.get("action") == "call_tool"
            direct_resp = result.get("response", "")
            return should_call, direct_resp
        except Exception:
            return usage_rate < 0.5, ""

    def _generate_function_call(self, system: str, tools: List[Dict], conversations: List[Dict], meta: Dict) -> Dict:
        prompt = build_function_call_prompt(system, tools, conversations, meta)
        try:
            params = LLM_CALL_PARAMS["function_call"]
            response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens).strip()
            function_call = json.loads(self._extract_json(response))
            return function_call
        except Exception:
            return {"name": tools[0]["name"] if tools else "unknown_tool", "arguments": {}}

    def _generate_gpt_response(self, system: str, tools: List[Dict], conversations: List[Dict]) -> str:
        prompt = build_gpt_response_prompt(system, tools, conversations)
        params = LLM_CALL_PARAMS["gpt_response"]
        response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
        return self._clean_response(response)

    def _should_end_conversation(
        self,
        conversations: List[Dict],
        meta: Dict,
        current_turn: int,
        min_turns: int,
        tools_used: set,
        tools_available: List[str],
    ) -> Tuple[bool, str]:
        if len(conversations) < min_turns:
            return False, ""

        recent_turns = conversations[-3:]
        for turn in reversed(recent_turns):
            if turn.get("from") == "function_call":
                try:
                    fc = json.loads(turn["value"])
                    tool_name = fc.get("name", "").lower()
                    if any(k in tool_name for k in ["transfer", "escalate", "finalize", "complete", "close"]):
                        return True, f"Ending tool called: {tool_name}"
                except Exception:
                    pass

        if len(conversations) >= min_turns + 2:
            prompt = f"""
Has this conversation reached a natural conclusion?

GOAL: {meta.get('expected_user_goal', 'Complete the task')}

RECENT CONVERSATION:
{self._format_conversations(conversations[-6:])}

Consider:
- Has the user's goal been addressed?
- Is there a sense of completion or satisfaction?
- Has the user said something like "thank you" or "that's all"?

Answer with JSON:
{{"should_end": true/false, "reason": "..."}}
"""
            try:
                params = LLM_CALL_PARAMS["completion_check"]
                response = call_gpt(prompt, temperature=params.temperature, max_tokens=params.max_tokens)
                result = json.loads(self._extract_json(response))
                if result.get("should_end"):
                    return True, result.get("reason", "Goal achieved")
            except Exception:
                pass

        return False, ""

    def _parse_min_turns(self, estimated: str) -> int:
        try:
            parts = str(estimated).split("-")
            if len(parts) >= 1 and parts[0].isdigit():
                return int(parts[0])
        except Exception:
            pass
        return 6

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _clean_response(self, text: str) -> str:
        text = text.strip()
        prefixes = ["Assistant:", "Response:", "GPT:", "AI:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        return text

    def _format_conversations(self, convs: List[Dict]) -> str:
        lines = []
        for c in convs:
            role = c.get("from")
            value = c.get("value", "")
            if role == "function_call":
                try:
                    fc = json.loads(value)
                    lines.append(f"[TOOL] {fc['name']}(...)")
                except Exception:
                    lines.append("[TOOL] (invalid)")
            elif role == "observation":
                try:
                    obs = json.loads(value)
                    status = obs.get("status", "unknown")
                    lines.append(f"[RESULT] status={status}")
                except Exception:
                    lines.append("[RESULT] (invalid)")
            else:
                preview = value[:100] + "..." if len(value) > 100 else value
                lines.append(f"[{role.upper()}] {preview}")
        return "\n".join(lines)


__all__ = ["MultiTurnGenerator"]
