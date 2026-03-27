import json

from src.agents import multi_turn as mt_mod


def test_multi_turn_generator_runs(monkeypatch):
    """
    Use a deterministic stub for call_gpt to exercise the multi-turn loop.
    The stub returns different payloads based on prompt cues.
    """

    def stub_call(prompt, temperature=0.7, model=None, max_tokens=None):
        text = prompt.lower()
        # Initial user query
        if "generate a realistic initial user query" in text:
            return "Hi, can you help?"
        # Decide action
        if "decide your next action" in text:
            return json.dumps({"action": "call_tool", "reason": "test", "response": ""})
        # Function call generation
        if "generate the next tool call" in text:
            return json.dumps({"name": "lookup", "arguments": {"id": "123"}})
        # Tool execution simulation
        if "simulating the execution of a tool call" in text:
            return json.dumps({"status": "success", "data": {"value": 1}})
        # Assistant response after tool
        if "generate your response to the user based on the tool result" in text:
            return "Done."
        # Completion check
        if "has this conversation achieved its goal" in text:
            return json.dumps({"is_complete": True, "reason": "complete"})
        # User simulator
        if "simulate a realistic user" in text:
            return "User: thanks"
        # Default fallback
        return ""

    monkeypatch.setattr(mt_mod, "call_gpt", stub_call)
    gen = mt_mod.MultiTurnGenerator()
    scenario = {
        "system": "system prompt",
        "tools": [
            {
                "name": "lookup",
                "description": "desc",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            }
        ],
        "meta": {
            "expected_user_goal": "goal",
            "expected_tool_sequence": ["lookup"],
            "success_criteria": "done",
        },
    }
    result = gen.generate(scenario, num_turns=2)
    assert result["conversations"], "Conversations should not be empty"
    roles = {c["from"] for c in result["conversations"]}
    assert {"human", "function_call", "observation", "gpt"}.issubset(roles)


def test_multi_turn_decides_direct_response(monkeypatch):
    def stub_call(prompt, temperature=0.7, model=None, max_tokens=None):
        text = prompt.lower()
        if "generate a realistic initial user query" in text:
            return "Hello"
        if "decide your next action" in text:
            return json.dumps({"action": "respond_directly", "reason": "clarify", "response": "Need more info"})
        if "generate the user's next natural response" in text:
            return "User: thanks"
        if "has this conversation achieved its goal" in text:
            return json.dumps({"is_complete": True, "reason": "done"})
        return ""

    monkeypatch.setattr(mt_mod, "call_gpt", stub_call)
    gen = mt_mod.MultiTurnGenerator()
    scenario = {"system": "sys", "tools": [], "meta": {"expected_user_goal": "goal", "success_criteria": "done"}}
    res = gen.generate(scenario, num_turns=1)
    assert any(c["from"] == "gpt" and c["value"] == "Need more info" for c in res["conversations"])
