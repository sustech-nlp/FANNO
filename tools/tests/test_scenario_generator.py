import json

from src.agents import scenario_generator as sg_mod
from src.config import LOGIC_PATTERNS, ScenarioConfig


def test_scenario_generator_parses_llm(monkeypatch):
    # Stub LLM to return a valid scenario payload
    def stub_call(prompt, temperature=0.8, model=None, max_tokens=None):
        payload = {
            "system": "The current time is 2024-05-15 15:00:00 EST.\n" + ("policy " * 60),
            "tools": [
                {
                    "name": "lookup",
                    "description": "fetch info",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                }
            ],
            "meta": {
                "domain": "ecommerce",
                "logic_pattern": "smooth",
                "expected_user_goal": "fetch",
                "potential_issues": [],
                "expected_tool_sequence": ["lookup"],
                "success_criteria": "done",
            },
        }
        return json.dumps(payload)

    monkeypatch.setattr(sg_mod, "call_gpt", stub_call)
    gen = sg_mod.ScenarioGenerator()
    cfg = ScenarioConfig(num_tools=1, num_turns=8, logic_pattern=list(LOGIC_PATTERNS.keys())[0])
    seed = {"doc": "sample seed text"}
    scenario = gen.generate(seed, cfg)
    assert "system" in scenario and len(scenario["system"]) > 100
    assert isinstance(scenario["tools"], list) and scenario["tools"]
    assert "meta" in scenario and scenario["meta"]["logic_pattern"] == cfg.logic_pattern


def test_scenario_generator_invalid_response(monkeypatch):
    def stub_call(prompt, temperature=0.8, model=None, max_tokens=None):
        return '{"system": "short", "tools": []}'  # too short system prompt

    monkeypatch.setattr(sg_mod, "call_gpt", stub_call)
    gen = sg_mod.ScenarioGenerator()
    cfg = ScenarioConfig(num_tools=1, num_turns=8, logic_pattern=list(LOGIC_PATTERNS.keys())[0])
    seed = {"doc": "seed"}
    try:
        gen.generate(seed, cfg)
    except ValueError as e:
        assert "System prompt too short" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid scenario response")
