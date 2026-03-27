import json

from src.agents import quality as quality_mod


def test_quality_evaluator_returns_score(monkeypatch):
    # Stub LLM evaluation response
    def stub_call(prompt, temperature=0.3, model=None, max_tokens=None):
        return json.dumps({"score": 9})

    monkeypatch.setattr(quality_mod, "call_gpt", stub_call)
    evaluator = quality_mod.QualityEvaluator()
    scenario = {"system": "x" * 120, "meta": {}}
    tools = [
        {
            "name": "tool",
            "description": "desc",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    ]
    score = evaluator.evaluate(scenario, tools, [])
    assert score == 9
