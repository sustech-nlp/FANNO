import json

from src.agents import world_model as wm_mod


def test_world_model_executes_via_llm(monkeypatch):
    # Stub LLM to return an observation payload
    def stub_call(prompt, temperature=0.7, model=None, max_tokens=None):
        return json.dumps({"status": "success", "data": {"result": "ok"}})

    monkeypatch.setattr(wm_mod, "call_gpt", stub_call)
    wm = wm_mod.WorldModel()
    function_call = {"name": "lookup", "arguments": {"id": "123"}}
    observation = wm.execute(function_call, "sys", [{"name": "lookup"}], [])
    assert observation["status"] == "success"
    assert observation["data"]["result"] == "ok"
    assert wm.get_state_summary()["execution_count"] == 1


def test_world_model_parse_error_fallback(monkeypatch):
    def stub_call(prompt, temperature=0.7, model=None, max_tokens=None):
        return "not json"

    monkeypatch.setattr(wm_mod, "call_gpt", stub_call)
    wm = wm_mod.WorldModel()
    obs = wm.execute({"name": "lookup", "arguments": {}}, "sys", [{"name": "lookup"}], [])
    assert obs["status"] == "error"
    assert "Failed to execute" in obs["error"]
