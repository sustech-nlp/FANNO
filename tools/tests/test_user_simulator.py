from src.agents import user_simulator as us_mod


def test_user_simulator_parses_response(monkeypatch):
    def stub_call(prompt, temperature=0.9, model=None, max_tokens=None):
        return "User: Sure, go ahead."

    monkeypatch.setattr(us_mod, "call_gpt", stub_call)
    sim = us_mod.UserSimulator()
    resp = sim.generate_response("sys", [], "goal", {})
    assert resp == "Sure, go ahead."
