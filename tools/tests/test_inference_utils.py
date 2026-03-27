from src import inference_utils as iu


def test_inference_endpoints_present():
    endpoints = iu.get_endpoints()
    assert "gpt-4o" in endpoints
    assert "gpt-5" in endpoints
    # ensure entries have endpoint and model keys
    sample = endpoints["gpt-4o"][0]
    assert sample.get("endpoints")
    assert sample.get("model")


def test_select_endpoint_returns_entry():
    endpoints = iu.get_endpoints()
    entry = iu.select_endpoint("gpt-4o")
    assert entry in endpoints["gpt-4o"]
    assert entry["endpoints"].startswith("https://")
