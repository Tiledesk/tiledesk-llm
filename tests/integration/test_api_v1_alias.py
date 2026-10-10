"""
/api/v1/* is an alias of every /api/* endpoint (same handler, same contract);
/api/* keeps answering. Routes that are natively /api/v1 (System One) are served
as they are, never rewritten.
"""
import pytest


@pytest.mark.parametrize("path", ["/api/ask", "/api/qa", "/api/v2/qa", "/api/agentic-compliance/sessions"])
def test_v1_alias_reaches_the_same_handler(client, path):
    plain = client.post(path, json={"not": "valid"})
    aliased = client.post(path.replace("/api/", "/api/v1/", 1), json={"not": "valid"})
    assert plain.status_code == 422
    assert aliased.status_code == plain.status_code
    assert aliased.json()["detail"] == plain.json()["detail"]


def test_native_v1_routes_are_not_rewritten(client):
    resp = client.get("/api/v1/systemone/providers")
    assert resp.status_code == 200


def test_unknown_path_is_404_on_both(client):
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api/v1/nope").status_code == 404
