import importlib

from fastapi.testclient import TestClient


def _reload_module(module_name):
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_targon_server_requires_configured_auth(monkeypatch):
    monkeypatch.delenv("MANTIS_SERVER_AUTH_KEY", raising=False)
    server = _reload_module("mantis_model_iteration_tool.targon_server")

    client = TestClient(server.app)
    response = client.get("/api/challenges")

    assert response.status_code == 503
    assert "MANTIS_SERVER_AUTH_KEY" in response.json()["detail"]


def test_targon_server_rejects_missing_and_invalid_bearer(monkeypatch):
    monkeypatch.setenv("MANTIS_SERVER_AUTH_KEY", "correct-token")
    server = _reload_module("mantis_model_iteration_tool.targon_server")

    client = TestClient(server.app)

    assert client.get("/api/challenges").status_code == 401
    assert client.get(
        "/api/challenges",
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401


def test_targon_server_accepts_valid_bearer(monkeypatch):
    monkeypatch.setenv("MANTIS_SERVER_AUTH_KEY", "correct-token")
    server = _reload_module("mantis_model_iteration_tool.targon_server")

    client = TestClient(server.app)
    response = client.get(
        "/api/challenges",
        headers={"Authorization": "Bearer correct-token"},
    )

    assert response.status_code == 200
    assert "ETH-1H-BINARY" in response.json()


def test_targon_eval_requires_configured_auth(monkeypatch):
    monkeypatch.delenv("MANTIS_EVAL_API_KEY", raising=False)
    eval_server = _reload_module("mantis_model_iteration_tool.targon_eval.server")

    client = TestClient(eval_server.app)
    response = client.get("/cache/status")

    assert response.status_code == 503
    assert "MANTIS_EVAL_API_KEY" in response.json()["detail"]


def test_targon_eval_accepts_valid_bearer(monkeypatch):
    monkeypatch.setenv("MANTIS_EVAL_API_KEY", "eval-token")
    eval_server = _reload_module("mantis_model_iteration_tool.targon_eval.server")

    client = TestClient(eval_server.app)
    assert client.get("/cache/status").status_code == 401

    response = client.get(
        "/cache/status",
        headers={"Authorization": "Bearer eval-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"cached": {}}


def test_public_health_responses_are_low_sensitivity(monkeypatch):
    monkeypatch.setenv("MANTIS_SERVER_AUTH_KEY", "server-token")
    monkeypatch.setenv("MANTIS_EVAL_API_KEY", "eval-token")
    server = _reload_module("mantis_model_iteration_tool.targon_server")
    eval_server = _reload_module("mantis_model_iteration_tool.targon_eval.server")

    server_health = TestClient(server.app).get("/health").json()
    eval_health = TestClient(eval_server.app).get("/health").json()

    assert "agents_running" not in server_health
    assert "cached_data_periods" not in eval_health
    assert server_health["auth_required"] is True
    assert eval_health["auth_required"] is True
