import json

from fastapi.testclient import TestClient

import api.main as api

client = TestClient(api.app)


def _plan_payload():
    return {
        "template": "architecture",
        "palette": "light",
        "brand": "Test Studio",
        "eyebrow": "SPACE · LIGHT",
        "headline": "Modern spaces with purpose",
        "subtitle": "Thoughtful residential design.",
        "cta": "Explore projects",
        "card1": "Architecture",
        "card2": "Interiors",
        "card3": "Planning",
        "aboutText": "We create considered spaces.",
        "contactText": "Tell us about your project.",
        "customSections": [{"title": "Five-phase process", "text": "Plan, design, review, and hand over."}],
        "visualPrompts": ["Photorealistic contemporary architecture campus"],
    }


def test_health_never_returns_secret(monkeypatch):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "apiKey" not in data and "GEMINI_API_KEY" not in data


def test_short_prompt_is_rejected():
    response = client.post("/api/generate-plan", json={"prompt": "hi"})
    assert response.status_code == 422


def test_missing_key_returns_setup_instructions(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    response = client.post("/api/generate-plan", json={"prompt": "Create a modern architecture website with a five phase process."})
    assert response.status_code == 503


def test_plan_response_with_mocked_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit_test_key")

    async def fake_generate(_model, _payload, _key):
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(_plan_payload())}]}}]}

    monkeypatch.setattr(api, "gemini_generate", fake_generate)
    response = client.post("/api/generate-plan", json={"prompt": "Create a modern architecture website with a five phase workflow."})
    assert response.status_code == 200
    assert response.json()["template"] == "architecture"
    assert response.json()["brand"] == "Test Studio"


def test_image_response_with_mocked_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "unit_test_key")

    async def fake_generate(_model, _payload, _key):
        return {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}}]}}]}

    monkeypatch.setattr(api, "gemini_generate", fake_generate)
    response = client.post("/api/generate-image", json={"prompt": "A photorealistic skyline at blue hour", "category": "Cities"})
    assert response.status_code == 200
    assert response.json()["dataUrl"].startswith("data:image/png;base64,")
