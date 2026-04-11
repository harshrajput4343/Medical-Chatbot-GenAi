import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_config_loaded():
    assert settings.SECRET_KEY == "test_secret_key_12345"
    assert settings.DEBUG is True

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_auth_pages_render():
    # Login page
    response = client.get("/auth/login")
    assert response.status_code == 200
    assert "Log in" in response.text or "Sign in" in response.text
    
    # Register page
    response = client.get("/auth/register")
    assert response.status_code == 200
    assert "Register" in response.text or "Create Account" in response.text

def test_unauthorized_access():
    # Attempt to access dashboard without cookie
    response = client.get("/dashboard", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/auth/login"
