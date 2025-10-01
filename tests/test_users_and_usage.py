import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_create_and_list_users():
    email = f"u_{uuid.uuid4().hex[:8]}@example.com"
    api_key = f"key_{uuid.uuid4().hex[:8]}"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # create
        r = await ac.post("/users", json={"email": email, "api_key": api_key})
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["email"] == email
        # API key should NOT be in response for security
        assert "api_key" not in data
        assert "id" in data and "created_at" in data

        # list
        r = await ac.get("/users")
        assert r.status_code == 200
        emails = [u["email"] for u in r.json()]
        assert email in emails
        

@pytest.mark.asyncio
async def test_usage_flow():
    # make a fresh user so FK passes
    email = f"u_{uuid.uuid4().hex[:8]}@example.com"
    api_key = f"key_{uuid.uuid4().hex[:8]}"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/users", json={"email": email, "api_key": api_key})
        assert r.status_code == 201

        # Set up authentication headers
        headers = {"X-API-Key": api_key}

        # record usage (uses auth header, not body)
        r = await ac.post("/usage", json={"endpoint": "/rates"}, headers=headers)
        assert r.status_code == 201
        row = r.json()
        # API key should NOT be in response
        assert "api_key" not in row
        assert row["endpoint"] == "/rates"

        # fetch recent usage (uses auth header, not URL parameter)
        r = await ac.get("/usage?limit=5", headers=headers)
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) >= 1
        # API key should NOT be in usage responses
        assert "api_key" not in rows[0]
        assert rows[0]["endpoint"] == "/rates"
