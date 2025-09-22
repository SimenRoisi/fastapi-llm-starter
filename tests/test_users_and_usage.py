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
        assert data["api_key"] == api_key
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

        # record usage
        r = await ac.post("/usage", json={"api_key": api_key, "endpoint": "/rates"})
        assert r.status_code == 201
        row = r.json()
        assert row["api_key"] == api_key
        assert row["endpoint"] == "/rates"

        # fetch recent usage
        r = await ac.get(f"/usage/{api_key}?limit=5")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) >= 1
        assert rows[0]["api_key"] == api_key
