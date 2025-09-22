# FastAPI LLM Starter

A minimal but production-ready FastAPI backend with PostgreSQL, SQLAlchemy ORM, Alembic migrations, and Docker Compose.  
The project demonstrates modern API development practices and includes integration with OpenAI’s API for LLM-powered endpoints.  
It is designed as a learning project and a foundation for future Retrieval-Augmented Generation (RAG) features.

---

## ✨ Features
- **FastAPI** for async web APIs with automatic docs (`/docs` and `/redoc`).
- **SQLAlchemy ORM (2.x)** for database models and queries.
- **PostgreSQL** for persistent storage.
- **Alembic** for schema migrations.
- **Pydantic** for input/output validation.
- **Docker Compose** to run the full stack (API + Postgres + Adminer).
- **OpenAI integration** with a sample `/assist` endpoint (chat-based assistant).
- **Extensible design** — ready for adding RAG, vector search, or other features.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/SimenRoisi/fastapi-llm-starter.git
cd fastapi-llm-starter
```
### 2. Create a .env file
Copy .env.example → .env and fill in your own values:
```
DB_USER=app
DB_PASS=devpass
DB_HOST=db
DB_PORT=5432
DB_NAME=appdb
OPENAI_API_KEY=sk-...
```
3. Build and start services
```
docker compose up --build
```
This starts:  
API at http://localhost:8000  
Swagger docs at http://localhost:8000/docs  
Adminer DB UI at http://localhost:8080  
```
docker compose exec api alembic upgrade head
```
5. Try the API

Create a user:
```
curl -X POST http://localhost:8000/users \
     -H "Content-Type: application/json" \
     -d '{"email":"me@example.com","api_key":"my-test-key"}'
```

Call the assistant endpoint:
```
curl -X POST http://localhost:8000/assist \
     -H "Content-Type: application/json" \
     -H "X-API-Key: my-test-key" \
     -d '{"prompt":"Hello, who are you?"}'
```

🛠 Tech Stack
Language: Python 3.12
Framework: FastAPI
Database: PostgreSQL
ORM: SQLAlchemy 2.x
Migrations: Alembic
LLM API: OpenAI SDK
Containerization: Docker & Docker Compose
