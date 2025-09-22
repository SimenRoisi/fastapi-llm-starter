from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from fastapi import Header, Query
from openai import APIError, OpenAIError

from .db import engine, get_session
from .schemas import UserCreate, UserOut, UsageOut, UsageCreate, AssistRequest, AssistResponse, UsageSummary, DocumentCreate, DocumentOut
from .llm import chat_once
from .models import Document, User, ApiUsage

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 
    yield
    # shutdown:
    await engine.dispose()


app = FastAPI(title="Minimal API")

async def require_api_key(
        x_api_key: str = Header(..., alias="X-API-Key"),
        session: AsyncSession = Depends(get_session),
) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail = "Missing X-API key")
    res = await session.execute(
        text("SELECT 1 FROM users WHERE api_key = :k"),
        {"k": x_api_key},
    )
    if res.scalar() is None:
        raise HTTPException(status_code=401, detail = "Invalid API key")
    return x_api_key

@app.get("/healthz")
async def healthz():
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ok"}

@app.post("/users", response_model=UserOut, status_code=201)
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    try:
        res = await session.execute(
            text("""
                 INSERT INTO users (email, api_key)
                 VALUES (:email, :api_key)
                 RETURNING id, email, api_key, created_at
            """),
            {"email": payload.email, "api_key": payload.api_key},
        )
        row = res.mappings().one()
        await session.commit()
        return row
    except IntegrityError:
        await session.rollback()
        # Unique constraints on email/api_key will trigger this
        raise HTTPException(status_code=409, detail="Email or API key already exists")

@app.get("/users", response_model=list[UserOut])
async def list_users(session: AsyncSession = Depends(get_session)):
    res = await session.execute(
        text("SELECT id, email, api_key, created_at FROM users ORDER BY id")
    )
    return list(res.mappings().all())

@app.post("/usage", response_model=UsageOut, status_code=201)
async def record_usage(
    payload: UsageCreate, 
    session: AsyncSession = Depends(get_session),
    api_key: str = Depends(require_api_key),     # <-- comes from X-API-Key
):
    try:
        res = await session.execute(
            text("""
                INSERT INTO api_usage (api_key, endpoint)
                VALUES (:api_key, :endpoint)
                RETURNING id, api_key, endpoint, timestamp
            """),
        {"api_key": api_key, "endpoint": payload.endpoint},
        )
        row = res.mappings().one()
        await session.commit()
        return row
    except IntegrityError:
        await session.rollback()
        # likely a foreign key violation if api_key doesn't exist
        raise HTTPException(status_code=400, detail="api_key not found or invalid")
    
@app.get("/usage/{api_key}", response_model=list[UsageOut])
async def usage_for_key(api_key: str, limit: int=100, session: AsyncSession=Depends(get_session)):
    res = await session.execute(
        text("""
            SELECT id, api_key, endpoint, timestamp
            FROM api_usage
            WHERE api_key = :api_key
            ORDER BY timestamp DESC
            LIMIT :limit
        """),
        {"api_key": api_key, "limit": limit},
    )
    return list(res.mappings().all())

@app.get("/")
def root():
    return {"service": "Minimal API", "docs": "/docs"}


@app.post("/assist", response_model=AssistResponse)
async def assist(
        body: AssistRequest,
        session: AsyncSession = Depends(get_session),
        api_key: str = Depends(require_api_key),
):
    """
    Enkel LLM-proxy: tar inn prompt, returnerer svar.
    Logger bruk via eksisterende /usage-logikk
    """

    system_prompt = (
        "Du er en hjelpsom AI-assistent for kundeservice. "
        "Svar kort, presist og hjelpsomt."
    )
    try:
        reply = await chat_once(system_prompt, body.prompt)
    except (APIError, OpenAIError) as e:
        # Surface a clean 502 instead of a 500
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e
    except Exception as e:
        # Catch-all so you see a useful message
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e
    
     # eksplisitt logg bruken i api_usage-tabellen:
    await session.execute(
        text("INSERT INTO api_usage (api_key, endpoint) VALUES (:k, :e)"),
        {"k": api_key, "e": "/assist"},
    )
    await session.commit()
    return AssistResponse(reply=reply)

@app.get("/usage/summary", response_model=list[UsageSummary])
async def usage_summary(
    hours: int = Query(24, ge=1, le=24*30),
    session: AsyncSession = Depends(get_session),
    api_key: str = Depends(require_api_key),
):
    q = text("""
        SELECT endpoint, COUNT(*)::int AS calls
        FROM public.api_usage
        WHERE api_key = :k
        GROUP BY endpoint
        ORDER BY calls DESC
    """)
    res = await session.execute(q, {"k": api_key, "hours": hours})

    # NB: bruk posisjons-unpacking fra SQLAlchemy Row,
    # og kast det om til Pydantic-objekter (eller dicts).
    rows = res.all()
    print("DEBUG summary rows:", rows)  # se i `docker compose logs -f api`
    return [UsageSummary(endpoint=endpoint, calls=calls) for (endpoint, calls) in rows]

@app.get("/documents", response_model=list[DocumentOut])
async def list_documents(session: AsyncSession = Depends(get_session),
                         api_key: str = Depends(require_api_key)):
    # look up user id by api_key
    user_id = await session.scalar(select(User.id).where(User.api_key == api_key))
    if user_id is None:
        # shouldn't happen because require api key just checked, but just in case
        raise HTTPException(status_code=404, detail="User not found")
    
    docs = (await session.execute(
        select(Document)
        .where(Document.owner_id == user_id)
        .order_by(Document.created_at.desc())
    )).scalars().all()

    return docs
    

@app.post("/documents", response_model=DocumentOut, status_code=201)
async def create_doc(
    payload: DocumentCreate,
    session: AsyncSession = Depends(get_session),
    api_key: str = Depends(require_api_key)
):
    user_id = await session.scalar(select(User.id).where(User.api_key == api_key))
    doc = Document(owner_id=user_id, title=payload.title, content=payload.content)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc
