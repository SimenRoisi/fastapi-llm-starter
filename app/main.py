from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from sqlalchemy import select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from openai import APIError, OpenAIError

from .db import engine, get_session
from .models import Document, User, ApiUsage
from .schemas import (
    UserCreate, UserOut,
    UsageOut, UsageCreate, UsageSummary,
    AssistRequest, AssistResponse,
    DocumentCreate, DocumentOut,
)
from .llm import chat_once
from .auth import hash_api_key, verify_api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("startup")
    yield
    # shutdown
    print("shutdown")
    await engine.dispose()


app = FastAPI(title="Minimal API", lifespan=lifespan)


# --- Auth / Current user dependency ----------------------------------------- #
async def get_current_user(
    x_api_key: Annotated[str, Header(..., alias="X-API-Key")],
    session: AsyncSession = Depends(get_session),
) -> User:
    # Since API keys are hashed, we need to check each user
    # TODO: Optimize this with a more efficient lookup method
    users = (await session.execute(select(User))).scalars().all()
    
    for user in users:
        if verify_api_key(x_api_key, user.api_key):
            return user
    
    raise HTTPException(status_code=401, detail="Invalid API key")


# --- Health & root ----------------------------------------------------------- #
@app.get("/healthz")
async def healthz():
    # Simple DB liveness check
    async with engine.connect() as conn:
        await conn.exec_driver_sql("SELECT 1")
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "Minimal API", "docs": "/docs"}


# --- Users ------------------------------------------------------------------- #
@app.post("/users", response_model=UserOut, status_code=201)
async def create_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    # Hash the API key before storing
    hashed_api_key = hash_api_key(payload.api_key)
    user = User(email=payload.email, api_key=hashed_api_key)
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Email or API key already exists")
    await session.refresh(user)
    return user


@app.get("/users", response_model=list[UserOut])
async def list_users(session: AsyncSession = Depends(get_session)):
    return (await session.execute(select(User).order_by(User.id))).scalars().all()


# --- Usage logging ----------------------------------------------------------- #
@app.post("/usage", response_model=UsageOut, status_code=201)
async def record_usage(
    payload: UsageCreate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    row = ApiUsage(user_id=user.id, endpoint=payload.endpoint)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return UsageOut(id=row.id, endpoint=row.endpoint, timestamp=row.timestamp)


@app.get("/usage", response_model=list[UsageOut])
async def usage_for_current_user(
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    q = (
        select(ApiUsage)
        .where(ApiUsage.user_id == user.id)
        .order_by(desc(ApiUsage.timestamp))
        .limit(limit)
    )
    rows = (await session.execute(q)).scalars().all()
    return [
        UsageOut(id=r.id, endpoint=r.endpoint, timestamp=r.timestamp)
        for r in rows
    ]

@app.get("/usage/summary", response_model=list[UsageSummary])
async def usage_summary(
    hours: int = Query(24, ge=1, le=24 * 30),
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = (
        select(ApiUsage.endpoint, func.count().label("calls"))
        .where(ApiUsage.user_id == user.id, ApiUsage.timestamp >= cutoff)
        .group_by(ApiUsage.endpoint)
        .order_by(func.count().desc())
    )
    rows = await session.execute(q)
    return [UsageSummary(endpoint=e, calls=int(c)) for e, c in rows.all()]


# --- Assist (LLM proxy) ------------------------------------------------------ #
@app.post("/assist", response_model=AssistResponse)
async def assist(
    body: AssistRequest,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    system_prompt = (
        "Du er en hjelpsom AI-assistent for kundeservice. "
        "Svar kort, presist og hjelpsomt."
    )
    try:
        reply = await chat_once(system_prompt, body.prompt)
    except (APIError, OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

    session.add(ApiUsage(user_id=user.id, endpoint="/assist"))
    await session.commit()
    return AssistResponse(reply=reply)


# --- Documents --------------------------------------------------------------- #
@app.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    docs = (
        await session.execute(
            select(Document)
            .where(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
        )
    ).scalars().all()
    return docs


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def create_doc(
    payload: DocumentCreate,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(get_current_user),
):
    doc = Document(owner_id=user.id, title=payload.title, content=payload.content)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc
