from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text, select, desc, func, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from fastapi import Header, Query
from openai import APIError, OpenAIError
from datetime import datetime, timedelta, timezone  

from .db import engine, get_session
from .schemas import UserCreate, UserOut, UsageOut, UsageCreate, AssistRequest, AssistResponse, UsageSummary, DocumentCreate, DocumentOut
from .llm import chat_once
from .models import Document, User, ApiUsage

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: 
    print('startup')
    yield
    # shutdown:
    print('shutdown')
    await engine.dispose()


app = FastAPI(title="Minimal API", lifespan=lifespan)

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
    user = User(email=payload.email, api_key=payload.api_key)
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
    rows = (await session.execute(select(User).order_by(User.id))).scalars().all()
    return rows

@app.post("/usage", response_model=UsageOut, status_code=201)
async def record_usage(
    payload: UsageCreate, 
    session: AsyncSession = Depends(get_session),
    api_key: str = Depends(require_api_key)):     # <-- comes from X-API-Key
    
    row = ApiUsage(api_key=api_key, endpoint=payload.endpoint)
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=400, detail="api_key not found or invalid")
    await session.refresh(row)
    return row
    
@app.get("/usage/{api_key}", response_model=list[UsageOut])
async def usage_for_key(api_key: str, limit: int=100, session: AsyncSession=Depends(get_session)):
    q = (select(ApiUsage)
         .where(ApiUsage.api_key == api_key)
         .order_by(desc(ApiUsage.timestamp))
         .limit(limit))
    return (await session.execute(q)).scalars().all()

@app.get("/")
def root():
    return {"service": "Minimal API", "docs": "/docs"}


@app.post("/assist", response_model=AssistResponse)
async def assist(body: AssistRequest,
                 session: AsyncSession = Depends(get_session),
                 api_key: str = Depends(require_api_key)):
    system_prompt = "Du er en hjelpsom AI-assistent for kundeservice. Svar kort, presist og hjelpsomt."
    try:
        reply = await chat_once(system_prompt, body.prompt)
    except (APIError, OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}") from e

    session.add(ApiUsage(api_key=api_key, endpoint="/assist"))
    await session.commit()
    return AssistResponse(reply=reply)


@app.get("/usage/summary", response_model=list[UsageSummary])
async def usage_summary(
    hours: int = Query(24, ge=1, le=24*30),
    session: AsyncSession = Depends(get_session),
    api_key: str = Depends(require_api_key),
):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = (select(ApiUsage.endpoint, func.count().cast(Integer).label("calls"))
         .where(ApiUsage.api_key == api_key, ApiUsage.timestamp >= cutoff)
         .group_by(ApiUsage.endpoint)
         .order_by(func.count().desc()))
    rows = await session.execute(q)
    return [UsageSummary(endpoint=e, calls=c) for e, c in rows.all()]

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
