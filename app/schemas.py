from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    api_key: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    api_key: str
    created_at: datetime

class UsageCreate(BaseModel):
    endpoint: str
    
class UsageOut(BaseModel):
    id: int
    api_key: str
    endpoint: str
    timestamp: datetime

class UsageSummary(BaseModel):
    endpoint: str
    calls: int

class AssistRequest(BaseModel):
    prompt: str

class AssistResponse(BaseModel):
    reply: str

class DocumentCreate(BaseModel):
    title: str
    content: str

class DocumentOut(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime