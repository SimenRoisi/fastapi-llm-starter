from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import List, Optional, Dict, Any

class UserCreate(BaseModel):
    email: EmailStr
    api_key: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

class UsageCreate(BaseModel):
    endpoint: str
    
class UsageOut(BaseModel):
    id: int
    endpoint: str
    timestamp: datetime

class UsageSummary(BaseModel):
    endpoint: str
    calls: int

class AssistRequest(BaseModel):
    prompt: str

class AssistResponse(BaseModel):
    reply: str

# Document schemas
class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Document title")
    content: str = Field(..., min_length=10, description="Document content")
    source_type: str = Field(default="text", description="Type of document source")
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class DocumentOut(BaseModel):
    id: int
    title: str
    content: str
    source_type: str
    created_at: datetime
    chunk_count: Optional[int] = None  # Number of chunks created

class DocumentSummary(BaseModel):
    """Lightweight document info without full content."""
    id: int
    title: str
    source_type: str
    created_at: datetime
    chunk_count: Optional[int] = None

# RAG-specific schemas
class RAGChatRequest(BaseModel):
    """Request for RAG-enhanced chat completion."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="User question or prompt")
    document_ids: Optional[List[int]] = Field(
        default=None, 
        description="Optional list of document IDs to search within. If None, searches all user documents"
    )
    max_context_chunks: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="Maximum number of relevant chunks to include in context"
    )
    similarity_threshold: float = Field(
        default=0.1, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score for relevant chunks"
    )

class RAGContext(BaseModel):
    """Information about retrieved context."""
    chunk_id: int
    document_id: int
    document_title: str
    content: str
    similarity_score: float
    chunk_index: int

class RAGChatResponse(BaseModel):
    """Response from RAG-enhanced chat."""
    response: str
    context_used: List[RAGContext]
    total_chunks_found: int
    processing_time_ms: Optional[int] = None

class DocumentProcessRequest(BaseModel):
    """Request to process/reprocess a document for RAG."""
    document_id: int
    force_reprocess: bool = Field(
        default=False, 
        description="Force reprocessing even if chunks already exist"
    )

class DocumentProcessResponse(BaseModel):
    """Response from document processing."""
    document_id: int
    chunks_created: int
    processing_time_ms: int
    success: bool
    message: str

class DocumentSearchRequest(BaseModel):
    """Request to search through documents without chat completion."""
    query: str = Field(..., min_length=1, max_length=500)
    document_ids: Optional[List[int]] = None
    top_k: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

class DocumentSearchResponse(BaseModel):
    """Response from document search."""
    results: List[RAGContext]
    total_found: int
    query: str