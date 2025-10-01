"""
RAG (Retrieval-Augmented Generation) API endpoints.
Handles document upload, processing, and RAG-enhanced chat completions.
"""

import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func

from app.db import get_async_db
from app.auth import get_current_user
from app.models import User, Document, DocumentChunk
from app.schemas import (
    DocumentCreate, DocumentOut, DocumentSummary,
    RAGChatRequest, RAGChatResponse, RAGContext,
    DocumentProcessRequest, DocumentProcessResponse,
    DocumentSearchRequest, DocumentSearchResponse
)
from app.document_processor import DocumentProcessor
from app.vector_storage import VectorStorageService
from app.llm import get_completion

# Create router for RAG endpoints
router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize services
document_processor = DocumentProcessor(
    chunk_size=1000,      # 1000 tokens per chunk
    chunk_overlap=200,    # 200 token overlap
    max_chunk_size=2000   # Never exceed 2000 tokens
)
vector_service = VectorStorageService()


@router.post("/documents", response_model=DocumentOut)
async def create_document(
    document: DocumentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Upload and create a new document.
    The document will be automatically processed into chunks for RAG.
    """
    try:
        # Create document record
        db_document = Document(
            owner_id=current_user.id,
            title=document.title,
            content=document.content,
            source_type=document.source_type,
            doc_metadata=document.doc_metadata
        )
        
        db.add(db_document)
        await db.flush()  # Get the ID
        
        # Process document into chunks
        start_time = time.time()
        chunks = document_processor.process_document(
            content=document.content,
            title=document.title,
            source_type=document.source_type
        )
        
        # Convert chunks to the format expected by vector service
        chunk_data = [
            {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count
            }
            for chunk in chunks
        ]
        
        # Store chunks with embeddings
        await vector_service.store_document_chunks(db, db_document.id, chunk_data)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        await db.commit()
        
        return DocumentOut(
            id=db_document.id,
            title=db_document.title,
            content=db_document.content,
            source_type=db_document.source_type,
            created_at=db_document.created_at,
            chunk_count=len(chunks)
        )
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating document: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentSummary])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get list of user's documents with chunk counts."""
    
    # Query documents with chunk counts
    stmt = select(
        Document.id,
        Document.title,
        Document.source_type,
        Document.created_at,
        func.count(DocumentChunk.id).label("chunk_count")
    ).outerjoin(DocumentChunk).where(
        Document.owner_id == current_user.id
    ).group_by(Document.id).order_by(Document.created_at.desc())
    
    result = await db.execute(stmt)
    documents = result.fetchall()
    
    return [
        DocumentSummary(
            id=doc.id,
            title=doc.title,
            source_type=doc.source_type,
            created_at=doc.created_at,
            chunk_count=doc.chunk_count
        )
        for doc in documents
    ]


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Get a specific document with chunk count."""
    
    # Get document
    stmt = select(Document).where(
        and_(Document.id == document_id, Document.owner_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get chunk count
    chunk_count_stmt = select(func.count(DocumentChunk.id)).where(
        DocumentChunk.document_id == document_id
    )
    chunk_result = await db.execute(chunk_count_stmt)
    chunk_count = chunk_result.scalar()
    
    return DocumentOut(
        id=document.id,
        title=document.title,
        content=document.content,
        source_type=document.source_type,
        created_at=document.created_at,
        chunk_count=chunk_count
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """Delete a document and all its chunks."""
    
    # Check if document exists and belongs to user
    stmt = select(Document).where(
        and_(Document.id == document_id, Document.owner_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Delete chunks first (or use cascade in your models)
        await vector_service.delete_document_chunks(db, document_id)
        
        # Delete document
        await db.delete(document)
        await db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat(
    request: RAGChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    RAG-enhanced chat completion.
    Searches user's documents for relevant context and generates a response.
    """
    start_time = time.time()
    
    try:
        # Search for relevant document chunks
        search_results = await vector_service.similarity_search(
            db=db,
            query=request.prompt,
            user_id=current_user.id,
            top_k=request.max_context_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        # Filter by specific documents if requested
        if request.document_ids:
            search_results = [
                result for result in search_results 
                if result.document_id in request.document_ids
            ]
        
        # Format context for LLM
        context = vector_service.format_context_for_llm(search_results, max_context_tokens=4000)
        
        # Build enhanced prompt with context
        if context and context != "No relevant context found.":
            enhanced_prompt = f"""Based on the following context from the user's documents, please answer the question.

Context:
{context}

Question: {request.prompt}

Please provide a helpful answer based on the context provided. If the context doesn't contain relevant information, please say so."""
        else:
            enhanced_prompt = f"""The user asked: {request.prompt}

I couldn't find relevant context in their uploaded documents. Please provide a general helpful response to their question."""
        
        # Get completion from LLM
        response = await get_completion(enhanced_prompt)
        
        # Format context for response
        context_used = [
            RAGContext(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_title=result.document_title,
                content=result.content,
                similarity_score=result.similarity_score,
                chunk_index=result.chunk_index
            )
            for result in search_results
        ]
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return RAGChatResponse(
            response=response,
            context_used=context_used,
            total_chunks_found=len(search_results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing RAG chat: {str(e)}"
        )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Search through documents without generating a chat response.
    Useful for exploring what content is available.
    """
    try:
        # Search for relevant chunks
        search_results = await vector_service.similarity_search(
            db=db,
            query=request.query,
            user_id=current_user.id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        # Filter by specific documents if requested
        if request.document_ids:
            search_results = [
                result for result in search_results 
                if result.document_id in request.document_ids
            ]
        
        # Convert to response format
        results = [
            RAGContext(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                document_title=result.document_title,
                content=result.content,
                similarity_score=result.similarity_score,
                chunk_index=result.chunk_index
            )
            for result in search_results
        ]
        
        return DocumentSearchResponse(
            results=results,
            total_found=len(results),
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching documents: {str(e)}"
        )


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Reprocess a document into chunks (useful if processing failed or you want to update chunk settings).
    """
    start_time = time.time()
    
    # Check if document exists and belongs to user
    stmt = select(Document).where(
        and_(Document.id == request.document_id, Document.owner_id == current_user.id)
    )
    result = await db.execute(stmt)
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    try:
        # Check if chunks already exist
        existing_chunks = await vector_service.get_document_chunks(db, request.document_id)
        
        if existing_chunks and not request.force_reprocess:
            return DocumentProcessResponse(
                document_id=request.document_id,
                chunks_created=len(existing_chunks),
                processing_time_ms=0,
                success=False,
                message="Document already processed. Use force_reprocess=true to reprocess."
            )
        
        # Delete existing chunks if reprocessing
        if existing_chunks:
            await vector_service.delete_document_chunks(db, request.document_id)
        
        # Process document into chunks
        chunks = document_processor.process_document(
            content=document.content,
            title=document.title,
            source_type=document.source_type
        )
        
        # Convert chunks to the format expected by vector service
        chunk_data = [
            {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count
            }
            for chunk in chunks
        ]
        
        # Store chunks with embeddings
        await vector_service.store_document_chunks(db, request.document_id, chunk_data)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DocumentProcessResponse(
            document_id=request.document_id,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            success=True,
            message=f"Successfully processed document into {len(chunks)} chunks."
        )
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )