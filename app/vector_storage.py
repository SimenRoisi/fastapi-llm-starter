"""
Vector storage and similarity search service for RAG.
Handles storing document chunks with embeddings and performing similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models import DocumentChunk, Document
from app.embeddings import EmbeddingService
import json


class VectorSearchResult:
    """Represents a search result with content and similarity score."""
    
    def __init__(
        self, 
        content: str, 
        similarity_score: float, 
        chunk_id: int, 
        document_id: int,
        document_title: str,
        chunk_index: int,
        metadata: Optional[Dict] = None
    ):
        self.content = content
        self.similarity_score = similarity_score
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.chunk_index = chunk_index
        self.metadata = metadata or {}


class VectorStorageService:
    """
    Service for storing and searching document embeddings.
    
    This implementation uses your existing PostgreSQL/SQLite database to store vectors.
    For production with large datasets, consider dedicated vector databases like:
    - Pinecone (cloud)
    - Weaviate (self-hosted)
    - Qdrant (self-hosted)
    - ChromaDB (lightweight)
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    async def store_document_chunks(
        self, 
        db: AsyncSession, 
        document_id: int, 
        chunks: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Store document chunks with their embeddings.
        
        Args:
            db: Database session
            document_id: ID of the parent document
            chunks: List of chunk data with content, index, etc.
            
        Returns:
            List of created chunk IDs
        """
        # Extract text content for embedding generation
        chunk_texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings in batch (more efficient)
        embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)
        
        chunk_ids = []
        
        for i, (chunk_data, embedding) in enumerate(zip(chunks, embeddings)):
            # Create DocumentChunk record
            db_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk_data["chunk_index"],
                content=chunk_data["content"],
                embedding=embedding,  # Store as JSON
                token_count=chunk_data.get("token_count")
            )
            
            db.add(db_chunk)
            await db.flush()  # Get the ID without committing
            chunk_ids.append(db_chunk.id)
        
        await db.commit()
        return chunk_ids
    
    async def similarity_search(
        self, 
        db: AsyncSession, 
        query: str, 
        user_id: Optional[int] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.1
    ) -> List[VectorSearchResult]:
        """
        Search for similar document chunks using semantic similarity.
        
        Args:
            db: Database session
            query: Search query text
            user_id: Optional user ID to filter documents by owner
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of VectorSearchResult objects ordered by similarity
        """
        # Generate embedding for the query
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Build query to get chunks with their document info
        query_stmt = select(
            DocumentChunk.id,
            DocumentChunk.content,
            DocumentChunk.embedding,
            DocumentChunk.chunk_index,
            DocumentChunk.document_id,
            Document.title.label("document_title"),
            Document.doc_metadata.label("document_metadata")
        ).join(Document).where(
            DocumentChunk.embedding.is_not(None)  # Only chunks with embeddings
        )
        
        # Filter by user if specified
        if user_id is not None:
            query_stmt = query_stmt.where(Document.owner_id == user_id)
        
        result = await db.execute(query_stmt)
        chunks = result.fetchall()
        
        if not chunks:
            return []
        
        # Calculate similarities
        similarities = []
        
        for chunk in chunks:
            chunk_embedding = chunk.embedding
            if not chunk_embedding:
                continue
            
            similarity = self.embedding_service.cosine_similarity(
                query_embedding, 
                chunk_embedding
            )
            
            if similarity >= similarity_threshold:
                similarities.append((chunk, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k results
        top_results = similarities[:top_k]
        
        # Convert to VectorSearchResult objects
        search_results = []
        
        for chunk, similarity in top_results:
            result = VectorSearchResult(
                content=chunk.content,
                similarity_score=similarity,
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_index=chunk.chunk_index,
                metadata=chunk.document_metadata
            )
            search_results.append(result)
        
        return search_results
    
    async def get_document_chunks(
        self, 
        db: AsyncSession, 
        document_id: int
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a specific document.
        
        Args:
            db: Database session
            document_id: ID of the document
            
        Returns:
            List of DocumentChunk objects
        """
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    async def delete_document_chunks(
        self, 
        db: AsyncSession, 
        document_id: int
    ) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            db: Database session
            document_id: ID of the document
            
        Returns:
            True if chunks were deleted
        """
        # Note: In production, you might want to use cascade deletes in your model relationships
        stmt = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        result = await db.execute(stmt)
        chunks = result.scalars().all()
        
        for chunk in chunks:
            await db.delete(chunk)
        
        await db.commit()
        return len(chunks) > 0
    
    def format_context_for_llm(
        self, 
        search_results: List[VectorSearchResult], 
        max_context_tokens: int = 4000
    ) -> str:
        """
        Format search results into context for LLM prompt.
        
        Args:
            search_results: List of search results
            max_context_tokens: Maximum tokens to include in context
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(search_results):
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            content_tokens = len(result.content) // 4
            
            if current_tokens + content_tokens > max_context_tokens:
                break
            
            # Format the context with source information
            formatted_content = (
                f"[Document: {result.document_title}, "
                f"Chunk {result.chunk_index + 1}, "
                f"Similarity: {result.similarity_score:.3f}]\n"
                f"{result.content}\n"
            )
            
            context_parts.append(formatted_content)
            current_tokens += content_tokens
        
        return "\n---\n".join(context_parts)


# Usage example for your reference:
"""
# Initialize service
vector_service = VectorStorageService()

# Store document chunks (usually done after document upload)
chunk_data = [
    {
        "content": "First chunk content...",
        "chunk_index": 0,
        "token_count": 150
    },
    {
        "content": "Second chunk content...",
        "chunk_index": 1,
        "token_count": 200
    }
]

chunk_ids = await vector_service.store_document_chunks(db, document_id=1, chunks=chunk_data)

# Search for similar content
search_results = await vector_service.similarity_search(
    db, 
    query="machine learning algorithms",
    user_id=1,  # Optional: only search user's documents
    top_k=5
)

# Format results for LLM context
context = vector_service.format_context_for_llm(search_results, max_context_tokens=3000)
"""