# RAG Implementation Guide

## Overview

Your FastAPI LLM Starter now includes a complete **Retrieval-Augmented Generation (RAG)** system that allows users to:

1. Upload documents and automatically process them into searchable chunks
2. Perform semantic search across their documents
3. Get AI responses enhanced with relevant context from their documents

## Architecture

### Core Components

1. **Document Processing** (`app/document_processor.py`)
   - Intelligently chunks large documents while preserving context
   - Handles sentence-based splitting with configurable overlap
   - Token counting for optimal chunk sizes

2. **Embedding Service** (`app/embeddings.py`)
   - Converts text to semantic vectors using OpenAI's embedding API
   - Batch processing for efficiency
   - Cosine similarity calculations for relevance

3. **Vector Storage** (`app/vector_storage.py`)
   - Stores document chunks with embeddings in your existing database
   - Semantic similarity search across user documents
   - Context formatting for LLM prompts

4. **RAG Endpoints** (`app/rag_endpoints.py`)
   - Complete REST API for document management
   - RAG-enhanced chat completion
   - Document search and processing

### Database Models

- **Document**: Stores original documents with metadata
- **DocumentChunk**: Stores processed chunks with embeddings for efficient retrieval

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `tiktoken`: Token counting for OpenAI models
- `numpy`: Vector operations for similarity calculations

### 2. Run Database Migration

```bash
alembic upgrade head
```

This creates the new RAG tables (`document_chunks` with proper indexes).

### 3. Set Environment Variables

Make sure you have your OpenAI API key set:
```bash
export OPENAI_API_KEY="your-key-here"
```

## API Usage Examples

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/rag/documents" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Machine Learning Guide", 
    "content": "Machine learning is a subset of artificial intelligence...",
    "source_type": "text"
  }'
```

**What happens**: Document is automatically chunked and embedded for search.

### 2. RAG Chat (The Magic!)

```bash
curl -X POST "http://localhost:8000/rag/chat" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_context_chunks": 5,
    "similarity_threshold": 0.2
  }'
```

**Response includes**:
- AI-generated answer based on your documents
- Context chunks that were used
- Similarity scores for transparency

### 3. Search Documents

```bash
curl -X POST "http://localhost:8000/rag/search" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deep learning algorithms",
    "top_k": 10
  }'
```

### 4. List Documents

```bash
curl -X GET "http://localhost:8000/rag/documents" \
  -H "X-API-Key: your-api-key"
```

### 5. Delete Document

```bash
curl -X DELETE "http://localhost:8000/rag/documents/1" \
  -H "X-API-Key: your-api-key"
```

## How RAG Works

### 1. Document Processing Flow

```
Upload Document → Chunk into Pieces → Generate Embeddings → Store in Database
```

**Example**: A 5000-word document becomes:
- 5-7 chunks of ~1000 tokens each
- 200-token overlap between chunks for context preservation
- Each chunk gets a 1536-dimension embedding vector

### 2. Search & Response Flow

```
User Query → Generate Query Embedding → Find Similar Chunks → Format Context → LLM Response
```

**Example**: "What is machine learning?" 
- Finds chunks about ML concepts
- Provides context to the LLM
- LLM generates informed response

## Configuration Options

### Document Processor Settings

```python
processor = DocumentProcessor(
    chunk_size=1000,      # Target tokens per chunk
    chunk_overlap=200,    # Overlap between chunks  
    max_chunk_size=2000   # Never exceed this limit
)
```

### Search Parameters

- `similarity_threshold`: Minimum relevance score (0.0 to 1.0)
- `max_context_chunks`: How many chunks to include in LLM context
- `top_k`: Maximum search results to consider

## Performance Considerations

### Vector Database Scaling

**Current Setup**: Uses your PostgreSQL/SQLite with JSON columns
- ✅ **Good for**: <10k documents, simple deployment
- ❌ **Limitations**: No specialized vector indexes, slower at scale

**Production Scaling Options**:
- **Pinecone**: Managed vector database (recommended)
- **Weaviate**: Self-hosted with advanced features  
- **Qdrant**: Lightweight, Docker-friendly
- **ChromaDB**: Embedded option

### Embedding Costs

OpenAI embeddings pricing (text-embedding-3-small):
- ~$0.02 per 1M tokens
- 1000-token chunk ≈ $0.00002

**Cost Example**: 100 documents × 5 chunks each × 1000 tokens = 500k tokens = ~$0.01

### Authentication Performance

**Current Issue**: `get_current_user()` fetches all users for bcrypt verification
**Solution**: Add API key indexing (your existing todo item)

## Customization Examples

### 1. Add PDF Support

```python
# Install: pip install PyPDF2
import PyPDF2

def extract_pdf_text(pdf_content: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Use in document creation endpoint
```

### 2. Custom Chunking Strategy

```python
# For code documents
def chunk_by_functions(code: str) -> List[str]:
    # Split by function definitions
    chunks = re.split(r'\n(?=def |class )', code)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
```

### 3. Metadata Filtering

```python
# Search only recent documents
async def search_recent_documents(
    query: str, 
    days: int = 30
) -> List[VectorSearchResult]:
    cutoff_date = datetime.now() - timedelta(days=days)
    # Add date filter to search query
```

## Testing Your RAG System

### 1. Test Document Processing

```python
from app.document_processor import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.process_document(
    content="Your test document content here...",
    title="Test Doc"
)
print(f"Created {len(chunks)} chunks")
```

### 2. Test Embedding Generation

```python
from app.embeddings import EmbeddingService

service = EmbeddingService()
embedding = await service.generate_embedding("test query")
print(f"Embedding dimension: {len(embedding)}")
```

### 3. Test End-to-End

1. Upload a document via API
2. Verify chunks were created in database
3. Test search with various queries
4. Check RAG chat responses

## Common Issues & Solutions

### 1. Import Errors
**Problem**: `ImportError: No module named 'tiktoken'`
**Solution**: `pip install tiktoken numpy`

### 2. Empty Search Results  
**Problem**: No chunks found for queries
**Solutions**: 
- Check if documents were processed (chunks created)
- Lower similarity threshold
- Verify embedding generation

### 3. Slow Performance
**Solutions**:
- Add database indexes on frequently queried fields
- Implement authentication optimization
- Consider vector database for large scale

### 4. Context Too Large
**Problem**: Too much context for LLM
**Solutions**:
- Reduce `max_context_chunks`
- Implement smarter chunk selection
- Truncate context by token count

## Next Steps for Enhancement

1. **Add File Upload Support** (PDF, DOCX, TXT)
2. **Implement Conversation Memory** (chat history)
3. **Add Citation Tracking** (show which documents influenced response)
4. **Create Document Collections** (organize docs by topic)
5. **Add Async Processing** (handle large document uploads)
6. **Implement Hybrid Search** (combine keyword + semantic search)

## Production Checklist

- [ ] Add input validation for file uploads
- [ ] Implement rate limiting for embedding API calls
- [ ] Set up monitoring for embedding costs
- [ ] Add document processing queues for large files
- [ ] Configure vector database for scale
- [ ] Add document versioning
- [ ] Implement user document quotas
- [ ] Add comprehensive error handling

Your RAG system is now ready for development and testing! Start by uploading a few test documents and exploring the chat functionality.