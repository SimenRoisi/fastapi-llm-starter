
"""
Document processing service for RAG implementation.
Handles document chunking, text extraction, and preprocessing.
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a processed document chunk."""
    content: str
    chunk_index: int
    token_count: int
    metadata: Optional[Dict[str, Any]] = None


class DocumentProcessor:
    """
    Service for processing documents into RAG-ready chunks.
    
    Key concepts:
    - Chunking: Split large documents into smaller, semantically meaningful pieces
    - Overlap: Include some text from previous chunk to maintain context
    - Token limits: Ensure chunks fit within model context windows
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunk_size: int = 2000
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            max_chunk_size: Maximum tokens allowed per chunk
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for better processing.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text.strip()
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for more intelligent chunking.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting (you might want to use spaCy or NLTK for better results)
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_by_sentences(self, text: str) -> List[DocumentChunk]:
        """
        Create chunks by combining sentences while respecting token limits.
        This approach maintains better semantic coherence than arbitrary splitting.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of DocumentChunk objects
        """
        cleaned_text = self.clean_text(text)
        sentences = self.split_by_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds max chunk size, split it forcefully
            if sentence_tokens > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        token_count=current_tokens
                    ))
                    chunk_index += 1
                
                # Split long sentence into smaller pieces
                word_chunks = self._split_long_text(sentence)
                for word_chunk in word_chunks:
                    chunks.append(DocumentChunk(
                        content=word_chunk,
                        chunk_index=chunk_index,
                        token_count=self.count_tokens(word_chunk)
                    ))
                    chunk_index += 1
                
                current_chunk = ""
                current_tokens = 0
                continue
            
            # Check if adding this sentence would exceed chunk size
            potential_tokens = current_tokens + sentence_tokens
            
            if potential_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    token_count=current_tokens
                ))
                chunk_index += 1
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens = potential_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                token_count=current_tokens
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str, target_tokens: int) -> str:
        """
        Get the last portion of text for chunk overlap.
        
        Args:
            text: Source text
            target_tokens: Target number of tokens for overlap
            
        Returns:
            Overlap text
        """
        words = text.split()
        
        # Start from the end and work backwards
        overlap_words = []
        current_tokens = 0
        
        for word in reversed(words):
            word_tokens = self.count_tokens(word)
            if current_tokens + word_tokens > target_tokens:
                break
            overlap_words.insert(0, word)
            current_tokens += word_tokens
        
        return " ".join(overlap_words)
    
    def _split_long_text(self, text: str) -> List[str]:
        """
        Split text that's too long into smaller pieces by words.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_document(
        self, 
        content: str, 
        title: str = "", 
        source_type: str = "text"
    ) -> List[DocumentChunk]:
        """
        Main method to process a document into chunks ready for embedding.
        
        Args:
            content: Document content
            title: Document title (will be prepended to first chunk)
            source_type: Type of source (text, pdf, url, etc.)
            
        Returns:
            List of processed document chunks
        """
        # Prepend title to content for better context
        full_content = content
        if title:
            full_content = f"Title: {title}\n\n{content}"
        
        # Generate chunks
        chunks = self.chunk_by_sentences(full_content)
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata = {
                "source_type": source_type,
                "total_chunks": len(chunks),
                "has_title": bool(title)
            }
        
        return chunks


# Usage example for your reference:
"""
# Initialize processor
processor = DocumentProcessor(
    chunk_size=1000,      # Target 1000 tokens per chunk
    chunk_overlap=200,    # 200 token overlap between chunks
    max_chunk_size=2000   # Never exceed 2000 tokens
)

# Process document
document_content = "Your long document text here..."
chunks = processor.process_document(
    content=document_content,
    title="My Document Title",
    source_type="text"
)

# Each chunk contains:
# - chunk.content: The text content
# - chunk.chunk_index: Position in document (0, 1, 2...)
# - chunk.token_count: Number of tokens
# - chunk.metadata: Additional information
"""