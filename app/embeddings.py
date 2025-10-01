"""
Embedding service for RAG implementation.
Handles text-to-vector conversion using OpenAI's embeddings API.
"""

import asyncio
import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI
import os
import tiktoken

# Singleton OpenAI client for efficiency
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """Get singleton OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


class EmbeddingService:
    """
    Service for generating and working with text embeddings.
    
    Key concepts:
    - Embeddings: Dense vector representations of text that capture semantic meaning
    - Similarity: Closer vectors = more similar content
    - Model: text-embedding-3-small (1536 dimensions, cost-efficient)
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = get_openai_client()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # Compatible tokenizer
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector (1536 dimensions)
        """
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.
        More efficient for processing multiple chunks.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for cost estimation."""
        return len(self.tokenizer.encode(text))
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Returns:
            Similarity score between -1 and 1 (1 = identical, 0 = unrelated, -1 = opposite)
        """
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Avoid division by zero
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    @staticmethod
    def find_most_similar(
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: The query vector
            candidate_embeddings: List of vectors to search through
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = EmbeddingService.cosine_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Usage example for your reference:
"""
# Initialize service
embedding_service = EmbeddingService()

# Generate single embedding
text = "This is a document about machine learning"
embedding = await embedding_service.generate_embedding(text)

# Generate batch embeddings (more efficient)
texts = ["First chunk", "Second chunk", "Third chunk"]
embeddings = await embedding_service.generate_embeddings_batch(texts)

# Find similar content
query = "machine learning algorithms"
query_embedding = await embedding_service.generate_embedding(query)
similar_indices = EmbeddingService.find_most_similar(query_embedding, embeddings)
"""