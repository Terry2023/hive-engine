"""
Embedding Layer - Internal Cognition Layer for Hive-Mind Architecture
Converts text/concepts into high-dimensional vectors for direct agent-to-agent communication.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger
import torch

class EmbeddingLayer:
    """
    Handles embedding generation and similarity operations.
    Uses sentence-transformers for dense vector representations.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the embedding layer.
        
        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2, 384-dim)
            device: 'cuda' or 'cpu'. Auto-detects if None.
        """
        # Auto-detect GPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.success(f"Embedding layer initialized. Dimension: {self.dimension}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Convert text into embedding vector.
        
        Args:
            text: Input text to encode
            
        Returns:
            numpy array of shape (dimension,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        logger.debug(f"Encoded text (length {len(text)}): vector shape {embedding.shape}")
        return embedding
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts into embeddings efficiently.
        
        Args:
            texts: List of strings to encode
            
        Returns:
            numpy array of shape (num_texts, dimension)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        logger.debug(f"Batch encoded {len(texts)} texts")
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Embedding vectors
            
        Returns:
            Similarity score [0.0, 1.0] where 1.0 = identical
        """
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def find_closest(
        self, 
        query_vec: np.ndarray, 
        candidate_vecs: np.ndarray,
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Find closest vectors to a query vector.
        
        Args:
            query_vec: Query embedding (dimension,)
            candidate_vecs: Candidates array (num_candidates, dimension)
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Normalize query
        query_norm = query_vec / np.linalg.norm(query_vec)
        
        # Normalize all candidates
        candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        
        # Compute all similarities at once
        similarities = np.dot(candidate_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        logger.debug(f"Top-{top_k} matches: {results}")
        
        return results
    
    def semantic_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate semantic distance (1 - cosine_similarity).
        Useful for drift detection.
        
        Returns:
            Distance [0.0, 1.0] where 0.0 = identical
        """
        return 1.0 - self.cosine_similarity(vec1, vec2)
    
    def get_info(self) -> dict:
        """Return layer configuration info."""
        return {
            "model": self.model.get_sentence_embedding_dimension(),
            "dimension": self.dimension,
            "device": self.device
        }