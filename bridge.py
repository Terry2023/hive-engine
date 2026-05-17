"""
Bridge Layer - Translation between Embedding and Symbolic Layers
Handles bidirectional conversion: embeddings <-> symbolic triples
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
from loguru import logger
from datetime import datetime

from embedding_layer import EmbeddingLayer
from symbolic_layer import SymbolicLayer

class Bridge:
    """
    Manages translation between high-dimensional embeddings and discrete symbolic triples.
    Uses FAISS for fast nearest-neighbor search.
    """
    
    def __init__(
        self,
        embedding_layer: EmbeddingLayer,
        symbolic_layer: SymbolicLayer,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the bridge.
        
        Args:
            embedding_layer: Initialized embedding layer
            symbolic_layer: Initialized symbolic layer
            confidence_threshold: Minimum similarity for valid translation
        """
        self.emb = embedding_layer
        self.sym = symbolic_layer
        self.confidence_threshold = confidence_threshold
        
        # FAISS index for fast similarity search
        self.index = None
        self.triple_registry = []  # Maps index positions to triple IDs
        
        # Translation cache (avoid re-computing)
        self.cache = {}
        
        # Calibration tracking
        self.translation_log = []
        
        logger.info(f"Bridge initialized (confidence threshold: {confidence_threshold})")
    
    def index_symbolic_layer(self):
        """
        Build FAISS index from all triples in symbolic layer.
        Call this whenever the symbolic layer changes significantly.
        """
        logger.info("Building FAISS index from symbolic layer...")
        
        # Get all triples
        triples = self.sym.export_triples()
        
        if not triples:
            logger.warning("No triples to index")
            return
        
        # Convert each triple to embedding
        triple_texts = []
        self.triple_registry = []
        
        for triple in triples:
            # Format: "subject predicate object"
            text = f"{triple['subject']} {triple['predicate']} {triple['object']}"
            triple_texts.append(text)
            self.triple_registry.append(triple['id'])
        
        # Generate embeddings
        embeddings = self.emb.encode_batch(triple_texts)
        
        # Create FAISS index
        dimension = self.emb.dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        logger.success(f"Indexed {len(triple_texts)} triples into FAISS")
    
    def embedding_to_symbolic(
        self, 
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Translate embedding to symbolic triple(s).
        
        Args:
            query_embedding: Input vector to translate
            top_k: Return top K matching triples
            
        Returns:
            List of matches with confidence scores
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call index_symbolic_layer() first.")
        
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype('float32')
        
        # Search
        similarities, indices = self.index.search(query_norm, top_k)
        
        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for insufficient results
                continue
            
            triple_id = self.triple_registry[idx]
            confidence = float(sim)
            
            # Parse triple ID back to components
            parts = triple_id.split("::")
            if len(parts) == 3:
                result = {
                    'subject': parts[0],
                    'predicate': parts[1],
                    'object': parts[2],
                    'confidence': confidence,
                    'triple_id': triple_id,
                    'meets_threshold': confidence >= self.confidence_threshold
                }
                results.append(result)
        
        # Log translation
        self.translation_log.append({
            'timestamp': datetime.now().isoformat(),
            'direction': 'embedding_to_symbolic',
            'top_confidence': results[0]['confidence'] if results else 0.0,
            'num_candidates': len(results)
        })
        
        return results
    
    def symbolic_to_embedding(
        self, 
        subject: str, 
        predicate: str, 
        obj: str
    ) -> np.ndarray:
        """
        Translate symbolic triple to embedding.
        
        Args:
            subject, predicate, obj: Triple components
            
        Returns:
            Embedding vector
        """
        # Check cache
        cache_key = f"{subject}::{predicate}::{obj}"
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self.cache[cache_key]
        
        # Generate embedding
        triple_text = f"{subject} {predicate} {obj}"
        embedding = self.emb.encode(triple_text)
        
        # Cache it
        self.cache[cache_key] = embedding
        
        # Log translation
        self.translation_log.append({
            'timestamp': datetime.now().isoformat(),
            'direction': 'symbolic_to_embedding',
            'triple': cache_key
        })
        
        return embedding
    
    def round_trip_test(
        self, 
        subject: str, 
        predicate: str, 
        obj: str
    ) -> Dict:
        """
        Test semantic preservation through round-trip translation.
        symbolic -> embedding -> symbolic
        
        Returns:
            Dictionary with fidelity metrics
        """
        logger.info(f"Round-trip test: ({subject}, {predicate}, {obj})")
        
        # Step 1: symbolic -> embedding
        original_embedding = self.symbolic_to_embedding(subject, predicate, obj)
        
        # Step 2: embedding -> symbolic
        matches = self.embedding_to_symbolic(original_embedding, top_k=1)
        
        if not matches:
            return {
                'success': False,
                'error': 'No matches found'
            }
        
        best_match = matches[0]
        
        # Check if we recovered the original
        exact_match = (
            best_match['subject'] == subject and
            best_match['predicate'] == predicate and
            best_match['object'] == obj
        )
        
        result = {
            'success': exact_match,
            'confidence': best_match['confidence'],
            'recovered_triple': (
                best_match['subject'],
                best_match['predicate'],
                best_match['object']
            ),
            'original_triple': (subject, predicate, obj),
            'meets_threshold': best_match['meets_threshold']
        }
        
        logger.info(f"  Exact match: {exact_match}, Confidence: {result['confidence']:.4f}")
        
        return result
    
    def propose_new_concept(
        self, 
        query_embedding: np.ndarray,
        agent_id: str
    ) -> Optional[Dict]:
        """
        When embedding doesn't match any symbolic concept well,
        propose a new triple for the ontology.
        
        Args:
            query_embedding: Unmatched embedding
            agent_id: Which agent is proposing
            
        Returns:
            Proposed new triple (requires validation)
        """
        matches = self.embedding_to_symbolic(query_embedding, top_k=3)
        
        # Check if top match is below threshold
        if matches and matches[0]['confidence'] >= self.confidence_threshold:
            return None  # Good match exists, no need for new concept
        
        # Analyze semantic neighbors to infer meaning
        neighbor_concepts = [m['object'] for m in matches[:3]]
        
        proposal = {
            'agent': agent_id,
            'embedding': query_embedding,
            'nearest_concepts': neighbor_concepts,
            'confidence_gap': self.confidence_threshold - (matches[0]['confidence'] if matches else 0.0),
            'status': 'pending_validation',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.warning(f"New concept proposed by {agent_id} (confidence gap: {proposal['confidence_gap']:.4f})")
        
        return proposal
    
    def calibrate_agents(
        self, 
        test_triples: List[Tuple[str, str, str]]
    ) -> Dict:
        """
        Run calibration round to detect agent drift.
        All agents translate same test embeddings and compare results.
        
        Args:
            test_triples: Standard test cases
            
        Returns:
            Calibration report
        """
        logger.info(f"Running calibration on {len(test_triples)} test cases...")
        
        results = []
        for subj, pred, obj in test_triples:
            rt = self.round_trip_test(subj, pred, obj)
            results.append(rt)
        
        # Calculate metrics
        exact_matches = sum(1 for r in results if r['success'])
        avg_confidence = np.mean([r['confidence'] for r in results])
        threshold_passes = sum(1 for r in results if r['meets_threshold'])
        
        report = {
            'total_tests': len(test_triples),
            'exact_matches': exact_matches,
            'accuracy': exact_matches / len(test_triples),
            'avg_confidence': float(avg_confidence),
            'threshold_passes': threshold_passes,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Calibration: {report['accuracy']:.2%} accuracy, {report['avg_confidence']:.4f} avg confidence")
        
        return report
    
    def get_stats(self) -> Dict:
        """Return bridge statistics."""
        return {
            'indexed_triples': len(self.triple_registry),
            'cache_size': len(self.cache),
            'total_translations': len(self.translation_log),
            'confidence_threshold': self.confidence_threshold,
            'faiss_indexed': self.index is not None
        }