"""
Symbolic Layer - Cross-Agent Coordination Layer for Hive-Mind Architecture
Manages graph-based ontology of logic triples for traceable reasoning.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from datetime import datetime
import json

class SymbolicLayer:
    """
    Manages a directed graph of symbolic knowledge triples.
    Format: (subject, predicate, object)
    Example: ("Agent1", "proposes", "ForecastMethod:Regression")
    """
    
    def __init__(self):
        """Initialize empty symbolic graph."""
        self.graph = nx.MultiDiGraph()  # Allows multiple edges between nodes
        self.triple_history = []  # Track all triples added (for audit)
        logger.info("Symbolic layer initialized")
    
    def add_triple(
        self, 
        subject: str, 
        predicate: str, 
        obj: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a symbolic triple to the graph.
        
        Args:
            subject: Source node (e.g., "Agent1")
            predicate: Relationship type (e.g., "proposes")
            obj: Target node (e.g., "Method:Regression")
            metadata: Optional extra data (timestamp, confidence, etc.)
            
        Returns:
            Triple ID for reference
        """
        if metadata is None:
            metadata = {}
        
        # Add timestamp
        metadata['timestamp'] = datetime.now().isoformat()
        
        # Add nodes if they don't exist
        if subject not in self.graph:
            self.graph.add_node(subject, node_type="entity")
        if obj not in self.graph:
            self.graph.add_node(obj, node_type="entity")
        
        # Add edge with metadata
        edge_key = self.graph.add_edge(subject, obj, predicate=predicate, **metadata)
        
        triple_id = f"{subject}::{predicate}::{obj}"
        
        # Track in history
        self.triple_history.append({
            'id': triple_id,
            'subject': subject,
            'predicate': predicate,
            'object': obj,
            'metadata': metadata
        })
        
        logger.debug(f"Added triple: ({subject}) --[{predicate}]--> ({obj})")
        
        return triple_id
    
    def query_triples(
        self, 
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query triples matching given criteria.
        
        Args:
            subject: Filter by subject (None = any)
            predicate: Filter by predicate (None = any)
            obj: Filter by object (None = any)
            
        Returns:
            List of matching triples with metadata
        """
        results = []
        
        for s, o, data in self.graph.edges(data=True):
            # Check filters
            if subject is not None and s != subject:
                continue
            if obj is not None and o != obj:
                continue
            if predicate is not None and data.get('predicate') != predicate:
                continue
            
            results.append({
                'subject': s,
                'predicate': data.get('predicate'),
                'object': o,
                'metadata': {k: v for k, v in data.items() if k != 'predicate'}
            })
        
        logger.debug(f"Query returned {len(results)} triples")
        return results
    
    def get_neighbors(self, node: str, direction: str = "out") -> List[Tuple[str, str, str]]:
        """
        Get all connected triples for a node.
        
        Args:
            node: Node to query
            direction: "out" (outgoing), "in" (incoming), or "both"
            
        Returns:
            List of (subject, predicate, object) tuples
        """
        results = []
        
        if direction in ["out", "both"]:
            for _, target, data in self.graph.out_edges(node, data=True):
                results.append((node, data.get('predicate', 'relates_to'), target))
        
        if direction in ["in", "both"]:
            for source, _, data in self.graph.in_edges(node, data=True):
                results.append((source, data.get('predicate', 'relates_to'), node))
        
        return results
    
    def find_path(self, start: str, end: str, max_hops: int = 5) -> Optional[List[str]]:
        """
        Find reasoning path between two nodes.
        
        Args:
            start: Starting node
            end: Target node
            max_hops: Maximum path length
            
        Returns:
            List of nodes in path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, start, end)
            if len(path) <= max_hops + 1:
                logger.debug(f"Found path: {' -> '.join(path)}")
                return path
            else:
                logger.debug(f"Path too long ({len(path)} nodes)")
                return None
        except nx.NetworkXNoPath:
            logger.debug(f"No path from {start} to {end}")
            return None
    
    def get_subgraph(self, center_node: str, radius: int = 1) -> nx.DiGraph:
        """
        Extract local subgraph around a node.
        
        Args:
            center_node: Node to center on
            radius: Number of hops to include
            
        Returns:
            NetworkX subgraph
        """
        if center_node not in self.graph:
            return nx.DiGraph()
        
        # Get all nodes within radius
        nodes = {center_node}
        current_layer = {center_node}
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                # Add neighbors
                next_layer.update(self.graph.successors(node))
                next_layer.update(self.graph.predecessors(node))
            nodes.update(next_layer)
            current_layer = next_layer
        
        return self.graph.subgraph(nodes)
    
    def compress_path(self, path_nodes: List[str], summary: str) -> str:
        """
        Compress a reasoning path into a summary triple.
        Useful for long-horizon memory consolidation.
        
        Args:
            path_nodes: List of nodes to compress
            summary: Human-readable summary
            
        Returns:
            New triple ID
        """
        if len(path_nodes) < 2:
            raise ValueError("Path must have at least 2 nodes")
        
        start, end = path_nodes[0], path_nodes[-1]
        compressed_id = self.add_triple(
            start,
            "summarizes_to",
            end,
            metadata={
                'summary': summary,
                'original_path_length': len(path_nodes),
                'compressed': True
            }
        )
        
        logger.info(f"Compressed {len(path_nodes)}-node path into: {summary}")
        return compressed_id
    
    def export_triples(self) -> List[Dict[str, Any]]:
        """Export all triples as JSON-serializable list."""
        return self.triple_history.copy()
    
    def import_triples(self, triples: List[Dict[str, Any]]):
        """Import triples from saved state."""
        for triple in triples:
            self.add_triple(
                triple['subject'],
                triple['predicate'],
                triple['object'],
                metadata=triple.get('metadata', {})
            )
        logger.info(f"Imported {len(triples)} triples")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return graph statistics."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_triples': self.graph.number_of_edges(),
            'num_unique_predicates': len(set(
                data.get('predicate', '') 
                for _, _, data in self.graph.edges(data=True)
            )),
            'total_history': len(self.triple_history)
        }
    
    def visualize_summary(self, max_nodes: int = 20) -> str:
        """
        Generate text-based graph summary.
        
        Returns:
            Human-readable graph description
        """
        stats = self.get_stats()
        
        summary = [
            "=== Symbolic Graph Summary ===",
            f"Nodes: {stats['num_nodes']}",
            f"Triples: {stats['num_triples']}",
            f"Unique predicates: {stats['num_unique_predicates']}",
            ""
        ]
        
        # Show sample triples
        if self.triple_history:
            summary.append("Recent triples:")
            for triple in self.triple_history[-min(5, len(self.triple_history)):]:
                summary.append(
                    f"  ({triple['subject']}) --[{triple['predicate']}]--> ({triple['object']})"
                )
        
        return "\n".join(summary)