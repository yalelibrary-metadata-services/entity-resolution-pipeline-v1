"""
Clustering module for entity resolution
Handles entity clustering using graph-based algorithms
"""

import logging
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import jsonlines
import community as community_louvain
try:
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False

from .utils import Timer

# Configure logger
logger = logging.getLogger(__name__)

class Clusterer:
    """Clusterer for entity resolution"""
    
    def __init__(self, config):
        """Initialize the clusterer with configuration"""
        self.config = config
        self.algorithm = config['clustering']['algorithm']
        self.similarity_threshold = config['clustering']['similarity_threshold']
        self.min_cluster_size = config['clustering']['min_cluster_size']
        self.max_cluster_size = config['clustering']['max_cluster_size']
        
        # Initialize data structures
        self.graph = nx.Graph()
        self.clusters = {}
        self.entity_to_cluster = {}
        
        # Processing state
        self.clustered = False
    
    def cluster(self, match_pairs):
        """Cluster entities based on match pairs"""
        with Timer() as timer:
            logger.info(f"Starting entity clustering using {self.algorithm} algorithm")
            
            # Build graph from match pairs
            self._build_graph(match_pairs)
            
            # Apply clustering algorithm
            if self.algorithm == 'connected_components':
                self._apply_connected_components()
            elif self.algorithm == 'louvain':
                self._apply_louvain()
            elif self.algorithm == 'leiden':
                self._apply_leiden()
            else:
                logger.error(f"Unknown clustering algorithm: {self.algorithm}")
                raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")
            
            # Filter clusters
            self._filter_clusters()
            
            logger.info(f"Clustering completed: {len(self.clusters)} clusters found")
            self.clustered = True
        
        logger.info(f"Clustering time: {timer.elapsed:.2f} seconds")
        return self
    
    def _build_graph(self, match_pairs):
        """Build graph from match pairs"""
        logger.info(f"Building graph from {len(match_pairs)} match pairs")
        
        # Add nodes and edges
        for person1_id, person2_id, probability in match_pairs:
            # Skip if probability is below threshold
            if probability < self.similarity_threshold:
                continue
            
            # Add nodes if they don't exist
            if not self.graph.has_node(person1_id):
                self.graph.add_node(person1_id)
            
            if not self.graph.has_node(person2_id):
                self.graph.add_node(person2_id)
            
            # Add edge with weight = probability
            self.graph.add_edge(person1_id, person2_id, weight=probability)
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _apply_connected_components(self):
        """Apply connected components algorithm"""
        logger.info("Applying connected components algorithm")
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        # Convert to clusters
        self.clusters = {i: list(component) for i, component in enumerate(components)}
        
        logger.info(f"Found {len(self.clusters)} connected components")
    
    def _apply_louvain(self):
        """Apply Louvain community detection algorithm"""
        logger.info("Applying Louvain community detection algorithm")
        
        # Apply Louvain algorithm
        partitions = community_louvain.best_partition(self.graph, weight='weight')
        
        # Group by partition
        partition_to_nodes = {}
        for node, partition in partitions.items():
            if partition not in partition_to_nodes:
                partition_to_nodes[partition] = []
            partition_to_nodes[partition].append(node)
        
        # Convert to clusters
        self.clusters = {i: nodes for i, (_, nodes) in enumerate(partition_to_nodes.items())}
        
        logger.info(f"Found {len(self.clusters)} communities using Louvain algorithm")
    
    def _apply_leiden(self):
        """Apply Leiden community detection algorithm"""
        if not LEIDEN_AVAILABLE:
            logger.warning("Leiden algorithm not available, falling back to Louvain")
            self._apply_louvain()
            return
        
        logger.info("Applying Leiden community detection algorithm")
        
        try:
            import igraph as ig
            
            # Convert NetworkX graph to igraph
            ig_graph = ig.Graph.from_networkx(self.graph)
            
            # Get edge weights
            weights = [self.graph.get_edge_data(u, v).get('weight', 1.0) 
                      for u, v in self.graph.edges]
            
            # Apply Leiden algorithm
            partitions = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                weights=weights
            )
            
            # Convert to clusters
            self.clusters = {i: [ig_graph.vs[node]['_nx_name'] for node in partition] 
                            for i, partition in enumerate(partitions)}
            
            logger.info(f"Found {len(self.clusters)} communities using Leiden algorithm")
        
        except Exception as e:
            logger.error(f"Error applying Leiden algorithm: {e}")
            logger.warning("Falling back to Louvain algorithm")
            self._apply_louvain()
    
    def _filter_clusters(self):
        """Filter clusters based on size"""
        original_count = len(self.clusters)
        
        # Filter by size
        filtered_clusters = {}
        singleton_count = 0
        oversized_count = 0
        cluster_idx = 0
        
        for _, nodes in self.clusters.items():
            if len(nodes) < self.min_cluster_size:
                singleton_count += 1
                continue
            
            if len(nodes) > self.max_cluster_size:
                oversized_count += 1
                # TODO: Consider splitting oversized clusters
                # For now, just keep them
            
            filtered_clusters[cluster_idx] = nodes
            cluster_idx += 1
        
        self.clusters = filtered_clusters
        
        # Build entity to cluster mapping
        self.entity_to_cluster = {}
        for cluster_id, nodes in self.clusters.items():
            for node in nodes:
                self.entity_to_cluster[node] = cluster_id
        
        logger.info(f"Filtered clusters: {original_count} -> {len(self.clusters)}")
        logger.info(f"Removed {singleton_count} singletons and marked {oversized_count} oversized clusters")
    
    def save_clusters(self, output_path):
        """Save clusters to output file"""
        logger.info(f"Saving clusters to {output_path}")
        
        try:
            # Ensure directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON Lines
            with jsonlines.open(output_path, mode='w') as writer:
                for cluster_id, entities in self.clusters.items():
                    writer.write({
                        'cluster_id': cluster_id,
                        'size': len(entities),
                        'entities': entities
                    })
            
            logger.info(f"Saved {len(self.clusters)} clusters to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving clusters: {e}")
            return False
    
    def get_clusters(self):
        """Get clusters"""
        return self.clusters
    
    def get_entity_to_cluster(self):
        """Get entity to cluster mapping"""
        return self.entity_to_cluster
    
    def is_clustered(self):
        """Check if entities have been clustered"""
        return self.clustered
    
    def get_state(self):
        """Get the current state for checkpointing"""
        return {
            'clusters': self.clusters,
            'entity_to_cluster': self.entity_to_cluster,
            'clustered': self.clustered
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        self.clusters = state['clusters']
        self.entity_to_cluster = state['entity_to_cluster']
        self.clustered = state['clustered']
        
        logger.info(f"Loaded clusterer state: {len(self.clusters)} clusters")
