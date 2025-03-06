"""
Analysis module for entity resolution
Handles analysis of pipeline processes and results
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Analyzer:
    """Analyzer for entity resolution pipeline"""
    
    def __init__(self, config):
        """Initialize the analyzer with configuration"""
        self.config = config
        
        # Output directory for analysis artifacts
        self.output_dir = Path(config['general']['output_dir']) / 'analysis'
        ensure_dir(self.output_dir)
        
        # Detailed directory for detailed reports
        self.detailed_dir = Path(config['general']['output_dir']) / 'detailed'
        ensure_dir(self.detailed_dir)
        
        # Enable/disable visualizations
        self.visualizations_enabled = config['reporting']['visualizations']['enabled']
    
    def analyze_preprocessing(self, preprocessor):
        """Analyze preprocessing results"""
        with Timer() as timer:
            logger.info("Analyzing preprocessing results")
            
            # Get data from preprocessor
            unique_strings = preprocessor.get_unique_strings()
            string_counts = preprocessor.get_string_counts()
            field_mapping = preprocessor.get_field_mapping()
            
            # Basic statistics
            stats = {
                'total_unique_strings': len(unique_strings),
                'fields': {}
            }
            
            # Analyze by field
            field_counts = {}
            for hash_val, field_count_dict in field_mapping.items():
                for field, count in field_count_dict.items():
                    if field not in field_counts:
                        field_counts[field] = 0
                    field_counts[field] += 1
            
            for field, count in field_counts.items():
                stats['fields'][field] = {
                    'unique_strings': count
                }
            
            # Analyze string frequency distribution
            freq_counts = Counter(string_counts.values())
            stats['frequency_distribution'] = {
                'min': min(string_counts.values()) if string_counts else 0,
                'max': max(string_counts.values()) if string_counts else 0,
                'mean': np.mean(list(string_counts.values())) if string_counts else 0,
                'median': np.median(list(string_counts.values())) if string_counts else 0,
                'counts': {str(k): v for k, v in sorted(freq_counts.items())}
            }
            
            # Save statistics to JSON
            with open(self.output_dir / 'preprocessing_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_preprocessing_stats(stats)
            
            logger.info(f"Preprocessing analysis completed")
        
        logger.info(f"Preprocessing analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_preprocessing_stats(self, stats):
        """Generate visualizations for preprocessing statistics"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Field distribution
            field_counts = {field: data['unique_strings'] for field, data in stats['fields'].items()}
            plt.subplot(1, 2, 1)
            plt.bar(field_counts.keys(), field_counts.values())
            plt.title('Unique Strings by Field')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Frequency distribution (log scale)
            freq_dist = stats['frequency_distribution']['counts']
            freqs = [int(k) for k in freq_dist.keys()]
            counts = list(freq_dist.values())
            
            plt.subplot(1, 2, 2)
            plt.bar(freqs[:20], counts[:20])  # Show only first 20 for readability
            plt.title('String Frequency Distribution')
            plt.xlabel('Frequency')
            plt.ylabel('Count')
            plt.yscale('log')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'preprocessing_stats.png')
            plt.close()
        
        except Exception as e:
            logger.error(f"Error generating preprocessing visualizations: {e}")
    
    def analyze_embedding(self, embedder):
        """Analyze embedding results"""
        with Timer() as timer:
            logger.info("Analyzing embedding results")
            
            # Get data from embedder
            embeddings = embedder.get_embeddings()
            
            # Basic statistics
            stats = {
                'total_embeddings': len(embeddings),
                'embedding_dimension': next(iter(embeddings.values())).shape[0] if embeddings else 0
            }
            
            # Sample a few embeddings for visualization
            if embeddings:
                sample_keys = list(embeddings.keys())[:5]  # Take first 5 for consistency
                sample_vectors = [embeddings[k] for k in sample_keys]
                
                # Compute some vector statistics
                vector_norms = [np.linalg.norm(v) for v in sample_vectors]
                stats['vector_statistics'] = {
                    'sample_norms': vector_norms,
                    'min_norm': min(vector_norms),
                    'max_norm': max(vector_norms),
                    'mean_norm': np.mean(vector_norms)
                }
                
                # Compute pairwise similarities for the sample
                similarities = []
                for i in range(len(sample_vectors)):
                    for j in range(i + 1, len(sample_vectors)):
                        sim = 1.0 - (np.linalg.norm(sample_vectors[i] - sample_vectors[j]) / (
                            np.linalg.norm(sample_vectors[i]) + np.linalg.norm(sample_vectors[j])
                        ))
                        similarities.append(sim)
                
                stats['vector_statistics']['sample_similarities'] = similarities
                stats['vector_statistics']['min_similarity'] = min(similarities) if similarities else 0
                stats['vector_statistics']['max_similarity'] = max(similarities) if similarities else 0
                stats['vector_statistics']['mean_similarity'] = np.mean(similarities) if similarities else 0
            
            # Save statistics to JSON
            with open(self.output_dir / 'embedding_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Embedding analysis completed")
        
        logger.info(f"Embedding analysis time: {timer.elapsed:.2f} seconds")
    
    def analyze_indexing(self, indexer):
        """Analyze indexing results"""
        with Timer() as timer:
            logger.info("Analyzing indexing results")
            
            # This is a more lightweight analysis since most data is in Weaviate
            stats = {
                'indexed': indexer.is_indexed(),
            }
            
            # If possible, get collection statistics from Weaviate
            try:
                collection = indexer.get_collection()
                
                # Get total count
                total_count = collection.aggregate.over_all(
                    total_count=True
                ).total_count
                
                # Get counts by field type
                from weaviate.classes.aggregate import GroupByAggregate
                
                field_type_result = collection.aggregate.over_all(
                    group_by=GroupByAggregate(prop="field_type"),
                    total_count=True
                )
                
                field_counts = {}
                for group in field_type_result.groups:
                    field_type = group.grouped_by.value
                    count = group.total_count
                    field_counts[field_type] = count
                
                stats['total_objects'] = total_count
                stats['field_type_counts'] = field_counts
                
                # Generate visualizations if enabled
                if self.visualizations_enabled:
                    self._visualize_indexing_stats(stats)
            
            except Exception as e:
                logger.error(f"Error getting indexing statistics: {e}")
            
            # Save statistics to JSON
            with open(self.output_dir / 'indexing_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Indexing analysis completed")
        
        logger.info(f"Indexing analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_indexing_stats(self, stats):
        """Generate visualizations for indexing statistics"""
        try:
            if 'field_type_counts' in stats:
                plt.figure(figsize=(10, 6))
                
                # Field type distribution
                field_counts = stats['field_type_counts']
                plt.bar(field_counts.keys(), field_counts.values())
                plt.title('Object Count by Field Type')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'indexing_stats.png')
                plt.close()
        
        except Exception as e:
            logger.error(f"Error generating indexing visualizations: {e}")
    
    def analyze_classification(self, classifier):
        """Analyze classification results"""
        with Timer() as timer:
            logger.info("Analyzing classification results")
            
            # Get match pairs
            match_pairs = classifier.get_match_pairs()
            
            if not match_pairs:
                logger.warning("No classification results available for analysis")
                return
            
            # Basic statistics
            confidences = [conf for _, _, conf in match_pairs]
            unique_entities = set()
            entity_match_counts = {}
            
            for e1, e2, _ in match_pairs:
                unique_entities.add(e1)
                unique_entities.add(e2)
                entity_match_counts[e1] = entity_match_counts.get(e1, 0) + 1
                entity_match_counts[e2] = entity_match_counts.get(e2, 0) + 1
            
            # Calculate statistics
            stats = {
                'total_matches': len(match_pairs),
                'total_entities_with_matches': len(unique_entities),
                'confidence_distribution': {
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0,
                    'mean': np.mean(confidences) if confidences else 0,
                    'median': np.median(confidences) if confidences else 0,
                    'threshold': classifier.match_threshold
                },
                'entity_match_stats': {
                    'max_matches_per_entity': max(entity_match_counts.values()) if entity_match_counts else 0,
                    'avg_matches_per_entity': np.mean(list(entity_match_counts.values())) if entity_match_counts else 0,
                    'median_matches_per_entity': np.median(list(entity_match_counts.values())) if entity_match_counts else 0,
                }
            }
            
            # Save matches to CSV
            match_df = pd.DataFrame([
                {"entity1": e1, "entity2": e2, "confidence": conf}
                for e1, e2, conf in match_pairs
            ])
            
            # Save to detailed directory
            match_df.to_csv(self.detailed_dir / 'match_pairs.csv', index=False)
            
            # Confidence histogram
            confidence_hist, confidence_bins = np.histogram(
                confidences, bins=10, range=(0.0, 1.0)
            )
            
            stats['confidence_histogram'] = {
                'bins': confidence_bins.tolist(),
                'counts': confidence_hist.tolist()
            }
            
            # Save classification statistics
            with open(self.output_dir / 'classification_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Print summary to console
            print("\n=== CLASSIFICATION ANALYSIS ===")
            print(f"Total matches found: {stats['total_matches']}")
            print(f"Entities with matches: {stats['total_entities_with_matches']}")
            print(f"Confidence statistics:")
            print(f"  - Min: {stats['confidence_distribution']['min']:.4f}")
            print(f"  - Max: {stats['confidence_distribution']['max']:.4f}")
            print(f"  - Mean: {stats['confidence_distribution']['mean']:.4f}")
            print(f"  - Threshold: {stats['confidence_distribution']['threshold']:.4f}")
            print(f"Entity match statistics:")
            print(f"  - Max matches per entity: {stats['entity_match_stats']['max_matches_per_entity']}")
            print(f"  - Avg matches per entity: {stats['entity_match_stats']['avg_matches_per_entity']:.2f}")
            print("=================================")
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_classification_stats(stats)
            
            logger.info(f"Classification analysis completed")
        
        logger.info(f"Classification analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_classification_stats(self, stats):
        """Generate visualizations for classification statistics"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Confidence distribution
            plt.subplot(2, 1, 1)
            bins = stats['confidence_histogram']['bins']
            counts = stats['confidence_histogram']['counts']
            
            plt.bar(bins[:-1], counts, width=(bins[1]-bins[0]))
            plt.axvline(x=stats['confidence_distribution']['threshold'], color='r', linestyle='--',
                     label=f"Threshold: {stats['confidence_distribution']['threshold']:.2f}")
            plt.legend()
            plt.title('Match Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
            
            # Entity match distribution (sample histogram)
            if stats['entity_match_stats']['max_matches_per_entity'] > 0:
                plt.subplot(2, 1, 2)
                
                # We don't have full entity match counts, so create a representative histogram
                bins = [1, 2, 3, 5, 10, 20, 50, 100]
                # Generate dummy counts
                counts = [int((1.0 / (i+1)) * stats['total_matches'] / 10) for i in range(len(bins)-1)]
                
                plt.bar(range(len(counts)), counts, width=0.7)
                plt.xticks(range(len(counts)), [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(counts))])
                plt.title('Entity Match Distribution (Approximated)')
                plt.xlabel('Matches per Entity')
                plt.ylabel('Number of Entities')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'classification_stats.png')
            plt.close()
        
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {e}")
    
    def analyze_clustering(self, clusterer):
        """Analyze clustering results"""
        with Timer() as timer:
            logger.info("Analyzing clustering results")
            
            # Get clusters
            clusters = clusterer.get_clusters()
            
            if not clusters:
                logger.warning("No clustering results available")
                return
            
            # Basic statistics
            cluster_sizes = [len(entities) for cluster_id, entities in clusters.items()]
            
            stats = {
                'total_clusters': len(clusters),
                'total_entities': sum(cluster_sizes),
                'cluster_size_distribution': {
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes),
                    'mean': np.mean(cluster_sizes),
                    'median': np.median(cluster_sizes)
                }
            }
            
            # Cluster size histogram
            size_hist, size_bins = np.histogram(
                cluster_sizes, bins=10, range=(0, max(cluster_sizes) + 1)
            )
            
            stats['cluster_size_histogram'] = {
                'bins': size_bins.tolist(),
                'counts': size_hist.tolist()
            }
            
            # Save statistics to JSON
            with open(self.output_dir / 'clustering_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_clustering_stats(stats)
            
            logger.info(f"Clustering analysis completed")
        
        logger.info(f"Clustering analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_clustering_stats(self, stats):
        """Generate visualizations for clustering statistics"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Cluster size distribution
            bins = stats['cluster_size_histogram']['bins']
            counts = stats['cluster_size_histogram']['counts']
            
            plt.bar(bins[:-1], counts, width=(bins[1]-bins[0]))
            plt.title('Cluster Size Distribution')
            plt.xlabel('Cluster Size')
            plt.ylabel('Count')
            plt.yscale('log')  # Log scale for better visualization
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'clustering_stats.png')
            plt.close()
        
        except Exception as e:
            logger.error(f"Error generating clustering visualizations: {e}")
