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
    
    def analyze_training(self, classifier):
        """Analyze classifier training results"""
        with Timer() as timer:
            logger.info("Analyzing classifier training results")
            
            # Get metrics and feature importance
            metrics = classifier.metrics
            
            if not metrics:
                logger.warning("No training metrics available")
                return
            
            # Save metrics to JSON
            with open(self.output_dir / 'training_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_training_metrics(metrics)
            
            logger.info(f"Training analysis completed")
        
        logger.info(f"Training analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_training_metrics(self, metrics):
        """Generate visualizations for training metrics"""
        try:
            # Feature importance
            if 'feature_importance' in metrics:
                plt.figure(figsize=(10, 6))
                
                feature_importance = metrics['feature_importance']
                features = list(feature_importance.keys())
                importance = list(feature_importance.values())
                
                # Sort by importance
                sorted_idx = np.argsort(importance)
                plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx])
                plt.title('Feature Importance')
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'feature_importance.png')
                plt.close()
            
            # Training metrics
            if 'train_metrics' in metrics and isinstance(metrics['train_metrics'], list):
                train_metrics = metrics['train_metrics']
                iterations = [m['iteration'] for m in train_metrics]
                loss = [m['loss'] for m in train_metrics]
                precision = [m['precision'] for m in train_metrics]
                recall = [m['recall'] for m in train_metrics]
                f1 = [m['f1'] for m in train_metrics]
                
                plt.figure(figsize=(15, 10))
                
                # Loss curve
                plt.subplot(2, 2, 1)
                plt.plot(iterations, loss)
                plt.title('Training Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                
                # Precision curve
                plt.subplot(2, 2, 2)
                plt.plot(iterations, precision)
                plt.title('Precision')
                plt.xlabel('Iteration')
                plt.ylabel('Precision')
                
                # Recall curve
                plt.subplot(2, 2, 3)
                plt.plot(iterations, recall)
                plt.title('Recall')
                plt.xlabel('Iteration')
                plt.ylabel('Recall')
                
                # F1 curve
                plt.subplot(2, 2, 4)
                plt.plot(iterations, f1)
                plt.title('F1 Score')
                plt.xlabel('Iteration')
                plt.ylabel('F1')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'training_curves.png')
                plt.close()
        
        except Exception as e:
            logger.error(f"Error generating training visualizations: {e}")
    
    def analyze_classification(self, classifier):
        """Analyze classification results"""
        with Timer() as timer:
            logger.info("Analyzing classification results")
            
            # Get match pairs
            match_pairs = classifier.get_match_pairs()
            
            if not match_pairs:
                logger.warning("No classification results available")
                return
            
            # Basic statistics
            stats = {
                'total_matches': len(match_pairs),
                'confidence_distribution': {
                    'min': min(p for _, _, p in match_pairs),
                    'max': max(p for _, _, p in match_pairs),
                    'mean': np.mean([p for _, _, p in match_pairs]),
                    'median': np.median([p for _, _, p in match_pairs])
                }
            }
            
            # Confidence distribution
            confidence_values = [p for _, _, p in match_pairs]
            confidence_hist, confidence_bins = np.histogram(
                confidence_values, bins=10, range=(0.0, 1.0)
            )
            
            stats['confidence_histogram'] = {
                'bins': confidence_bins.tolist(),
                'counts': confidence_hist.tolist()
            }
            
            # Save statistics to JSON
            with open(self.output_dir / 'classification_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_classification_stats(stats)
            
            logger.info(f"Classification analysis completed")
        
        logger.info(f"Classification analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_classification_stats(self, stats):
        """Generate visualizations for classification statistics"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Confidence distribution
            bins = stats['confidence_histogram']['bins']
            counts = stats['confidence_histogram']['counts']
            
            plt.bar(bins[:-1], counts, width=(bins[1]-bins[0]))
            plt.title('Match Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')
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
