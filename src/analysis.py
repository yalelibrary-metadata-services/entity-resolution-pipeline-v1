"""
Enhanced Analysis module for entity resolution
Handles comprehensive analysis of pipeline processes and results
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, classification_report
)
import time

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Analyzer:
    """Enhanced analyzer for entity resolution pipeline"""
    
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
        
        # Initialize storage for analysis data
        self.classification_data = {}
        self.clustering_data = {}
        self.feature_analysis = {}
        self.misclassified_pairs = []
        
        logger.info("Enhanced analyzer initialized")
    
    def analyze_preprocessing(self, preprocessor):
        """Analyze preprocessing results with enhanced reporting"""
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
            
            # Create detailed string length analysis by field
            length_stats = {}
            for field in field_counts.keys():
                field_strings = []
                for hash_val, field_dict in field_mapping.items():
                    if field in field_dict and hash_val in unique_strings:
                        field_strings.append(unique_strings[hash_val])
                
                if field_strings:
                    lengths = [len(s) for s in field_strings]
                    length_stats[field] = {
                        'min_length': min(lengths),
                        'max_length': max(lengths),
                        'mean_length': sum(lengths) / len(lengths),
                        'median_length': sorted(lengths)[len(lengths) // 2],
                        'std_length': np.std(lengths),
                        'length_histogram': np.histogram(lengths, bins=20)[0].tolist(),
                        'length_bins': np.histogram(lengths, bins=20)[1].tolist()
                    }
            
            stats['string_length_stats'] = length_stats
            
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
            
            # Generate detailed CSV reports
            self._generate_preprocessing_csv_reports(
                unique_strings, string_counts, field_mapping
            )
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_preprocessing_stats(stats)
            
            logger.info(f"Preprocessing analysis completed")
        
        logger.info(f"Preprocessing analysis time: {timer.elapsed:.2f} seconds")
    
    def _generate_preprocessing_csv_reports(self, unique_strings, string_counts, field_mapping):
        """Generate detailed CSV reports for preprocessing stage"""
        # Create string frequency report
        freq_data = []
        for hash_val, string in unique_strings.items():
            # Get fields where this string appears
            fields = []
            for field, count in field_mapping.get(hash_val, {}).items():
                fields.append(f"{field} ({count})")
            
            freq_data.append({
                'hash': hash_val,
                'string': string,
                'frequency': string_counts.get(hash_val, 0),
                'length': len(string),
                'fields': ', '.join(fields)
            })
        
        # Create and save dataframe
        freq_df = pd.DataFrame(freq_data)
        freq_df.sort_values('frequency', ascending=False, inplace=True)
        freq_df.to_csv(self.detailed_dir / 'string_frequencies.csv', index=False)
        
        # Create field-specific string reports
        for field in set(field for mapping in field_mapping.values() for field in mapping.keys()):
            field_data = []
            for hash_val, field_dict in field_mapping.items():
                if field in field_dict and hash_val in unique_strings:
                    field_data.append({
                        'hash': hash_val,
                        'string': unique_strings[hash_val],
                        'frequency': string_counts.get(hash_val, 0),
                        'length': len(unique_strings[hash_val]),
                        'field_count': field_dict[field]
                    })
            
            if field_data:
                field_df = pd.DataFrame(field_data)
                field_df.sort_values('frequency', ascending=False, inplace=True)
                field_df.to_csv(self.detailed_dir / f'field_{field}_strings.csv', index=False)
    
    def _visualize_preprocessing_stats(self, stats):
        """Generate enhanced visualizations for preprocessing statistics"""
        try:
            # Create a multi-panel figure
            plt.figure(figsize=(20, 15))
            
            # 1. Field distribution
            plt.subplot(2, 2, 1)
            field_counts = {field: data['unique_strings'] for field, data in stats['fields'].items()}
            fields = list(field_counts.keys())
            counts = list(field_counts.values())
            
            plt.bar(fields, counts, color=sns.color_palette("viridis", len(fields)))
            plt.title('Unique Strings by Field', fontsize=14)
            plt.xlabel('Field', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 2. Frequency distribution (log scale)
            plt.subplot(2, 2, 2)
            freq_dist = stats['frequency_distribution']['counts']
            freqs = [int(k) for k in freq_dist.keys()][:20]
            counts = [freq_dist[str(f)] for f in freqs]
            
            plt.bar(freqs, counts, color='skyblue')
            plt.title('String Frequency Distribution', fontsize=14)
            plt.xlabel('Frequency', fontsize=12)
            plt.ylabel('Count (log scale)', fontsize=12)
            plt.yscale('log')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 3. String length distribution by field
            plt.subplot(2, 1, 2)
            length_stats = stats.get('string_length_stats', {})
            
            if length_stats:
                for i, (field, field_stats) in enumerate(length_stats.items()):
                    bins = field_stats['length_bins'][:-1]  # Remove last bin edge
                    hist = field_stats['length_histogram']
                    plt.plot(bins, hist, label=field, linewidth=2)
                
                plt.title('String Length Distribution by Field', fontsize=14)
                plt.xlabel('String Length (characters)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'preprocessing_analysis.png', dpi=300)
            plt.close()
            
            # Additional individual field visualizations
            for field, field_stats in length_stats.items():
                plt.figure(figsize=(10, 6))
                bins = field_stats['length_bins']
                hist = field_stats['length_histogram']
                
                plt.bar(
                    bins[:-1], 
                    hist, 
                    width=np.diff(bins), 
                    color=sns.color_palette("viridis")[0],
                    alpha=0.7
                )
                
                plt.title(f'String Length Distribution: {field}', fontsize=14)
                plt.xlabel('String Length (characters)', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.axvline(
                    field_stats['mean_length'], 
                    color='red', 
                    linestyle='--', 
                    label=f"Mean: {field_stats['mean_length']:.1f}"
                )
                plt.axvline(
                    field_stats['median_length'], 
                    color='green', 
                    linestyle='--', 
                    label=f"Median: {field_stats['median_length']:.1f}"
                )
                plt.grid(linestyle='--', alpha=0.7)
                plt.legend(fontsize=10)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'length_distribution_{field}.png', dpi=300)
                plt.close()
        
        except Exception as e:
            logger.error(f"Error generating preprocessing visualizations: {e}")
    
    def analyze_embedding(self, embedder):
        """Analyze embedding results with enhanced reporting"""
        with Timer() as timer:
            logger.info("Analyzing embedding results")
            
            # Get data from embedder
            embeddings = embedder.get_embeddings()
            
            # Basic statistics
            stats = {
                'total_embeddings': len(embeddings),
                'embedding_dimension': next(iter(embeddings.values())).shape[0] if embeddings else 0
            }
            
            # Analyze vector properties
            if embeddings:
                # Compute vector norms and statistics
                norms = [np.linalg.norm(vec) for vec in embeddings.values()]
                
                norm_stats = {
                    'min_norm': min(norms),
                    'max_norm': max(norms),
                    'mean_norm': np.mean(norms),
                    'median_norm': np.median(norms),
                    'std_norm': np.std(norms),
                    'norm_histogram': np.histogram(norms, bins=20)[0].tolist(),
                    'norm_bins': np.histogram(norms, bins=20)[1].tolist()
                }
                
                stats['vector_statistics'] = norm_stats
                
                # Analyze vector components
                components = np.vstack(list(embeddings.values()))
                component_means = np.mean(components, axis=0)
                component_stds = np.std(components, axis=0)
                
                # Get top 10 highest mean components
                top_indices = np.argsort(np.abs(component_means))[-10:]
                
                component_stats = {
                    'mean_component_value': float(np.mean(component_means)),
                    'std_component_value': float(np.mean(component_stds)),
                    'top_components': [
                        {
                            'index': int(idx),
                            'mean_value': float(component_means[idx]),
                            'std_value': float(component_stds[idx])
                        } 
                        for idx in top_indices
                    ]
                }
                
                stats['component_statistics'] = component_stats
                
                # Sample vector similarity analysis
                if len(embeddings) >= 2:
                    # Sample 100 random pairs if there are many embeddings
                    max_pairs = min(100, len(embeddings) * (len(embeddings) - 1) // 2)
                    random_pairs = np.random.choice(list(embeddings.keys()), size=(max_pairs, 2), replace=True)
                    
                    similarities = []
                    for i in range(max_pairs):
                        key1, key2 = random_pairs[i]
                        if key1 != key2:  # Ensure we're not comparing the same vector
                            vec1 = embeddings[key1]
                            vec2 = embeddings[key2]
                            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                            similarities.append(similarity)
                    
                    if similarities:
                        similarity_stats = {
                            'min_similarity': min(similarities),
                            'max_similarity': max(similarities),
                            'mean_similarity': np.mean(similarities),
                            'median_similarity': np.median(similarities),
                            'similarity_histogram': np.histogram(similarities, bins=20)[0].tolist(),
                            'similarity_bins': np.histogram(similarities, bins=20)[1].tolist()
                        }
                        
                        stats['similarity_statistics'] = similarity_stats
            
            # Save statistics to JSON
            with open(self.output_dir / 'embedding_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_embedding_stats(stats)
            
            logger.info(f"Embedding analysis completed")
        
        logger.info(f"Embedding analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_embedding_stats(self, stats):
        """Generate enhanced visualizations for embedding statistics"""
        try:
            # Create a multi-panel figure
            plt.figure(figsize=(15, 15))
            
            # 1. Vector norm distribution
            plt.subplot(2, 2, 1)
            if 'vector_statistics' in stats:
                bins = stats['vector_statistics']['norm_bins'][:-1]  # Remove last bin edge
                hist = stats['vector_statistics']['norm_histogram']
                
                plt.bar(bins, hist, width=np.diff(stats['vector_statistics']['norm_bins']), alpha=0.7)
                plt.axvline(
                    stats['vector_statistics']['mean_norm'], 
                    color='red', 
                    linestyle='--', 
                    label=f"Mean: {stats['vector_statistics']['mean_norm']:.3f}"
                )
                plt.title('Vector Norm Distribution', fontsize=14)
                plt.xlabel('Vector Norm', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                plt.grid(linestyle='--', alpha=0.7)
            
            # 2. Vector similarity distribution (if available)
            plt.subplot(2, 2, 2)
            if 'similarity_statistics' in stats:
                bins = stats['similarity_statistics']['similarity_bins'][:-1]
                hist = stats['similarity_statistics']['similarity_histogram']
                
                plt.bar(bins, hist, width=np.diff(stats['similarity_statistics']['similarity_bins']), alpha=0.7)
                plt.axvline(
                    stats['similarity_statistics']['mean_similarity'], 
                    color='red', 
                    linestyle='--', 
                    label=f"Mean: {stats['similarity_statistics']['mean_similarity']:.3f}"
                )
                plt.title('Vector Similarity Distribution', fontsize=14)
                plt.xlabel('Cosine Similarity', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                plt.grid(linestyle='--', alpha=0.7)
            
            # 3. Top component values
            plt.subplot(2, 2, 3)
            if 'component_statistics' in stats and 'top_components' in stats['component_statistics']:
                top_components = stats['component_statistics']['top_components']
                indices = [comp['index'] for comp in top_components]
                values = [comp['mean_value'] for comp in top_components]
                
                plt.bar(range(len(indices)), values, alpha=0.7)
                plt.title('Top 10 Component Mean Values', fontsize=14)
                plt.xlabel('Component Index', fontsize=12)
                plt.ylabel('Mean Value', fontsize=12)
                plt.xticks(range(len(indices)), indices)
                plt.grid(linestyle='--', alpha=0.7)
            
            # 4. Component standard deviations
            plt.subplot(2, 2, 4)
            if 'component_statistics' in stats and 'top_components' in stats['component_statistics']:
                top_components = stats['component_statistics']['top_components']
                indices = [comp['index'] for comp in top_components]
                std_values = [comp['std_value'] for comp in top_components]
                
                plt.bar(range(len(indices)), std_values, alpha=0.7, color='orange')
                plt.title('Top 10 Component Standard Deviations', fontsize=14)
                plt.xlabel('Component Index', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                plt.xticks(range(len(indices)), indices)
                plt.grid(linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'embedding_analysis.png', dpi=300)
            plt.close()
        
        except Exception as e:
            logger.error(f"Error generating embedding visualizations: {e}")
    
    def analyze_indexing(self, indexer):
        """Analyze indexing results with enhanced reporting"""
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
                
                # Get frequency statistics
                try:
                    frequency_result = collection.aggregate.over_all(
                        fields=["frequency"],
                        group_by=GroupByAggregate(prop="field_type")
                    )
                    
                    frequency_stats = {}
                    for group in frequency_result.groups:
                        field_type = group.grouped_by.value
                        if hasattr(group, 'fields') and group.fields:
                            freq = group.fields.get('frequency', {})
                            frequency_stats[field_type] = {
                                'mean': freq.get('mean', 0),
                                'sum': freq.get('sum', 0),
                                'count': freq.get('count', 0)
                            }
                    
                    stats['frequency_statistics'] = frequency_stats
                except Exception as e:
                    logger.warning(f"Could not get frequency statistics: {e}")
                
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
        """Generate enhanced visualizations for indexing statistics"""
        try:
            # Create visualization for field type counts
            if 'field_type_counts' in stats:
                plt.figure(figsize=(12, 6))
                
                field_counts = stats['field_type_counts']
                fields = list(field_counts.keys())
                counts = list(field_counts.values())
                
                colors = sns.color_palette("viridis", len(fields))
                plt.bar(fields, counts, color=colors)
                
                # Add count labels on top of each bar
                for i, v in enumerate(counts):
                    plt.text(i, v + max(counts) * 0.01, f"{v:,}", 
                            ha='center', va='bottom', fontsize=10)
                
                plt.title('Object Count by Field Type', fontsize=14)
                plt.xlabel('Field Type', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'field_type_counts.png', dpi=300)
                plt.close()
                
                # Pie chart for field distribution
                plt.figure(figsize=(10, 10))
                plt.pie(counts, labels=fields, autopct='%1.1f%%', 
                       startangle=90, colors=colors)
                plt.title('Field Type Distribution', fontsize=14)
                plt.tight_layout()
                plt.savefig(self.output_dir / 'field_type_pie.png', dpi=300)
                plt.close()
            
            # Create visualization for frequency statistics if available
            if 'frequency_statistics' in stats:
                frequency_stats = stats['frequency_statistics']
                
                if frequency_stats:
                    plt.figure(figsize=(12, 6))
                    
                    fields = list(frequency_stats.keys())
                    means = [frequency_stats[field].get('mean', 0) for field in fields]
                    
                    plt.bar(fields, means, color=sns.color_palette("viridis", len(fields)))
                    
                    # Add mean labels on top of each bar
                    for i, v in enumerate(means):
                        plt.text(i, v + max(means) * 0.01, f"{v:.2f}", 
                                ha='center', va='bottom', fontsize=10)
                    
                    plt.title('Mean Frequency by Field Type', fontsize=14)
                    plt.xlabel('Field Type', fontsize=12)
                    plt.ylabel('Mean Frequency', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.savefig(self.output_dir / 'field_frequency_stats.png', dpi=300)
                    plt.close()
        
        except Exception as e:
            logger.error(f"Error generating indexing visualizations: {e}")
    
    def analyze_classification(self, classifier, feature_extractor, X_test=None, y_test=None, feature_names=None):
        """Analyze classification results with comprehensive feature analysis
        
        Args:
            classifier: Trained classifier instance
            feature_extractor: Feature extractor instance
            X_test: Optional test set feature vectors
            y_test: Optional test set labels
            feature_names: Optional feature names
        """
        with Timer() as timer:
            logger.info("Starting comprehensive classification analysis")
            
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
                confidences, bins=20, range=(0.0, 1.0)
            )
            
            stats['confidence_histogram'] = {
                'bins': confidence_bins.tolist(),
                'counts': confidence_hist.tolist()
            }
            
            # Add feature importance if available
            if hasattr(classifier, '_get_feature_importance'):
                feature_importance = classifier._get_feature_importance()
                stats['feature_importance'] = feature_importance
            
            # Feature analysis if test data is provided
            if X_test is not None and y_test is not None and feature_names is not None:
                # Store the data for later use
                self.classification_data['X_test'] = X_test
                self.classification_data['y_test'] = y_test
                self.classification_data['feature_names'] = feature_names
                
                # Get predictions
                y_pred_prob = classifier._sigmoid(X_test.dot(classifier.weights) + classifier.bias)
                y_pred = (y_pred_prob >= classifier.match_threshold).astype(int)
                
                # Calculate metrics
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                test_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                stats['test_metrics'] = test_metrics
                
                # Create complete test results CSV
                test_results = []
                for i in range(len(X_test)):
                    result = {
                        'instance_id': i,
                        'true_label': int(y_test[i]),
                        'predicted_label': int(y_pred[i]),
                        'prediction_probability': float(y_pred_prob[i]),
                        'correctly_classified': y_test[i] == y_pred[i]
                    }
                    
                    # Add feature values
                    for j, name in enumerate(feature_names):
                        result[f"feature_{name}"] = float(X_test[i, j])
                    
                    test_results.append(result)
                
                # Convert to DataFrame and save
                test_df = pd.DataFrame(test_results)
                test_df.to_csv(self.detailed_dir / 'test_set_results.csv', index=False)
                
                # Identify misclassified examples
                misclassified = test_df[test_df['correctly_classified'] == False]
                misclassified.to_csv(self.detailed_dir / 'misclassified_examples.csv', index=False)
                
                # Store misclassified pairs
                self.misclassified_pairs = misclassified.to_dict('records')
                
                # Feature correlation analysis
                corr_matrix = np.corrcoef(X_test, rowvar=False)
                feature_correlation = {}
                
                for i, name1 in enumerate(feature_names):
                    feature_correlation[name1] = {}
                    for j, name2 in enumerate(feature_names):
                        feature_correlation[name1][name2] = float(corr_matrix[i, j])
                
                stats['feature_correlation'] = feature_correlation
                
                # Feature distribution analysis
                feature_distributions = {}
                
                for i, name in enumerate(feature_names):
                    # Get values for this feature
                    values = X_test[:, i]
                    
                    # Calculate distribution stats
                    hist, bin_edges = np.histogram(values, bins=20)
                    
                    feature_distributions[name] = {
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'histogram': hist.tolist(),
                        'bins': bin_edges.tolist(),
                        # Separate distributions for positive and negative classes
                        'positive_mean': float(np.mean(values[y_test == 1])) if any(y_test == 1) else 0,
                        'negative_mean': float(np.mean(values[y_test == 0])) if any(y_test == 0) else 0,
                        'positive_std': float(np.std(values[y_test == 1])) if any(y_test == 1) else 0,
                        'negative_std': float(np.std(values[y_test == 0])) if any(y_test == 0) else 0,
                    }
                
                stats['feature_distributions'] = feature_distributions
                
                # Store feature analysis for later use
                self.feature_analysis = {
                    'feature_importance': feature_importance,
                    'feature_correlation': feature_correlation,
                    'feature_distributions': feature_distributions
                }
            
            # Save classification statistics to JSON
            with open(self.output_dir / 'classification_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Save detailed classification metrics
            with open(self.detailed_dir / 'classification_metrics.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_classification_stats(stats, X_test, y_test, feature_names)
            
            logger.info(f"Classification analysis completed")
        
        logger.info(f"Classification analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_classification_stats(self, stats, X_test=None, y_test=None, feature_names=None):
        """Generate enhanced visualizations for classification analysis"""
        try:
            # 1. Confidence Distribution
            plt.figure(figsize=(10, 6))
            
            bins = stats['confidence_histogram']['bins'][:-1]
            counts = stats['confidence_histogram']['counts']
            
            plt.bar(bins, counts, width=np.diff(stats['confidence_histogram']['bins']), alpha=0.7)
            plt.axvline(x=stats['confidence_distribution']['threshold'], color='r', linestyle='--', 
                       label=f"Threshold: {stats['confidence_distribution']['threshold']:.2f}")
            plt.axvline(x=stats['confidence_distribution']['mean'], color='g', linestyle='--', 
                       label=f"Mean: {stats['confidence_distribution']['mean']:.2f}")
            
            plt.title('Match Confidence Distribution', fontsize=14)
            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.legend()
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'confidence_distribution.png', dpi=300)
            plt.close()
            
            # 2. Feature Importance
            if 'feature_importance' in stats:
                feature_importance = stats['feature_importance']
                
                # Sort by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                feature_names = [x[0] for x in sorted_features]
                importance_values = [x[1] for x in sorted_features]
                
                plt.figure(figsize=(12, 8))
                
                # Create horizontal bar chart
                plt.barh(feature_names, importance_values, color=sns.color_palette("viridis", len(feature_names)))
                
                plt.title('Feature Importance', fontsize=14)
                plt.xlabel('Importance Score', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'feature_importance.png', dpi=300)
                plt.close()
            
            # 3. Test Metrics
            if 'test_metrics' in stats:
                # Confusion Matrix
                cm = np.array(stats['test_metrics']['confusion_matrix'])
                plt.figure(figsize=(8, 6))
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix', fontsize=14)
                plt.xlabel('Predicted Label', fontsize=12)
                plt.ylabel('True Label', fontsize=12)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300)
                plt.close()
                
                # Metrics Summary
                plt.figure(figsize=(8, 6))
                
                metrics = [
                    stats['test_metrics']['precision'],
                    stats['test_metrics']['recall'],
                    stats['test_metrics']['f1']
                ]
                
                plt.bar(['Precision', 'Recall', 'F1 Score'], metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
                
                # Add value labels
                for i, v in enumerate(metrics):
                    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
                
                plt.title('Classification Performance Metrics', fontsize=14)
                plt.ylabel('Score', fontsize=12)
                plt.ylim(0, 1.1)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'classification_metrics.png', dpi=300)
                plt.close()
            
            # 4. Feature Correlation Matrix
            if 'feature_correlation' in stats:
                corr_data = stats['feature_correlation']
                feature_names = list(corr_data.keys())
                
                # Create correlation matrix
                corr_matrix = np.zeros((len(feature_names), len(feature_names)))
                
                for i, name1 in enumerate(feature_names):
                    for j, name2 in enumerate(feature_names):
                        corr_matrix[i, j] = corr_data[name1][name2]
                
                plt.figure(figsize=(12, 10))
                
                mask = np.zeros_like(corr_matrix, dtype=bool)
                mask[np.triu_indices_from(mask)] = True
                
                sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                           vmin=-1, vmax=1, square=True, linewidths=.5, 
                           xticklabels=feature_names, yticklabels=feature_names)
                
                plt.title('Feature Correlation Matrix', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'feature_correlation.png', dpi=300)
                plt.close()
                
                # Create a detailed feature correlation heatmap
                plt.figure(figsize=(20, 16))
                
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           vmin=-1, vmax=1, square=True, linewidths=.5, 
                           xticklabels=feature_names, yticklabels=feature_names)
                
                plt.title('Detailed Feature Correlation Matrix', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                
                plt.savefig(self.detailed_dir / 'detailed_feature_correlation.png', dpi=300)
                plt.close()
            
            # 5. Feature Distribution Plots
            if 'feature_distributions' in stats and feature_names:
                # Get top features by importance
                top_features = feature_names
                if 'feature_importance' in stats:
                    sorted_features = sorted(
                        stats['feature_importance'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    top_features = [x[0] for x in sorted_features[:10]]  # Top 10 features
                
                # Create feature distribution plots for top features
                for feature in top_features:
                    if feature in stats['feature_distributions']:
                        dist_data = stats['feature_distributions'][feature]
                        
                        plt.figure(figsize=(10, 6))
                        
                        bins = dist_data['bins'][:-1]
                        hist = dist_data['histogram']
                        
                        plt.bar(bins, hist, width=np.diff(dist_data['bins']), alpha=0.7)
                        
                        # Add mean line
                        plt.axvline(x=dist_data['mean'], color='r', linestyle='--',
                                   label=f"Mean: {dist_data['mean']:.4f}")
                        
                        # Add class means if available
                        if 'positive_mean' in dist_data and 'negative_mean' in dist_data:
                            plt.axvline(x=dist_data['positive_mean'], color='g', linestyle='-.',
                                       label=f"Positive Mean: {dist_data['positive_mean']:.4f}")
                            plt.axvline(x=dist_data['negative_mean'], color='b', linestyle=':',
                                       label=f"Negative Mean: {dist_data['negative_mean']:.4f}")
                        
                        plt.title(f'Distribution of Feature: {feature}', fontsize=14)
                        plt.xlabel('Value', fontsize=12)
                        plt.ylabel('Frequency', fontsize=12)
                        plt.legend()
                        plt.grid(linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        plt.savefig(self.output_dir / f'feature_distribution_{feature}.png', dpi=300)
                        plt.close()
                
                # Create a combined plot for class separation
                if X_test is not None and y_test is not None and len(top_features) >= 2:
                    # Get top 2 features for visualization
                    feature1 = top_features[0]
                    feature2 = top_features[1]
                    
                    # Get feature indices
                    feature1_idx = feature_names.index(feature1)
                    feature2_idx = feature_names.index(feature2)
                    
                    # Extract values
                    x = X_test[:, feature1_idx]
                    y = X_test[:, feature2_idx]
                    
                    plt.figure(figsize=(10, 8))
                    
                    # Plot scatter by class
                    plt.scatter(x[y_test==0], y[y_test==0], marker='o', label='Negative Class', alpha=0.6)
                    plt.scatter(x[y_test==1], y[y_test==1], marker='^', label='Positive Class', alpha=0.6)
                    
                    plt.title(f'Class Separation Using Top Features', fontsize=14)
                    plt.xlabel(feature1, fontsize=12)
                    plt.ylabel(feature2, fontsize=12)
                    plt.legend()
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    plt.savefig(self.output_dir / 'class_separation.png', dpi=300)
                    plt.close()
            
            # 6. ROC and Precision-Recall curves if test data is available
            if X_test is not None and y_test is not None and hasattr(self, 'classification_data'):
                # Get classifier for predictions
                weights = self.classification_data.get('classifier_weights')
                bias = self.classification_data.get('classifier_bias')
                
                if weights is not None and bias is not None:
                    # Compute predictions
                    y_scores = 1 / (1 + np.exp(-np.dot(X_test, weights) - bias))
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, y_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(10, 8))
                    
                    plt.plot(fpr, tpr, color='darkorange', lw=2, 
                            label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate', fontsize=12)
                    plt.ylabel('True Positive Rate', fontsize=12)
                    plt.title('Receiver Operating Characteristic', fontsize=14)
                    plt.legend(loc="lower right")
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    plt.savefig(self.output_dir / 'roc_curve.png', dpi=300)
                    plt.close()
                    
                    # Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(y_test, y_scores)
                    
                    plt.figure(figsize=(10, 8))
                    
                    plt.plot(recall, precision, color='blue', lw=2)
                    plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', 
                               label=f'Baseline (ratio = {sum(y_test)/len(y_test):.2f})')
                    
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall', fontsize=12)
                    plt.ylabel('Precision', fontsize=12)
                    plt.title('Precision-Recall Curve', fontsize=14)
                    plt.legend()
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    plt.savefig(self.output_dir / 'precision_recall_curve.png', dpi=300)
                    plt.close()
        
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {e}", exc_info=True)
    
    def analyze_clustering(self, clusterer):
        """Analyze clustering results with enhanced reporting"""
        with Timer() as timer:
            logger.info("Analyzing clustering results")
            
            # Get clusters
            clusters = clusterer.get_clusters()
            
            if not clusters:
                logger.warning("No clustering results available")
                return
            
            # Basic statistics
            cluster_sizes = [len(entities) for cluster_id, entities in clusters.items()]
            unique_entities = set()
            for entities in clusters.values():
                unique_entities.update(entities)
            
            stats = {
                'total_clusters': len(clusters),
                'total_entities': len(unique_entities),
                'total_assignments': sum(cluster_sizes),
                'cluster_size_distribution': {
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes),
                    'mean': np.mean(cluster_sizes),
                    'median': np.median(cluster_sizes),
                    'std': np.std(cluster_sizes)
                }
            }
            
            # Cluster size histogram
            size_hist, size_bins = np.histogram(
                cluster_sizes, bins=20, range=(1, max(cluster_sizes) + 1)
            )
            
            stats['cluster_size_histogram'] = {
                'bins': size_bins.tolist(),
                'counts': size_hist.tolist()
            }
            
            # Analyze cluster distributions
            size_distribution = Counter(cluster_sizes)
            stats['size_counts'] = {
                str(size): count for size, count in sorted(size_distribution.items())
            }
            
            # Create detailed cluster CSV report
            cluster_data = []
            for cluster_id, entities in clusters.items():
                for entity in entities:
                    cluster_data.append({
                        'cluster_id': cluster_id,
                        'entity_id': entity
                    })
            
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_csv(self.detailed_dir / 'clusters.csv', index=False)
            
            # Create cluster summary report
            cluster_summary = []
            for cluster_id, entities in clusters.items():
                cluster_summary.append({
                    'cluster_id': cluster_id,
                    'size': len(entities),
                    'entities': ','.join(entities)
                })
            
            summary_df = pd.DataFrame(cluster_summary)
            summary_df.sort_values('size', ascending=False, inplace=True)
            summary_df.to_csv(self.detailed_dir / 'cluster_summary.csv', index=False)
            
            # Top clusters analysis
            top_clusters = sorted(
                [(cluster_id, len(entities)) for cluster_id, entities in clusters.items()],
                key=lambda x: x[1], 
                reverse=True
            )[:20]  # Get top 20 clusters
            
            stats['top_clusters'] = [
                {'cluster_id': cluster_id, 'size': size}
                for cluster_id, size in top_clusters
            ]
            
            # Calculate cluster connectivity metrics
            connectivity = {}
            
            if clusterer.graph:
                # Mean degree (average number of connections per entity)
                degrees = [d for _, d in clusterer.graph.degree()]
                connectivity['mean_degree'] = np.mean(degrees) if degrees else 0
                connectivity['max_degree'] = max(degrees) if degrees else 0
                
                # Component analysis
                components = list(clusterer.graph.subgraph(c) for c in nx.connected_components(clusterer.graph))
                component_sizes = [len(c) for c in components]
                
                connectivity['components'] = len(components)
                connectivity['largest_component'] = max(component_sizes) if component_sizes else 0
                
                stats['connectivity'] = connectivity
            
            # Save statistics to JSON
            with open(self.output_dir / 'clustering_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._visualize_clustering_stats(stats)
            
            logger.info(f"Clustering analysis completed")
        
        logger.info(f"Clustering analysis time: {timer.elapsed:.2f} seconds")
    
    def _visualize_clustering_stats(self, stats):
        """Generate enhanced visualizations for clustering statistics"""
        try:
            # 1. Cluster Size Distribution
            plt.figure(figsize=(10, 6))
            
            bins = stats['cluster_size_histogram']['bins'][:-1]
            counts = stats['cluster_size_histogram']['counts']
            
            plt.bar(bins, counts, width=np.diff(stats['cluster_size_histogram']['bins']), alpha=0.7)
            plt.axvline(x=stats['cluster_size_distribution']['mean'], color='r', linestyle='--',
                       label=f"Mean: {stats['cluster_size_distribution']['mean']:.2f}")
            plt.axvline(x=stats['cluster_size_distribution']['median'], color='g', linestyle='-.',
                       label=f"Median: {stats['cluster_size_distribution']['median']:.2f}")
            
            plt.title('Cluster Size Distribution', fontsize=14)
            plt.xlabel('Cluster Size', fontsize=12)
            plt.ylabel('Number of Clusters', fontsize=12)
            plt.yscale('log')  # Log scale for better visualization
            plt.legend()
            plt.grid(linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(self.output_dir / 'cluster_size_distribution.png', dpi=300)
            plt.close()
            
            # 2. Top Clusters
            if 'top_clusters' in stats:
                plt.figure(figsize=(12, 6))
                
                top_clusters = stats['top_clusters'][:15]  # Top 15 for readability
                cluster_ids = [str(cluster['cluster_id']) for cluster in top_clusters]
                sizes = [cluster['size'] for cluster in top_clusters]
                
                plt.bar(range(len(cluster_ids)), sizes, color=sns.color_palette("viridis", len(cluster_ids)))
                
                # Add size labels
                for i, v in enumerate(sizes):
                    plt.text(i, v + max(sizes) * 0.01, str(v), ha='center', va='bottom')
                
                plt.title('Top Clusters by Size', fontsize=14)
                plt.xlabel('Cluster ID', fontsize=12)
                plt.ylabel('Size', fontsize=12)
                plt.xticks(range(len(cluster_ids)), cluster_ids, rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'top_clusters.png', dpi=300)
                plt.close()
            
            # 3. Cluster Size Frequency
            if 'size_counts' in stats:
                plt.figure(figsize=(10, 6))
                
                sizes = [int(size) for size in stats['size_counts'].keys()]
                counts = list(stats['size_counts'].values())
                
                # Sort by size
                sorted_data = sorted(zip(sizes, counts))
                sorted_sizes = [x[0] for x in sorted_data]
                sorted_counts = [x[1] for x in sorted_data]
                
                plt.bar(sorted_sizes, sorted_counts, alpha=0.7)
                
                plt.title('Frequency of Cluster Sizes', fontsize=14)
                plt.xlabel('Cluster Size', fontsize=12)
                plt.ylabel('Number of Clusters', fontsize=12)
                plt.grid(linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'cluster_size_frequency.png', dpi=300)
                plt.close()
            
            # 4. Connectivity metrics if available
            if 'connectivity' in stats:
                connectivity = stats['connectivity']
                metrics = [
                    connectivity.get('mean_degree', 0),
                    connectivity.get('components', 0)
                ]
                
                plt.figure(figsize=(8, 6))
                
                plt.bar(['Mean Degree', 'Components'], metrics, color=['#1f77b4', '#ff7f0e'])
                
                # Add value labels
                for i, v in enumerate(metrics):
                    plt.text(i, v + max(metrics) * 0.01, f"{v:.2f}", ha='center')
                
                plt.title('Graph Connectivity Metrics', fontsize=14)
                plt.ylabel('Value', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(self.output_dir / 'connectivity_metrics.png', dpi=300)
                plt.close()
        
        except Exception as e:
            logger.error(f"Error generating clustering visualizations: {e}")
    
    def generate_feature_importance_report(self):
        """Generate a detailed report on feature importance with explanations"""
        if not self.feature_analysis or 'feature_importance' not in self.feature_analysis:
            logger.warning("No feature importance data available for report")
            return
        
        try:
            # Create DataFrame for feature importance
            importance_data = self.feature_analysis['feature_importance']
            importance_df = pd.DataFrame({
                'Feature': list(importance_data.keys()),
                'Importance': list(importance_data.values())
            })
            
            # Sort by importance
            importance_df.sort_values('Importance', ascending=False, inplace=True)
            
            # Save to CSV
            importance_df.to_csv(self.detailed_dir / 'feature_importance.csv', index=False)
            
            # Add feature distribution statistics if available
            if 'feature_distributions' in self.feature_analysis:
                distributions = self.feature_analysis['feature_distributions']
                
                # Create detailed report with distribution statistics
                detailed_data = []
                
                for _, row in importance_df.iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    
                    if feature in distributions:
                        dist = distributions[feature]
                        
                        detailed_data.append({
                            'Feature': feature,
                            'Importance': importance,
                            'Min': dist.get('min', None),
                            'Max': dist.get('max', None),
                            'Mean': dist.get('mean', None),
                            'Median': dist.get('median', None),
                            'Std': dist.get('std', None),
                            'Positive_Mean': dist.get('positive_mean', None),
                            'Negative_Mean': dist.get('negative_mean', None),
                            'Mean_Difference': abs(dist.get('positive_mean', 0) - dist.get('negative_mean', 0)),
                            'Class_Separation': abs(dist.get('positive_mean', 0) - dist.get('negative_mean', 0)) / 
                                             (dist.get('positive_std', 1) + dist.get('negative_std', 1))
                                             if (dist.get('positive_std', 0) + dist.get('negative_std', 0)) > 0 else 0
                        })
                
                # Create and save detailed dataframe
                detailed_df = pd.DataFrame(detailed_data)
                detailed_df.to_csv(self.detailed_dir / 'feature_importance_detailed.csv', index=False)
            
            logger.info(f"Generated feature importance report")
        
        except Exception as e:
            logger.error(f"Error generating feature importance report: {e}")
    
    def generate_feature_correlation_report(self):
        """Generate a detailed report on feature correlations"""
        if not self.feature_analysis or 'feature_correlation' not in self.feature_analysis:
            logger.warning("No feature correlation data available for report")
            return
        
        try:
            # Create DataFrame for feature correlations
            correlation_data = self.feature_analysis['feature_correlation']
            
            # Convert nested dictionary to flat format for CSV
            corr_rows = []
            
            for feature1, correlations in correlation_data.items():
                for feature2, correlation in correlations.items():
                    if feature1 != feature2:  # Skip self-correlations
                        corr_rows.append({
                            'Feature1': feature1,
                            'Feature2': feature2,
                            'Correlation': correlation
                        })
            
            # Create DataFrame and sort by absolute correlation
            corr_df = pd.DataFrame(corr_rows)
            corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
            corr_df.sort_values('Abs_Correlation', ascending=False, inplace=True)
            
            # Save to CSV
            corr_df.to_csv(self.detailed_dir / 'feature_correlations.csv', index=False)
            
            # Create a pivot table for the correlation matrix
            pivot_df = corr_df.pivot(index='Feature1', columns='Feature2', values='Correlation')
            pivot_df.to_csv(self.detailed_dir / 'correlation_matrix.csv')
            
            logger.info(f"Generated feature correlation report")
        
        except Exception as e:
            logger.error(f"Error generating feature correlation report: {e}")
    
    def generate_misclassified_examples_report(self):
        """Generate a detailed report on misclassified examples"""
        if not self.misclassified_pairs:
            logger.warning("No misclassified examples data available for report")
            return
        
        try:
            # Create DataFrame from misclassified examples
            misclassified_df = pd.DataFrame(self.misclassified_pairs)
            
            # Add analysis column to show which features contributed to misclassification
            if 'feature_importance' in self.feature_analysis:
                importance = self.feature_analysis['feature_importance']
                feature_names = [f"feature_{name}" for name in importance.keys()]
                
                # Add analysis columns
                misclassified_df['error_type'] = misclassified_df.apply(
                    lambda row: 'False Positive' if row['predicted_label'] == 1 and row['true_label'] == 0 
                                else 'False Negative',
                    axis=1
                )
                
                # For top 5 features by importance, analyze their contribution
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for feature_name, importance_value in top_features:
                    col_name = f"feature_{feature_name}"
                    if col_name in misclassified_df.columns:
                        # Determine if this feature likely contributed to misclassification
                        # by comparing with expected behavior (high for positive, low for negative)
                        # TODO: This would need actual domain knowledge of each feature's expected behavior
                        # For now, just flag unusually high or low values
                        mean_value = misclassified_df[col_name].mean()
                        std_value = misclassified_df[col_name].std()
                        
                        misclassified_df[f"{feature_name}_analysis"] = misclassified_df.apply(
                            lambda row: 'Unusual High' if row[col_name] > mean_value + std_value
                                      else 'Unusual Low' if row[col_name] < mean_value - std_value
                                      else 'Normal',
                            axis=1
                        )
            
            # Save to CSV
            misclassified_df.to_csv(self.detailed_dir / 'misclassified_examples_detailed.csv', index=False)
            
            # Create summary by error type
            error_summary = misclassified_df['error_type'].value_counts().reset_index()
            error_summary.columns = ['Error Type', 'Count']
            error_summary.to_csv(self.detailed_dir / 'misclassification_summary.csv', index=False)
            
            logger.info(f"Generated misclassified examples report")
        
        except Exception as e:
            logger.error(f"Error generating misclassified examples report: {e}")
    
    def generate_all_reports(self):
        """Generate all enhanced reports"""
        logger.info("Generating all enhanced reports")
        
        self.generate_feature_importance_report()
        self.generate_feature_correlation_report()
        self.generate_misclassified_examples_report()
        
        logger.info("All enhanced reports generated")
