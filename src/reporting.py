"""
Enhanced Reporting module for entity resolution
Handles generation of comprehensive reports and visualizations with a focus on feature analysis
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import jsonlines
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, classification_report
)

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Reporter:
    """Enhanced reporter for entity resolution pipeline with comprehensive feature analysis"""
    
    def __init__(self, config):
        """Initialize the reporter with configuration"""
        self.config = config
        self.output_formats = config['reporting'].get('output_format', ['json', 'csv'])
        self.detailed_metrics = config['reporting'].get('detailed_metrics', True)
        self.save_misclassified = config['reporting'].get('save_misclassified', True)
        self.visualizations_enabled = config['reporting'].get('visualizations', {}).get('enabled', True)
        
        # Output directories
        self.output_dir = Path(config['general']['output_dir'])
        self.detailed_dir = self.output_dir / 'detailed'
        self.visualizations_dir = self.output_dir / 'visualizations'
        self.feature_dir = self.output_dir / 'feature_analysis'
        
        # Ensure directories exist
        ensure_dir(self.output_dir)
        ensure_dir(self.detailed_dir)
        ensure_dir(self.visualizations_dir)
        ensure_dir(self.feature_dir)
        
        # Store state for later reference
        self.classifier_state = {}
        
        logger.info("Enhanced reporter initialized")
    
    def _ensure_numpy_arrays(self):
        """Ensure all data from classifier state is converted to NumPy arrays"""
        import numpy as np
        
        # Convert test data if it exists
        if 'test_data' in self.classifier_state:
            if self.classifier_state['test_data'] is not None:
                self.classifier_state['test_data'] = np.array(self.classifier_state['test_data'])
        
        # Convert test labels if they exist
        if 'test_labels' in self.classifier_state:
            if self.classifier_state['test_labels'] is not None:
                self.classifier_state['test_labels'] = np.array(self.classifier_state['test_labels'])
        
        # Convert test predictions if they exist
        if 'test_predictions' in self.classifier_state:
            if self.classifier_state['test_predictions'] is not None:
                self.classifier_state['test_predictions'] = np.array(self.classifier_state['test_predictions'])
                
        # Convert test scores if they exist
        if 'test_scores' in self.classifier_state:
            if self.classifier_state['test_scores'] is not None:
                self.classifier_state['test_scores'] = np.array(self.classifier_state['test_scores'])
        
        logger.info("Converted classifier state data to NumPy arrays where needed")

    def generate_reports(self, preprocessor, embedder, indexer, classifier, clusterer, output_dir=None):
        """Generate comprehensive reports for pipeline results"""
        with Timer() as timer:
            # Update output directory if provided
            if output_dir:
                self.output_dir = Path(output_dir)
                self.detailed_dir = self.output_dir / 'detailed'
                self.visualizations_dir = self.output_dir / 'visualizations'
                self.feature_dir = self.output_dir / 'feature_analysis'
                ensure_dir(self.output_dir)
                ensure_dir(self.detailed_dir)
                ensure_dir(self.visualizations_dir)
                ensure_dir(self.feature_dir)
            
            logger.info(f"Generating comprehensive reports in {self.output_dir}")
            
            # Store classifier state if available
            if hasattr(classifier, 'get_state'):
                self.classifier_state = classifier.get_state()
                # Convert lists to NumPy arrays
                self._ensure_numpy_arrays()
            
            # Generate summary report
            self._generate_summary_report(
                preprocessor, embedder, indexer, classifier, clusterer
            )
            
            # Generate detailed reports
            self._generate_detailed_reports(
                preprocessor, embedder, indexer, classifier, clusterer
            )
            
            # Generate visualization reports
            if self.visualizations_enabled:
                self._generate_visualization_reports(
                    preprocessor, embedder, indexer, classifier, clusterer
                )
            
            # Generate feature analysis reports
            self._generate_feature_analysis_reports(classifier)

            # Generate RFE reports if classifier has feature_extractor with RFE
            if hasattr(classifier, 'feature_extractor') and classifier.feature_extractor.rfe_enabled:
                self.generate_rfe_reports(classifier.feature_extractor)
            
            # Generate test set reports if available
            if 'test_data' in self.classifier_state and 'test_labels' in self.classifier_state:
                self.generate_test_set_csv_report(
                    self.classifier_state['test_data'],
                    self.classifier_state['test_labels'],
                    self.classifier_state.get('test_predictions', []),
                    self.classifier_state.get('feature_names', []),
                    preprocessor
                )
            
            # Generate misclassification analysis
            if self.save_misclassified:
                self._generate_misclassification_reports(classifier)
            
            # Generate feature distribution reports if test data available
            if 'test_data' in self.classifier_state and 'test_labels' in self.classifier_state:
                self.generate_feature_distribution_reports(
                    self.classifier_state['test_data'],
                    self.classifier_state['test_labels'],
                    self.classifier_state.get('feature_names', [])
                )
            
            # Generate feature correlation reports if available
            if 'feature_correlation' in self.classifier_state:
                self._generate_feature_correlation_reports_from_state()
            
            logger.info("Comprehensive report generation completed")
        
        logger.info(f"Report generation completed in {timer.elapsed:.2f} seconds")
    
    def _generate_summary_report(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate comprehensive summary report"""
        logger.info("Generating summary report")
        
        # Collect summary data
        summary = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing': {
                'unique_strings': len(preprocessor.get_unique_strings()) if hasattr(preprocessor, 'get_unique_strings') else 0,
                'records': len(preprocessor.get_record_field_hashes()) if hasattr(preprocessor, 'get_record_field_hashes') else 0,
                'person_ids': len(preprocessor.get_all_person_ids()) if hasattr(preprocessor, 'get_all_person_ids') else 0
            },
            'embedding': {
                'embeddings': len(embedder.get_embeddings()) if hasattr(embedder, 'get_embeddings') else 0,
                'dimension': self.config.get('openai', {}).get('embedding_dim', 1536)
            },
            'indexing': {
                'indexed': indexer.is_indexed() if hasattr(indexer, 'is_indexed') else False
            },
            'classification': {
                'matches': len(classifier.get_match_pairs()) if hasattr(classifier, 'get_match_pairs') else 0,
                'threshold': self.config.get('classification', {}).get('match_threshold', 0.5)
            },
            'clustering': {
                'clusters': len(clusterer.get_clusters()) if hasattr(clusterer, 'get_clusters') else 0,
                'algorithm': self.config.get('clustering', {}).get('algorithm', 'unknown')
            },
            'config': {
                'mode': self.config.get('general', {}).get('mode', 'unknown'),
                'openai_model': self.config.get('openai', {}).get('embedding_model', 'unknown')
            }
        }
        
        # Add classifier metrics if available in state
        if 'metrics' in self.classifier_state:
            metrics = self.classifier_state['metrics']
            if isinstance(metrics, dict):
                summary['classification']['metrics'] = {
                    'precision': metrics.get('precision', None),
                    'recall': metrics.get('recall', None),
                    'f1': metrics.get('f1', None),
                    'accuracy': metrics.get('accuracy', None)
                }
        
        # Add feature information if available
        if 'feature_names' in self.classifier_state:
            summary['features'] = {
                'total_features': len(self.classifier_state['feature_names']),
                'feature_list': self.classifier_state['feature_names']
            }
        
        # Save summary report in requested formats
        for format_type in self.output_formats:
            if format_type == 'json':
                with open(self.output_dir / 'summary_report.json', 'w') as f:
                    json.dump(summary, f, indent=2)
            
            elif format_type == 'csv':
                # Flatten nested dictionary for CSV
                flat_summary = self._flatten_dict(summary)
                
                with open(self.output_dir / 'summary_report.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['key', 'value'])
                    for key, value in flat_summary.items():
                        if isinstance(value, list):
                            writer.writerow([key, ','.join(map(str, value))])
                        else:
                            writer.writerow([key, value])
        
        logger.info("Summary report generated")
    
    def _generate_detailed_reports(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate detailed reports with full data"""
        logger.info("Generating detailed reports")
        
        # 1. Preprocessing details
        self._generate_preprocessing_reports(preprocessor)
        
        # 2. Match pairs details
        self._generate_match_pair_reports(classifier)
        
        # 3. Clustering details
        self._generate_clustering_reports(clusterer)
        
        # 4. Model weights and parameters
        self._generate_model_parameter_reports(classifier)
        
        logger.info("Detailed reports generated")
    
    def _generate_preprocessing_reports(self, preprocessor):
        """Generate detailed preprocessing reports"""
        try:
            # Get preprocessing data
            if not hasattr(preprocessor, 'get_unique_strings'):
                logger.warning("Preprocessor does not have get_unique_strings method")
                return
            
            unique_strings = preprocessor.get_unique_strings()
            string_counts = preprocessor.get_string_counts() if hasattr(preprocessor, 'get_string_counts') else {}
            field_mapping = preprocessor.get_field_mapping() if hasattr(preprocessor, 'get_field_mapping') else {}
            
            if not unique_strings:
                logger.warning("No unique strings data available")
                return
            
            # Generate report of unique strings with frequencies
            string_data = []
            for hash_val, string in unique_strings.items():
                # Get fields where this string appears
                fields = []
                for field, count in field_mapping.get(hash_val, {}).items():
                    fields.append(f"{field} ({count})")
                
                string_data.append({
                    'hash': hash_val,
                    'string': string,
                    'length': len(string),
                    'frequency': string_counts.get(hash_val, 0),
                    'fields': ', '.join(fields)
                })
            
            if string_data:
                strings_df = pd.DataFrame(string_data)
                strings_df.sort_values('frequency', ascending=False, inplace=True)
                strings_df.to_csv(self.detailed_dir / 'unique_strings.csv', index=False)
                
                # Also save a JSON version for programmatic access
                with open(self.detailed_dir / 'unique_strings.json', 'w') as f:
                    json.dump(string_data[:1000], f, indent=2)  # Limit to 1000 for reasonable file size
            
            # Generate field-specific reports
            field_types = set()
            for mapping in field_mapping.values():
                field_types.update(mapping.keys())
            
            for field in field_types:
                field_data = []
                for hash_val, field_dict in field_mapping.items():
                    if field in field_dict and hash_val in unique_strings:
                        field_data.append({
                            'hash': hash_val,
                            'string': unique_strings[hash_val],
                            'length': len(unique_strings[hash_val]),
                            'frequency': string_counts.get(hash_val, 0),
                            'field_count': field_dict[field]
                        })
                
                if field_data:
                    field_df = pd.DataFrame(field_data)
                    field_df.sort_values('frequency', ascending=False, inplace=True)
                    field_df.to_csv(self.detailed_dir / f'field_{field}_strings.csv', index=False)
            
            # Generate field statistics summary
            field_stats = {}
            for field in field_types:
                field_strings = []
                for hash_val, field_dict in field_mapping.items():
                    if field in field_dict and hash_val in unique_strings:
                        field_strings.append(unique_strings[hash_val])
                
                if field_strings:
                    lengths = [len(s) for s in field_strings]
                    field_stats[field] = {
                        'count': len(field_strings),
                        'min_length': min(lengths),
                        'max_length': max(lengths),
                        'mean_length': sum(lengths) / len(lengths),
                        'median_length': sorted(lengths)[len(lengths) // 2],
                        'std_length': np.std(lengths)
                    }
            
            # Save field statistics to JSON and CSV
            with open(self.detailed_dir / 'field_statistics.json', 'w') as f:
                json.dump(field_stats, f, indent=2)
            
            # Flatten for CSV
            field_stats_data = []
            for field, stats in field_stats.items():
                row = {'field': field}
                row.update(stats)
                field_stats_data.append(row)
            
            if field_stats_data:
                field_stats_df = pd.DataFrame(field_stats_data)
                field_stats_df.to_csv(self.detailed_dir / 'field_statistics.csv', index=False)
        
        except Exception as e:
            logger.error(f"Error generating preprocessing reports: {e}", exc_info=True)
    
    def _generate_match_pair_reports(self, classifier):
        """Generate detailed match pair reports"""
        try:
            if not hasattr(classifier, 'get_match_pairs'):
                logger.warning("Classifier does not have get_match_pairs method")
                return
            
            match_pairs = classifier.get_match_pairs()
            
            if not match_pairs:
                logger.warning("No match pairs available")
                return
            
            # Create comprehensive match pairs CSV
            match_data = []
            for e1, e2, conf in match_pairs:
                match_data.append({
                    'entity1': e1,
                    'entity2': e2,
                    'confidence': conf,
                    'threshold': classifier.match_threshold if hasattr(classifier, 'match_threshold') else 0.5
                })
            
            match_df = pd.DataFrame(match_data)
            match_df.sort_values('confidence', ascending=False, inplace=True)
            match_df.to_csv(self.detailed_dir / 'match_pairs.csv', index=False)
            
            # Create JSON version for programmatic access
            with open(self.detailed_dir / 'match_pairs.json', 'w') as f:
                json.dump(match_data[:1000], f, indent=2)  # Limit to 1000 for reasonable file size
            
            # Create confidence histogram data
            confidence_values = [conf for _, _, conf in match_pairs]
            hist, bin_edges = np.histogram(confidence_values, bins=20, range=(0, 1))
            
            histogram_data = []
            for i in range(len(hist)):
                histogram_data.append({
                    'bin_start': bin_edges[i],
                    'bin_end': bin_edges[i+1],
                    'count': hist[i]
                })
            
            # Save confidence histogram to CSV
            histogram_df = pd.DataFrame(histogram_data)
            histogram_df.to_csv(self.detailed_dir / 'confidence_histogram.csv', index=False)
            
            # Create confidence statistics
            confidence_stats = {
                'min': min(confidence_values) if confidence_values else 0,
                'max': max(confidence_values) if confidence_values else 0,
                'mean': np.mean(confidence_values) if confidence_values else 0,
                'median': np.median(confidence_values) if confidence_values else 0,
                'std': np.std(confidence_values) if confidence_values else 0,
                'total_pairs': len(match_pairs),
                'threshold': classifier.match_threshold if hasattr(classifier, 'match_threshold') else 0.5
            }
            
            # Save confidence statistics to JSON
            with open(self.detailed_dir / 'confidence_statistics.json', 'w') as f:
                json.dump(confidence_stats, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error generating match pair reports: {e}", exc_info=True)
    
    def _generate_clustering_reports(self, clusterer):
        """Generate detailed clustering reports"""
        try:
            if not hasattr(clusterer, 'get_clusters'):
                logger.warning("Clusterer does not have get_clusters method")
                return
            
            clusters = clusterer.get_clusters()
            
            if not clusters:
                logger.warning("No clusters available")
                return
            
            # Full cluster assignments CSV with additional metrics
            cluster_data = []
            for cluster_id, entities in clusters.items():
                for entity_idx, entity in enumerate(entities):
                    cluster_data.append({
                        'cluster_id': cluster_id,
                        'entity_id': entity,
                        'position_in_cluster': entity_idx + 1,
                        'cluster_size': len(entities),
                        'cluster_density': 1.0  # Placeholder, replace with actual density if available
                    })
            
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_csv(self.detailed_dir / 'clusters.csv', index=False)
            
            # Generate comprehensive cluster summary CSV
            summary_data = []
            for cluster_id, entities in clusters.items():
                # Calculate additional cluster metrics
                summary_data.append({
                    'cluster_id': cluster_id,
                    'size': len(entities),
                    'entities_sample': ','.join(entities[:5]) + ('...' if len(entities) > 5 else ''),
                    'entity_count': len(entities)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.sort_values('size', ascending=False, inplace=True)
            summary_df.to_csv(self.detailed_dir / 'cluster_summary.csv', index=False)
            
            # Generate size distribution CSV
            sizes = [len(entities) for entities in clusters.values()]
            size_counts = {}
            for size in sizes:
                size_counts[size] = size_counts.get(size, 0) + 1
            
            size_data = [{'cluster_size': size, 'count': count} 
                       for size, count in sorted(size_counts.items())]
            
            if size_data:
                size_df = pd.DataFrame(size_data)
                size_df.to_csv(self.detailed_dir / 'cluster_size_distribution.csv', index=False)
            
            # Generate cluster statistics JSON
            cluster_stats = {
                'total_clusters': len(clusters),
                'total_entities': sum(len(entities) for entities in clusters.values()),
                'size_statistics': {
                    'min': min(sizes) if sizes else 0,
                    'max': max(sizes) if sizes else 0,
                    'mean': np.mean(sizes) if sizes else 0,
                    'median': np.median(sizes) if sizes else 0,
                    'std': np.std(sizes) if sizes else 0
                },
                'size_distribution': {str(size): count for size, count in sorted(size_counts.items())}
            }
            
            with open(self.detailed_dir / 'cluster_statistics.json', 'w') as f:
                json.dump(cluster_stats, f, indent=2)
            
            # Generate most connected entities report if graph is available
            if hasattr(clusterer, 'graph') and clusterer.graph:
                try:
                    # Get top 100 most connected entities
                    connected_entities = sorted(
                        clusterer.graph.degree(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:100]
                    
                    connected_data = []
                    for entity, degree in connected_entities:
                        connected_data.append({
                            'entity_id': entity,
                            'connections': degree,
                            'cluster_id': next((cid for cid, entities in clusters.items() 
                                             if entity in entities), None)
                        })
                    
                    if connected_data:
                        connected_df = pd.DataFrame(connected_data)
                        connected_df.to_csv(self.detailed_dir / 'most_connected_entities.csv', index=False)
                
                except Exception as e:
                    logger.error(f"Error generating connected entities report: {e}")
        
        except Exception as e:
            logger.error(f"Error generating clustering reports: {e}", exc_info=True)
    
    def _generate_model_parameter_reports(self, classifier):
        """Generate detailed model parameter reports"""
        try:
            # Check if classifier has weights and bias
            if hasattr(classifier, 'weights') and hasattr(classifier, 'bias'):
                weights = classifier.weights
                bias = classifier.bias
                
                # Get feature names if available
                feature_names = self.classifier_state.get('feature_names', [])
                if not feature_names and hasattr(classifier, 'feature_names'):
                    feature_names = classifier.feature_names
                
                if not feature_names:
                    feature_names = [f"feature_{i}" for i in range(len(weights))]
                
                # Create weights dataframe
                weights_data = []
                for i, name in enumerate(feature_names):
                    if i < len(weights):
                        weights_data.append({
                            'feature': name,
                            'weight': weights[i],
                            'abs_weight': abs(weights[i])
                        })
                
                if weights_data:
                    weights_df = pd.DataFrame(weights_data)
                    weights_df.sort_values('abs_weight', ascending=False, inplace=True)
                    weights_df.to_csv(self.detailed_dir / 'model_weights.csv', index=False)
                
                # Save model parameters to JSON
                model_params = {
                    'bias': float(bias) if bias is not None else None,
                    'weights': {name: float(weights[i]) 
                              for i, name in enumerate(feature_names) 
                              if i < len(weights)},
                    'threshold': classifier.match_threshold if hasattr(classifier, 'match_threshold') else 0.5
                }
                
                with open(self.detailed_dir / 'model_parameters.json', 'w') as f:
                    json.dump(model_params, f, indent=2)
            
            # Save additional classifier parameters if available
            if hasattr(classifier, 'get_parameters'):
                params = classifier.get_parameters()
                if params:
                    with open(self.detailed_dir / 'classifier_parameters.json', 'w') as f:
                        json.dump(params, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error generating model parameter reports: {e}", exc_info=True)
    
    def _generate_visualization_reports(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate visualization reports"""
        logger.info("Generating visualization reports")
        
        # Set Seaborn style for better plots
        sns.set(style="whitegrid")
        
        # 1. Preprocessing visualizations
        self._generate_preprocessing_visualizations(preprocessor)
        
        # 2. Classification visualizations
        self._generate_classification_visualizations(classifier)
        
        # 3. Clustering visualizations
        self._generate_clustering_visualizations(clusterer)
        
        logger.info("Visualization reports generated")
    
    def _generate_preprocessing_visualizations(self, preprocessor):
        """Generate preprocessing visualizations"""
        try:
            # String frequency distribution
            if hasattr(preprocessor, 'get_string_counts'):
                string_counts = preprocessor.get_string_counts()
                if string_counts:
                    counts = list(string_counts.values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(counts, bins=50, log=True, alpha=0.7, color='skyblue')
                    plt.title('String Frequency Distribution', fontsize=14)
                    plt.xlabel('Frequency', fontsize=12)
                    plt.ylabel('Count (log scale)', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'string_frequency_distribution.png', dpi=300)
                    plt.close()
                    
                    # Top 20 most frequent strings
                    unique_strings = {}
                    if hasattr(preprocessor, 'get_unique_strings'):
                        unique_strings = preprocessor.get_unique_strings()
                    
                    top_strings = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                    if unique_strings and top_strings:
                        plt.figure(figsize=(12, 8))
                        
                        labels = []
                        for hash_val, _ in top_strings:
                            if hash_val in unique_strings:
                                s = unique_strings[hash_val]
                                # Truncate long strings
                                s = s[:30] + '...' if len(s) > 30 else s
                                labels.append(s)
                            else:
                                labels.append(hash_val[:10] + '...')
                        
                        values = [count for _, count in top_strings]
                        
                        plt.barh(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
                        plt.yticks(range(len(labels)), labels)
                        plt.title('Top 20 Most Frequent Strings', fontsize=14)
                        plt.xlabel('Frequency', fontsize=12)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'top_strings.png', dpi=300)
                        plt.close()
            
            # Field type distribution
            if hasattr(preprocessor, 'get_field_mapping'):
                field_mapping = preprocessor.get_field_mapping()
                if field_mapping:
                    field_counts = {}
                    for _, fields in field_mapping.items():
                        for field in fields:
                            field_counts[field] = field_counts.get(field, 0) + 1
                    
                    if field_counts:
                        plt.figure(figsize=(10, 6))
                        
                        sorted_items = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
                        labels = [item[0] for item in sorted_items]
                        values = [item[1] for item in sorted_items]
                        
                        plt.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
                        plt.xticks(rotation=45, ha='right')
                        plt.title('Field Type Distribution', fontsize=14)
                        plt.xlabel('Field Type', fontsize=12)
                        plt.ylabel('Count', fontsize=12)
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'field_type_distribution.png', dpi=300)
                        plt.close()
        
        except Exception as e:
            logger.error(f"Error generating preprocessing visualizations: {e}", exc_info=True)
    
    def _generate_classification_visualizations(self, classifier):
        """Generate classification visualizations"""
        try:
            if hasattr(classifier, 'get_match_pairs'):
                match_pairs = classifier.get_match_pairs()
                
                if match_pairs:
                    confidences = [conf for _, _, conf in match_pairs]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(confidences, bins=20, alpha=0.7, color='forestgreen')
                    plt.axvline(classifier.match_threshold if hasattr(classifier, 'match_threshold') else 0.5, 
                               color='red', linestyle='--', 
                               label=f'Threshold: {classifier.match_threshold if hasattr(classifier, "match_threshold") else 0.5:.2f}')
                    plt.title('Match Confidence Distribution', fontsize=14)
                    plt.xlabel('Confidence', fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'match_confidence_distribution.png', dpi=300)
                    plt.close()
                    
                    # Entity match counts
                    entity_matches = {}
                    for e1, e2, _ in match_pairs:
                        entity_matches[e1] = entity_matches.get(e1, 0) + 1
                        entity_matches[e2] = entity_matches.get(e2, 0) + 1
                    
                    if entity_matches:
                        match_counts = list(entity_matches.values())
                        
                        plt.figure(figsize=(10, 6))
                        plt.hist(match_counts, bins=20, alpha=0.7, color='steelblue')
                        plt.title('Matches per Entity Distribution', fontsize=14)
                        plt.xlabel('Number of Matches', fontsize=12)
                        plt.ylabel('Number of Entities', fontsize=12)
                        plt.yscale('log')  # Log scale for better visibility
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'matches_per_entity.png', dpi=300)
                        plt.close()
            
            # Generate ROC curve and precision-recall curve if test data available
            if ('test_data' in self.classifier_state and 
                'test_labels' in self.classifier_state and 
                'test_scores' in self.classifier_state):
                
                X_test = self.classifier_state['test_data']
                y_test = self.classifier_state['test_labels']
                y_scores = self.classifier_state['test_scores']
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_test, y_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (area = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
                plt.legend(loc="lower right")
                plt.grid(linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(self.visualizations_dir / 'roc_curve.png', dpi=300)
                plt.close()
                
                # Precision-Recall Curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
                
                plt.figure(figsize=(10, 8))
                plt.plot(recall, precision, color='blue', lw=2)
                
                # Add threshold markers
                for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    idx = (np.abs(thresholds - t)).argmin()
                    if idx < len(precision) - 1:  # Ensure index is valid
                        plt.plot(recall[idx], precision[idx], 'ro')
                        plt.annotate(f't={t:.1f}', 
                                    (recall[idx], precision[idx]),
                                    textcoords="offset points",
                                    xytext=(0,10),
                                    ha='center')
                
                # Add baseline
                plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', 
                           label=f'Baseline (support ratio = {sum(y_test)/len(y_test):.3f})')
                
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title('Precision-Recall Curve', fontsize=14)
                plt.legend()
                plt.grid(linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(self.visualizations_dir / 'precision_recall_curve.png', dpi=300)
                plt.close()
                
                # Save curve data to CSV for reference
                curve_data = []
                for i in range(len(fpr)):
                    curve_data.append({
                        'threshold': thresholds[i] if i < len(thresholds) else None,
                        'fpr': fpr[i],
                        'tpr': tpr[i],
                        'precision': precision[i] if i < len(precision) else None,
                        'recall': recall[i] if i < len(recall) else None
                    })
                
                curve_df = pd.DataFrame(curve_data)
                curve_df.to_csv(self.detailed_dir / 'curve_data.csv', index=False)
        
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {e}", exc_info=True)
    
    def _generate_clustering_visualizations(self, clusterer):
        """Generate clustering visualizations"""
        try:
            if hasattr(clusterer, 'get_clusters'):
                clusters = clusterer.get_clusters()
                
                if clusters:
                    # Cluster size distribution
                    sizes = [len(entities) for entities in clusters.values()]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(sizes, bins=20, alpha=0.7, color='mediumpurple')
                    plt.title('Cluster Size Distribution', fontsize=14)
                    plt.xlabel('Cluster Size', fontsize=12)
                    plt.ylabel('Number of Clusters', fontsize=12)
                    plt.yscale('log')  # Log scale for better visibility
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'cluster_size_distribution.png', dpi=300)
                    plt.close()
                    
                    # Top 15 largest clusters
                    top_clusters = sorted([(cid, len(entities)) 
                                         for cid, entities in clusters.items()], 
                                        key=lambda x: x[1], reverse=True)[:15]
                    
                    if top_clusters:
                        plt.figure(figsize=(12, 6))
                        
                        labels = [f"Cluster {cid}" for cid, _ in top_clusters]
                        values = [size for _, size in top_clusters]
                        
                        plt.bar(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
                        
                        # Add size labels on top of bars
                        for i, v in enumerate(values):
                            plt.text(i, v + max(values) * 0.01, str(v), ha='center')
                        
                        plt.xticks(range(len(labels)), labels, rotation=45)
                        plt.title('Top 15 Largest Clusters', fontsize=14)
                        plt.ylabel('Cluster Size', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'largest_clusters.png', dpi=300)
                        plt.close()
                    
                    # Generate cumulative distribution
                    size_counts = Counter(sizes)
                    sorted_sizes = sorted(size_counts.items())
                    
                    cum_clusters = 0
                    cum_entities = 0
                    cum_data = []
                    
                    total_clusters = len(clusters)
                    total_entities = sum(sizes)
                    
                    for size, count in sorted_sizes:
                        cum_clusters += count
                        cum_entities += size * count
                        
                        cum_data.append({
                            'cluster_size': size,
                            'cluster_count': count,
                            'cum_clusters': cum_clusters,
                            'cum_entities': cum_entities,
                            'cum_clusters_pct': 100 * cum_clusters / total_clusters,
                            'cum_entities_pct': 100 * cum_entities / total_entities
                        })
                    
                    if cum_data:
                        cum_df = pd.DataFrame(cum_data)
                        cum_df.to_csv(self.detailed_dir / 'cumulative_cluster_distribution.csv', index=False)
                        
                        # Plot cumulative percentage
                        plt.figure(figsize=(10, 6))
                        
                        plt.plot(cum_df['cluster_size'], cum_df['cum_clusters_pct'], 
                               label='Clusters', marker='o', markersize=4)
                        plt.plot(cum_df['cluster_size'], cum_df['cum_entities_pct'], 
                               label='Entities', marker='s', markersize=4)
                        
                        plt.title('Cumulative Distribution', fontsize=14)
                        plt.xlabel('Cluster Size (up to)', fontsize=12)
                        plt.ylabel('Cumulative Percentage (%)', fontsize=12)
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'cumulative_distribution.png', dpi=300)
                        plt.close()
        
        except Exception as e:
            logger.error(f"Error generating clustering visualizations: {e}", exc_info=True)
    
    def _generate_feature_analysis_reports(self, classifier):
        """Generate comprehensive feature analysis reports"""
        logger.info("Generating feature analysis reports")
        
        try:
            import numpy as np
            
            # Try to get feature importance from state or classifier
            feature_importance = None
            feature_names = None
            
            if 'feature_importance' in self.classifier_state:
                feature_importance = self.classifier_state['feature_importance']
                
            elif hasattr(classifier, '_get_feature_importance'):
                feature_importance = classifier._get_feature_importance()
            
            if 'feature_names' in self.classifier_state:
                feature_names = self.classifier_state['feature_names']
            
            if not feature_importance:
                logger.warning("No feature importance data available")
                return
            
            if not feature_names:
                # Try to infer names from keys
                if isinstance(feature_importance, dict):
                    feature_names = list(feature_importance.keys())
                else:
                    feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            # Create feature importance dataframe with normalized importance
            importance_data = []
            
            if isinstance(feature_importance, dict):
                total_importance = sum(abs(v) for v in feature_importance.values()) if feature_importance else 1
                
                for feature, importance in feature_importance.items():
                    importance_data.append({
                        'feature': feature,
                        'importance': importance,
                        'abs_importance': abs(importance),
                        'normalized_importance': abs(importance) / total_importance if total_importance > 0 else 0
                    })
            else:
                total_importance = sum(abs(v) for v in feature_importance) if feature_importance else 1
                
                for i, importance in enumerate(feature_importance):
                    if i < len(feature_names):
                        importance_data.append({
                            'feature': feature_names[i],
                            'importance': importance,
                            'abs_importance': abs(importance),
                            'normalized_importance': abs(importance) / total_importance if total_importance > 0 else 0
                        })
            
            if not importance_data:
                logger.warning("No feature importance data to report")
                return
            
            # Create dataframe and sort by absolute importance
            importance_df = pd.DataFrame(importance_data)
            importance_df.sort_values('abs_importance', ascending=False, inplace=True)
            
            # Save to CSV
            importance_df.to_csv(self.feature_dir / 'feature_importance.csv', index=False)
            
            # Create feature importance visualization
            plt.figure(figsize=(12, 10))
            
            # Show top 15 features for readability
            top_features = importance_df.head(15)
            # Fix seaborn barplot warning by specifying hue and legend
            sns.barplot(x='abs_importance', y='feature', data=top_features, color='steelblue')
            
            plt.title('Feature Importance (Absolute Values)', fontsize=16)
            plt.xlabel('Importance Score', fontsize=14)
            plt.ylabel('Feature', fontsize=14)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'feature_importance.png', dpi=300)
            plt.close()
            
            # Create pie chart for relative feature importance (top 8 features)
            plt.figure(figsize=(10, 10))
            
            top_for_pie = importance_df.head(8)
            # Add 'Other' category for remaining features
            if len(importance_df) > 8:
                other_sum = importance_df.iloc[8:]['abs_importance'].sum()
                other_row = pd.DataFrame([{
                    'feature': 'Other', 
                    'importance': other_sum, 
                    'abs_importance': other_sum,
                    'normalized_importance': other_sum / total_importance if total_importance > 0 else 0
                }])
                top_for_pie = pd.concat([top_for_pie, other_row])
            
            plt.pie(top_for_pie['abs_importance'], labels=top_for_pie['feature'], autopct='%1.1f%%',
                startangle=90, shadow=True, explode=[0.05]*len(top_for_pie))
            
            plt.title('Relative Feature Importance', fontsize=16)
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'feature_importance_pie.png', dpi=300)
            plt.close()
            
            # Generate additional feature importance analysis if test data is available
            if 'test_data' in self.classifier_state and 'test_labels' in self.classifier_state:
                X_test = self.classifier_state['test_data']
                y_test = self.classifier_state['test_labels']
                
                # Convert to numpy arrays if needed
                X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
                y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
                
                if X_test is not None and y_test is not None and len(feature_names) >= X_test.shape[1]:
                    feature_names = feature_names[:X_test.shape[1]]
                    
                    # Calculate feature means by class
                    feature_means = []
                    
                    for i, name in enumerate(feature_names):
                        if i < X_test.shape[1]:
                            feature_col = X_test[:, i]
                            
                            # Calculate means for positive and negative class
                            pos_mask = y_test == 1
                            neg_mask = y_test == 0
                            
                            pos_values = feature_col[pos_mask] if any(pos_mask) else np.array([])
                            neg_values = feature_col[neg_mask] if any(neg_mask) else np.array([])
                            
                            pos_mean = np.mean(pos_values) if len(pos_values) > 0 else 0
                            neg_mean = np.mean(neg_values) if len(neg_values) > 0 else 0
                            
                            # Calculate std for positive and negative class
                            pos_std = np.std(pos_values) if len(pos_values) > 0 else 0
                            neg_std = np.std(neg_values) if len(neg_values) > 0 else 0
                            
                            # Calculate separation (normalized difference between means)
                            separation = abs(pos_mean - neg_mean) / (pos_std + neg_std) if (pos_std + neg_std) > 0 else 0
                            
                            feature_means.append({
                                'feature': name,
                                'positive_mean': pos_mean,
                                'negative_mean': neg_mean,
                                'positive_std': pos_std,
                                'negative_std': neg_std,
                                'separation': separation,
                                'importance': importance_df[importance_df['feature'] == name]['importance'].values[0]
                                if name in importance_df['feature'].values else 0
                            })
                    
                    if feature_means:
                        # Create dataframe and sort by separation
                        means_df = pd.DataFrame(feature_means)
                        means_df.sort_values('separation', ascending=False, inplace=True)
                        
                        # Save to CSV
                        means_df.to_csv(self.feature_dir / 'feature_class_separation.csv', index=False)
                        
                        # Create visualization for top 10 features by separation
                        top_separated = means_df.head(10)
                        
                        plt.figure(figsize=(12, 8))
                        
                        # Create grouped bar chart
                        x = np.arange(len(top_separated))
                        width = 0.35
                        
                        plt.bar(x - width/2, top_separated['positive_mean'], width, label='Positive Class',
                            color='green', alpha=0.7)
                        plt.bar(x + width/2, top_separated['negative_mean'], width, label='Negative Class',
                            color='red', alpha=0.7)
                        
                        plt.xlabel('Feature', fontsize=14)
                        plt.ylabel('Mean Value', fontsize=14)
                        plt.title('Feature Means by Class (Top 10 by Separation)', fontsize=16)
                        plt.xticks(x, top_separated['feature'], rotation=45, ha='right')
                        plt.legend()
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        
                        plt.savefig(self.visualizations_dir / 'feature_class_separation.png', dpi=300)
                        plt.close()
        
        except Exception as e:
            logger.error(f"Error generating feature analysis reports: {e}", exc_info=True)
    
    def _generate_feature_correlation_reports_from_state(self):
        """Generate feature correlation reports from classifier state"""
        try:
            if 'feature_correlation' not in self.classifier_state:
                logger.warning("No feature correlation data in classifier state")
                return
            
            feature_correlation = self.classifier_state['feature_correlation']
            feature_names = self.classifier_state.get('feature_names', list(feature_correlation.keys()))
            
            # Create correlation matrix
            corr_matrix = np.zeros((len(feature_names), len(feature_names)))
            
            for i, name1 in enumerate(feature_names):
                for j, name2 in enumerate(feature_names):
                    if name1 in feature_correlation and name2 in feature_correlation[name1]:
                        corr_matrix[i, j] = feature_correlation[name1][name2]
            
            # Generate correlation reports
            self.generate_feature_correlation_reports(corr_matrix, feature_names)
        
        except Exception as e:
            logger.error(f"Error generating feature correlation reports from state: {e}", exc_info=True)
    
    def _generate_misclassification_reports(self, classifier):
        """Generate detailed misclassification analysis reports"""
        logger.info("Generating misclassification reports")
        
        try:
            import numpy as np
            
            # Check if classifier has test data and predictions in state
            if 'test_data' not in self.classifier_state or 'test_labels' not in self.classifier_state or 'test_predictions' not in self.classifier_state:
                logger.warning("Missing test data components in classifier state")
                return
            
            X_test = self.classifier_state['test_data']
            y_test = self.classifier_state['test_labels']
            predictions = self.classifier_state['test_predictions']
            
            # Convert to numpy arrays if needed
            X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
            predictions = np.array(predictions) if not isinstance(predictions, np.ndarray) else predictions
            
            feature_names = self.classifier_state.get('feature_names', [f"feature_{i}" for i in range(X_test.shape[1])])
            
            # Create dictionary to map feature name to index
            feature_to_idx = {name: i for i, name in enumerate(feature_names) if i < X_test.shape[1]}
            
            # Identify misclassified examples
            misclassified = []
            
            for i in range(len(y_test)):
                if i < len(predictions) and predictions[i] != y_test[i]:
                    misclassified_item = {
                        'instance_id': i,
                        'true_label': int(y_test[i]),
                        'predicted_label': int(predictions[i]),
                        'error_type': 'False Positive' if predictions[i] == 1 and y_test[i] == 0 
                                    else 'False Negative'
                    }
                    
                    # Add feature values
                    for j, name in enumerate(feature_names):
                        if j < X_test.shape[1]:
                            misclassified_item[f"feature_{name}"] = float(X_test[i, j])
                    
                    # Add additional attributes if available
                    if 'pair_ids' in self.classifier_state and i < len(self.classifier_state['pair_ids']):
                        pair = self.classifier_state['pair_ids'][i]
                        misclassified_item['entity1'] = pair[0]
                        misclassified_item['entity2'] = pair[1]
                    
                    misclassified.append(misclassified_item)
            
            if not misclassified:
                logger.info("No misclassified examples found")
                return
            
            # Create and save misclassified examples dataframe
            misclassified_df = pd.DataFrame(misclassified)
            misclassified_df.to_csv(self.detailed_dir / 'misclassified_examples.csv', index=False)
            
            # Create summary by error type
            error_counts = misclassified_df['error_type'].value_counts().reset_index()
            error_counts.columns = ['Error Type', 'Count']
            error_counts.to_csv(self.detailed_dir / 'misclassification_summary.csv', index=False)
            
            # Create visualizations
            # 1. Error type distribution
            plt.figure(figsize=(10, 6))
            
            # Fix seaborn warning by using color instead of palette
            sns.barplot(x='Error Type', y='Count', data=error_counts, color='royalblue')
            
            plt.title('Misclassification Types', fontsize=16)
            plt.ylabel('Count', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'misclassification_types.png', dpi=300)
            plt.close()
            
            # 2. Feature distribution for misclassified vs. correctly classified examples
            if 'feature_importance' in self.classifier_state:
                feature_importance = self.classifier_state['feature_importance']
                
                # Get top features by importance
                top_features = []
                if isinstance(feature_importance, dict):
                    top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    top_feature_names = [name for name, _ in top_features]
                else:
                    # If feature_importance is a list/array
                    indices = np.argsort(np.abs(feature_importance))[-5:][::-1]  # Top 5 by absolute value
                    top_feature_names = [feature_names[i] for i in indices if i < len(feature_names)]
                
                # Identify correctly classified examples
                correct_indices = [i for i in range(len(y_test)) if i < len(predictions) and predictions[i] == y_test[i]]
                correct_X = X_test[correct_indices]
                correct_y = y_test[correct_indices]
                
                # Create feature distribution visualizations for each top feature
                for feature in top_feature_names:
                    if feature in feature_to_idx:
                        feature_idx = feature_to_idx[feature]
                        feature_col = f"feature_{feature}"
                        
                        plt.figure(figsize=(12, 6))
                        
                        # Correctly classified distributions
                        if len(correct_X) > 0:
                            correct_pos = correct_X[correct_y == 1, feature_idx] if any(correct_y == 1) else []
                            correct_neg = correct_X[correct_y == 0, feature_idx] if any(correct_y == 0) else []
                            
                            if len(correct_pos) > 0 and np.var(correct_pos) > 0:
                                sns.kdeplot(correct_pos, label='Correct Positive', color='green', alpha=0.5)
                            if len(correct_neg) > 0 and np.var(correct_neg) > 0:
                                sns.kdeplot(correct_neg, label='Correct Negative', color='blue', alpha=0.5)
                        
                        # Misclassified distributions
                        if feature_col in misclassified_df.columns:
                            false_pos = misclassified_df[misclassified_df['error_type'] == 'False Positive'][feature_col]
                            false_neg = misclassified_df[misclassified_df['error_type'] == 'False Negative'][feature_col]
                            
                            # Only create density plots if there's enough variance
                            if len(false_pos) > 0 and np.var(false_pos) > 0:
                                sns.kdeplot(false_pos, label='False Positive', color='red', linestyle='--', alpha=0.7)
                            elif len(false_pos) > 0:
                                # If no variance, just plot the points
                                plt.scatter(false_pos, [0.01] * len(false_pos), color='red', marker='x', 
                                        label='False Positive', alpha=0.7)
                            
                            if len(false_neg) > 0 and np.var(false_neg) > 0:
                                sns.kdeplot(false_neg, label='False Negative', color='orange', linestyle='--', alpha=0.7)
                            elif len(false_neg) > 0:
                                # If no variance, just plot the points
                                plt.scatter(false_neg, [0.005] * len(false_neg), color='orange', marker='x', 
                                        label='False Negative', alpha=0.7)
                        
                        plt.title(f'Distribution of {feature} by Classification Result', fontsize=16)
                        plt.xlabel('Feature Value', fontsize=14)
                        plt.ylabel('Density', fontsize=14)
                        plt.legend()
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        
                        plt.savefig(self.visualizations_dir / f'feature_misclassification_{feature}.png', dpi=300)
                        plt.close()
                
                # Create a combined scatter plot of top 2 features to show misclassification regions
                if len(top_feature_names) >= 2:
                    feature1 = top_feature_names[0]
                    feature2 = top_feature_names[1]
                    
                    if feature1 in feature_to_idx and feature2 in feature_to_idx:
                        feature1_idx = feature_to_idx[feature1]
                        feature2_idx = feature_to_idx[feature2]
                        
                        plt.figure(figsize=(12, 10))
                        
                        # Plot correctly classified points
                        if len(correct_X) > 0:
                            plt.scatter(correct_X[correct_y == 0, feature1_idx], 
                                    correct_X[correct_y == 0, feature2_idx],
                                    color='blue', marker='o', alpha=0.6, label='Correct Negative')
                            
                            plt.scatter(correct_X[correct_y == 1, feature1_idx], 
                                    correct_X[correct_y == 1, feature2_idx],
                                    color='green', marker='^', alpha=0.6, label='Correct Positive')
                        
                        # Plot misclassified points
                        feature1_col = f"feature_{feature1}"
                        feature2_col = f"feature_{feature2}"
                        
                        if feature1_col in misclassified_df.columns and feature2_col in misclassified_df.columns:
                            false_pos = misclassified_df[misclassified_df['error_type'] == 'False Positive']
                            false_neg = misclassified_df[misclassified_df['error_type'] == 'False Negative']
                            
                            if len(false_pos) > 0:
                                plt.scatter(false_pos[feature1_col], false_pos[feature2_col],
                                        color='red', marker='x', s=100, alpha=0.8, label='False Positive')
                            
                            if len(false_neg) > 0:
                                plt.scatter(false_neg[feature1_col], false_neg[feature2_col],
                                        color='orange', marker='x', s=100, alpha=0.8, label='False Negative')
                        
                        plt.title(f'Classification Results: {feature1} vs {feature2}', fontsize=16)
                        plt.xlabel(feature1, fontsize=14)
                        plt.ylabel(feature2, fontsize=14)
                        plt.legend()
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        
                        plt.savefig(self.visualizations_dir / 'misclassification_scatter.png', dpi=300)
                        plt.close()
            
            logger.info("Misclassification reports generated")
        
        except Exception as e:
            logger.error(f"Error generating misclassification reports: {e}", exc_info=True)
    
    def generate_test_set_csv_report(self, X_test, y_test, y_pred, feature_names=None, preprocessor=None):
        """Generate comprehensive CSV report with test set results, features, and predictions
        
        Args:
            X_test: Test set feature vectors (list or numpy array)
            y_test: True labels (list or numpy array)
            y_pred: Predicted labels (list or numpy array)
            feature_names: Optional feature names
            preprocessor: Optional preprocessor for additional metadata
        """
        logger.info("Generating comprehensive test set CSV report")
        
        try:
            import numpy as np
            
            # Convert to numpy arrays if they're not already
            X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
            y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            # Create dataframe with all test data
            test_data = []
            
            for i in range(len(y_test)):
                if i < len(y_pred):
                    item = {
                        'instance_id': i,
                        'true_label': int(y_test[i]),
                        'predicted_label': int(y_pred[i]),
                        'correctly_classified': y_test[i] == y_pred[i]
                    }
                    
                    # Add prediction confidence if available
                    if 'test_scores' in self.classifier_state and i < len(self.classifier_state['test_scores']):
                        item['prediction_confidence'] = float(self.classifier_state['test_scores'][i])
                    
                    # Add pair IDs if available
                    if 'pair_ids' in self.classifier_state and i < len(self.classifier_state['pair_ids']):
                        pair = self.classifier_state['pair_ids'][i]
                        item['entity1'] = pair[0]
                        item['entity2'] = pair[1]
                    
                    # Add all feature values with meaningful column names
                    for j, name in enumerate(feature_names):
                        if j < X_test.shape[1]:
                            item[f"feature_{name}"] = float(X_test[i][j])
                    
                    test_data.append(item)
            
            if not test_data:
                logger.warning("No test data available for CSV report")
                return
            
            # Convert to dataframe
            test_df = pd.DataFrame(test_data)
            
            # Save full test set to CSV
            test_df.to_csv(self.detailed_dir / 'test_set_with_features.csv', index=False)
            
            # Create a separate misclassified examples report
            misclassified_df = test_df[test_df['correctly_classified'] == False]
            misclassified_df.to_csv(self.detailed_dir / 'test_set_misclassified.csv', index=False)
            
            # Create correctly classified examples report
            correct_df = test_df[test_df['correctly_classified'] == True]
            correct_df.to_csv(self.detailed_dir / 'test_set_correct.csv', index=False)
            
            # Create summary statistics
            summary = {
                'total_test_samples': len(test_df),
                'correctly_classified': len(correct_df),
                'misclassified': len(misclassified_df),
                'accuracy': len(correct_df) / len(test_df) if len(test_df) > 0 else 0,
            }
            
            # Calculate true/false positives/negatives
            true_negatives = len(correct_df[correct_df['true_label'] == 0])
            true_positives = len(correct_df[correct_df['true_label'] == 1])
            false_positives = len(misclassified_df[(misclassified_df['true_label'] == 0) & (misclassified_df['predicted_label'] == 1)])
            false_negatives = len(misclassified_df[(misclassified_df['true_label'] == 1) & (misclassified_df['predicted_label'] == 0)])
            
            summary['true_negatives'] = true_negatives
            summary['true_positives'] = true_positives
            summary['false_positives'] = false_positives
            summary['false_negatives'] = false_negatives
            
            # Calculate precision, recall, and F1
            tp = summary['true_positives']
            fp = summary['false_positives']
            fn = summary['false_negatives']
            
            summary['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            summary['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            summary['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            # Save summary to JSON
            with open(self.detailed_dir / 'test_set_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Create a confusion matrix CSV
            confusion = {
                'true_positive': summary['true_positives'],
                'false_positive': summary['false_positives'],
                'false_negative': summary['false_negatives'],
                'true_negative': summary['true_negatives']
            }
            
            confusion_df = pd.DataFrame([confusion])
            confusion_df.to_csv(self.detailed_dir / 'confusion_matrix.csv', index=False)
            
            # Create visualization for test set results
            if self.visualizations_enabled:
                # Confusion matrix visualization
                try:
                    cm = np.array([
                        [summary['true_negatives'], summary['false_positives']],
                        [summary['false_negatives'], summary['true_positives']]
                    ])
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Negative', 'Positive'],
                            yticklabels=['Negative', 'Positive'])
                    plt.ylabel('True Label', fontsize=12)
                    plt.xlabel('Predicted Label', fontsize=12)
                    plt.title('Confusion Matrix', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'confusion_matrix.png', dpi=300)
                    plt.close()
                    
                    # Performance metrics bar chart
                    plt.figure(figsize=(10, 6))
                    metrics = [summary['accuracy'], summary['precision'], summary['recall'], summary['f1']]
                    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    
                    # Use simple color parameter instead of palette
                    plt.bar(labels, metrics, color='skyblue')
                    
                    # Add values on top of bars
                    for i, v in enumerate(metrics):
                        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
                    
                    plt.ylim(0, 1.1)
                    plt.title('Classification Performance Metrics', fontsize=14)
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'performance_metrics.png', dpi=300)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create test set visualizations: {e}")
            
            logger.info(f"Test set report generated with {len(test_df)} samples")
        
        except Exception as e:
            logger.error(f"Error generating test set CSV report: {e}", exc_info=True)
    
    def generate_feature_correlation_reports(self, feature_correlation_matrix, feature_names=None):
        """Generate detailed feature correlation reports and visualizations
        
        Args:
            feature_correlation_matrix: Matrix of feature correlations (list or numpy array)
            feature_names: Optional list of feature names
        """
        logger.info("Generating feature correlation reports")
        
        try:
            import numpy as np
            
            # Convert to numpy array if it's not already
            feature_correlation_matrix = np.array(feature_correlation_matrix) if not isinstance(feature_correlation_matrix, np.ndarray) else feature_correlation_matrix
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(feature_correlation_matrix.shape[0])]
            
            # Ensure matrix is square and matches feature names length
            if feature_correlation_matrix.shape[0] != feature_correlation_matrix.shape[1]:
                logger.error("Correlation matrix is not square")
                return
            
            if feature_correlation_matrix.shape[0] != len(feature_names):
                logger.error("Correlation matrix size does not match feature names length")
                return
            
            # Create full correlation dataframe
            all_correlations = []
            
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):  # Only upper triangle to avoid duplicates
                    all_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': feature_correlation_matrix[i, j],
                        'abs_correlation': abs(feature_correlation_matrix[i, j])
                    })
            
            corr_df = pd.DataFrame(all_correlations)
            corr_df.sort_values('abs_correlation', ascending=False, inplace=True)
            
            # Save to CSV
            corr_df.to_csv(self.feature_dir / 'feature_correlations.csv', index=False)
            
            # Generate top correlations report
            top_correlations = corr_df.head(100)  # Top 100 correlations
            top_correlations.to_csv(self.feature_dir / 'top_feature_correlations.csv', index=False)
            
            # Add correlation sign analysis
            positive_corr = corr_df[corr_df['correlation'] > 0].shape[0]
            negative_corr = corr_df[corr_df['correlation'] < 0].shape[0]
            
            sign_analysis = {
                'positive_correlations': positive_corr,
                'negative_correlations': negative_corr,
                'total_correlations': len(corr_df),
                'positive_percentage': 100 * positive_corr / len(corr_df) if len(corr_df) > 0 else 0,
                'negative_percentage': 100 * negative_corr / len(corr_df) if len(corr_df) > 0 else 0
            }
            
            with open(self.feature_dir / 'correlation_sign_analysis.json', 'w') as f:
                json.dump(sign_analysis, f, indent=2)
            
            # Create correlation strength histogram
            plt.figure(figsize=(10, 6))
            plt.hist(corr_df['correlation'].values, bins=20, alpha=0.7, color='skyblue')
            plt.title('Feature Correlation Distribution', fontsize=14)
            plt.xlabel('Correlation Coefficient', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'correlation_distribution.png', dpi=300)
            plt.close()
            
            # Generate correlation heatmap
            plt.figure(figsize=(15, 12))
            
            mask = np.zeros_like(feature_correlation_matrix, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = False  # Show only upper triangle
            mask[np.tril_indices_from(mask, k=0)] = True   # Hide lower triangle and diagonal
            
            sns.heatmap(feature_correlation_matrix, mask=mask, annot=True,
                    cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
                    xticklabels=feature_names, yticklabels=feature_names)
            
            plt.title('Feature Correlation Matrix', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'feature_correlation_heatmap.png', dpi=300)
            plt.close()
            
            # Generate feature clusters based on correlation
            # Convert correlation to distance (1 - abs(correlation))
            try:
                from scipy.cluster import hierarchy
                from scipy.spatial.distance import pdist, squareform
                
                # Use pdist instead of squareform for computing the condensed distance matrix
                dists = pdist(feature_correlation_matrix, metric=lambda u, v: 1 - abs(np.corrcoef(u, v)[0, 1]))
                
                z = hierarchy.linkage(dists, method='average')
                
                plt.figure(figsize=(15, 10))
                
                # Plot dendrogram
                hierarchy.dendrogram(
                    z,
                    labels=feature_names,
                    leaf_rotation=90,
                    leaf_font_size=10
                )
                
                plt.title('Feature Clustering by Correlation', fontsize=16)
                plt.tight_layout()
                
                plt.savefig(self.visualizations_dir / 'feature_correlation_clusters.png', dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Could not generate correlation clustering: {e}")
            
            logger.info("Feature correlation reports generated")
        
        except Exception as e:
            logger.error(f"Error generating feature correlation reports: {e}", exc_info=True)
    
    def generate_rfe_reports(self, feature_extractor, analyzer=None):
        """Generate comprehensive RFE reports and visualizations
        
        Args:
            feature_extractor: Feature extractor with RFE results
            analyzer: Optional analyzer with RFE analysis results
        """
        logger.info("Generating RFE reports")
        
        try:
            # Create RFE directory
            rfe_dir = self.output_dir / 'rfe'
            ensure_dir(rfe_dir)
            
            # Check if RFE is enabled and has results
            if not feature_extractor.rfe_enabled:
                logger.info("RFE is not enabled in configuration, skipping reports")
                return
            
            if feature_extractor.rfe_model is None:
                logger.warning("No RFE model available for reporting")
                return
            
            # Get RFE results
            rfe_results = feature_extractor.get_rfe_results()
            if not rfe_results:
                logger.warning("No RFE results available")
                return
            
            # Generate summary report
            self._generate_rfe_summary_report(rfe_results, feature_extractor, rfe_dir)
            
            # Generate feature ranking report
            self._generate_rfe_ranking_report(rfe_results, rfe_dir)
            
            # Generate feature selection impact report if analyzer has RFE results
            if analyzer and hasattr(analyzer, 'rfe_analysis') and analyzer.rfe_analysis:
                self._generate_rfe_impact_report(analyzer.rfe_analysis, rfe_dir)
            
            # Generate visualizations if enabled
            if self.visualizations_enabled:
                self._generate_rfe_visualizations(rfe_results, rfe_dir)
            
            logger.info("RFE reports generated")
        
        except Exception as e:
            logger.error(f"Error generating RFE reports: {e}", exc_info=True)

    def _generate_rfe_summary_report(self, rfe_results, feature_extractor, output_dir):
        """Generate a comprehensive RFE summary report in multiple formats"""
        # Create markdown summary
        summary_lines = []
        summary_lines.append("# Recursive Feature Elimination (RFE) Summary Report")
        summary_lines.append("")
        
        # Add configuration details
        summary_lines.append("## Configuration")
        summary_lines.append(f"- Feature selection method: Recursive Feature Elimination")
        summary_lines.append(f"- Base estimator: LogisticRegression")
        summary_lines.append(f"- Step size: {feature_extractor.rfe_step}")
        summary_lines.append(f"- Cross-validation folds: {feature_extractor.rfe_cv}")
        summary_lines.append("")
        
        # Add feature selection summary
        summary_lines.append("## Feature Selection Summary")
        total_features = len(feature_extractor.feature_names)
        selected_features = rfe_results.get('selected_features', [])
        summary_lines.append(f"- Total features available: {total_features}")
        summary_lines.append(f"- Features selected: {len(selected_features)} ({len(selected_features)/total_features*100:.1f}%)")
        summary_lines.append("")
        
        # Add selected features list
        summary_lines.append("## Selected Features")
        for feature in selected_features:
            summary_lines.append(f"- {feature}")
        summary_lines.append("")
        
        # Add ranking information
        if 'feature_rankings' in rfe_results:
            summary_lines.append("## Feature Rankings")
            summary_lines.append("Features ranked by importance (lower rank = more important):")
            summary_lines.append("")
            
            # Create a table header
            summary_lines.append("| Rank | Feature | Selected |")
            summary_lines.append("|------|---------|----------|")
            
            # Sort features by rank
            sorted_rankings = sorted(rfe_results['feature_rankings'], key=lambda x: x['rank'])
            
            for ranking in sorted_rankings:
                feature = ranking['feature']
                rank = ranking['rank']
                selected = "✓" if ranking['selected'] else "✗"
                
                summary_lines.append(f"| {rank} | {feature} | {selected} |")
        
        # Write markdown report
        with open(output_dir / 'rfe_summary.md', 'w') as f:
            f.write('\n'.join(summary_lines))
        
        # Create JSON version
        summary_data = {
            'configuration': {
                'method': 'Recursive Feature Elimination',
                'base_estimator': 'LogisticRegression',
                'step_size': feature_extractor.rfe_step,
                'cv_folds': feature_extractor.rfe_cv
            },
            'selection_summary': {
                'total_features': total_features,
                'selected_features': len(selected_features),
                'selection_ratio': len(selected_features)/total_features
            },
            'selected_features': selected_features,
            'feature_rankings': rfe_results.get('feature_rankings', [])
        }
        
        with open(output_dir / 'rfe_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Create CSV version
        if 'feature_rankings' in rfe_results:
            rankings_df = pd.DataFrame(rfe_results['feature_rankings'])
            rankings_df.sort_values('rank', inplace=True)
            rankings_df.to_csv(output_dir / 'rfe_rankings.csv', index=False)

    def _generate_rfe_ranking_report(self, rfe_results, output_dir):
        """Generate a detailed feature ranking report with analysis"""
        if 'feature_rankings' not in rfe_results:
            logger.warning("No feature rankings available for RFE ranking report")
            return
        
        rankings = rfe_results['feature_rankings']
        
        # Calculate ranking statistics
        rank_values = [r['rank'] for r in rankings]
        min_rank = min(rank_values)
        max_rank = max(rank_values)
        
        # Group features by rank
        rank_groups = {}
        for r in rankings:
            rank = r['rank']
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(r['feature'])
        
        # Create detailed report
        report_lines = []
        report_lines.append("# Detailed Feature Ranking Analysis")
        report_lines.append("")
        
        report_lines.append("## Ranking Statistics")
        report_lines.append(f"- Total features evaluated: {len(rankings)}")
        report_lines.append(f"- Minimum rank: {min_rank}")
        report_lines.append(f"- Maximum rank: {max_rank}")
        report_lines.append(f"- Number of rank groups: {len(rank_groups)}")
        report_lines.append("")
        
        report_lines.append("## Features by Rank")
        for rank in sorted(rank_groups.keys()):
            features = rank_groups[rank]
            report_lines.append(f"### Rank {rank}")
            for feature in features:
                # Mark selected features
                is_selected = any(r['feature'] == feature and r['selected'] for r in rankings)
                status = "✓ (Selected)" if is_selected else "✗ (Eliminated)"
                report_lines.append(f"- {feature} {status}")
            report_lines.append("")
        
        # Write report
        with open(output_dir / 'detailed_ranking_analysis.md', 'w') as f:
            f.write('\n'.join(report_lines))

    def _generate_rfe_impact_report(self, rfe_analysis, output_dir):
        """Generate a report on the impact of feature selection on model performance"""
        report_lines = []
        report_lines.append("# Feature Selection Impact Analysis")
        report_lines.append("")
        
        stats = rfe_analysis.get('stats', {})
        
        report_lines.append("## Feature Selection Summary")
        report_lines.append(f"- Total features: {stats.get('total_features', 'N/A')}")
        report_lines.append(f"- Selected features: {stats.get('selected_features', 'N/A')} ({stats.get('selection_ratio', 0)*100:.1f}%)")
        report_lines.append("")
        
        # Add important features section
        if 'important_features' in stats:
            report_lines.append("## Most Important Features")
            for feature in stats['important_features'][:10]:  # Top 10
                report_lines.append(f"- {feature}")
            report_lines.append("")
        
        # Add eliminated features section
        if 'eliminated_features' in stats:
            report_lines.append("## Eliminated Features")
            report_lines.append("The following features were eliminated by RFE:")
            report_lines.append("")
            eliminated_count = len(stats['eliminated_features'])
            if eliminated_count > 20:
                # If there are many eliminated features, just show a sample
                report_lines.append(f"Total eliminated features: {eliminated_count}")
                report_lines.append("Sample of eliminated features:")
                for feature in stats['eliminated_features'][:20]:
                    report_lines.append(f"- {feature}")
                report_lines.append(f"... and {eliminated_count-20} more")
            else:
                for feature in stats['eliminated_features']:
                    report_lines.append(f"- {feature}")
            report_lines.append("")
        
        # Write report
        with open(output_dir / 'feature_selection_impact.md', 'w') as f:
            f.write('\n'.join(report_lines))

    def _generate_rfe_visualizations(self, rfe_results, output_dir):
        """Generate comprehensive visualizations for RFE results"""
        try:
            # Only proceed if we have feature rankings
            if 'feature_rankings' not in rfe_results:
                logger.warning("No feature rankings available for RFE visualizations")
                return
            
            rankings = rfe_results['feature_rankings']
            
            # Sort by rank for visualization
            sorted_rankings = sorted(rankings, key=lambda x: x['rank'])
            
            # 1. Feature Importance Ranking Plot with selection status
            plt.figure(figsize=(14, 10))
            
            # Display top 20 features for readability
            top_features = sorted_rankings[:20]
            features = [r['feature'] for r in top_features]
            ranks = [r['rank'] for r in top_features]
            is_selected = [r['selected'] for r in top_features]
            
            # Use a colormap based on selection status
            colors = ['#2ecc71' if s else '#e74c3c' for s in is_selected]
            
            y_pos = range(len(features))
            plt.barh(y_pos, ranks, color=colors)
            plt.yticks(y_pos, features)
            
            # Add a legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Selected'),
                Patch(facecolor='#e74c3c', label='Eliminated')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title('Feature Importance Rankings (Top 20)', fontsize=16)
            plt.xlabel('Rank (Lower is Better)', fontsize=14)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'feature_importance_ranking.png', dpi=300)
            plt.close()
            
            # 2. Feature Selection Ratio Pie Chart
            selected_count = sum(1 for r in rankings if r['selected'])
            eliminated_count = len(rankings) - selected_count
            
            plt.figure(figsize=(10, 8))
            
            plt.pie([selected_count, eliminated_count], 
                labels=['Selected', 'Eliminated'], 
                explode=(0.1, 0),
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                startangle=90,
                shadow=True)
            
            plt.title('Feature Selection Ratio', fontsize=16)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'feature_selection_ratio.png', dpi=300)
            plt.close()
            
            # 3. Rank Distribution Histogram
            all_ranks = [r['rank'] for r in rankings]
            
            plt.figure(figsize=(12, 6))
            
            plt.hist(all_ranks, bins=max(all_ranks), color='#3498db', edgecolor='black', alpha=0.7)
            
            # Add vertical line for selection threshold if we can determine it
            if any(r['selected'] for r in rankings):
                # Find the highest rank among selected features
                max_selected_rank = max(r['rank'] for r in rankings if r['selected'])
                plt.axvline(x=max_selected_rank, color='#27ae60', linestyle='--', 
                        label=f'Selection Threshold (Rank {max_selected_rank})')
                plt.legend(fontsize=12)
            
            plt.title('Distribution of Feature Ranks', fontsize=16)
            plt.xlabel('Rank', fontsize=14)
            plt.ylabel('Number of Features', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(output_dir / 'rank_distribution.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating RFE visualizations: {e}", exc_info=True)

    def generate_feature_distribution_reports(self, X, y, feature_names=None):
        """Generate detailed feature distribution reports and visualizations
        
        Args:
            X: Feature vectors (list or numpy array)
            y: Class labels (list or numpy array)
            feature_names: Optional list of feature names
        """
        logger.info("Generating feature distribution reports")
        
        try:
            import numpy as np
            
            # Convert to numpy arrays if they're not already
            X = np.array(X) if not isinstance(X, np.ndarray) else X
            y = np.array(y) if not isinstance(y, np.ndarray) else y
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Ensure dimensions match
            if X.shape[1] != len(feature_names):
                logger.warning(f"Feature vector dimensions ({X.shape[1]}) do not match feature names length ({len(feature_names)})")
                # Truncate feature names to match X.shape[1]
                feature_names = feature_names[:X.shape[1]]
            
            # Calculate statistics for each feature
            feature_stats = []
            
            for i, name in enumerate(feature_names):
                if i < X.shape[1]:
                    # Get values for this feature
                    values = X[:, i]
                    
                    # Split by class
                    pos_values = values[y == 1]
                    neg_values = values[y == 0]
                    
                    # Calculate statistics
                    stats = {
                        'feature': name,
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'mean': float(np.mean(values)),
                        'median': float(np.median(values)),
                        'std': float(np.std(values)),
                        'positive_mean': float(np.mean(pos_values)) if len(pos_values) > 0 else float('nan'),
                        'negative_mean': float(np.mean(neg_values)) if len(neg_values) > 0 else float('nan'),
                        'positive_std': float(np.std(pos_values)) if len(pos_values) > 0 else float('nan'),
                        'negative_std': float(np.std(neg_values)) if len(neg_values) > 0 else float('nan'),
                        'separation': float(abs(np.mean(pos_values) - np.mean(neg_values)) / 
                                        (np.std(pos_values) + np.std(neg_values)))
                                        if len(pos_values) > 0 and len(neg_values) > 0 and
                                            (np.std(pos_values) + np.std(neg_values)) > 0
                                        else float('nan')
                    }
                    
                    feature_stats.append(stats)
            
            if not feature_stats:
                logger.warning("No feature statistics could be calculated")
                return
            
            # Convert to dataframe and sort by separation
            stats_df = pd.DataFrame(feature_stats)
            
            # Handle NaN values in separation column
            # Fix the pandas chained assignment warning by using a copy
            stats_df_copy = stats_df.copy()
            stats_df_copy.loc[:, 'separation'] = stats_df_copy['separation'].fillna(0)
            stats_df = stats_df_copy
            
            stats_df.sort_values('separation', ascending=False, inplace=True)
            
            # Save to CSV
            stats_df.to_csv(self.feature_dir / 'feature_distributions.csv', index=False)
            
            # Create feature distribution visualization for top features
            # Determine how many features to visualize
            num_vis_features = min(10, len(feature_stats))
            top_features = stats_df.head(num_vis_features)
            
            for idx, row in top_features.iterrows():
                feature = row['feature']
                i = feature_names.index(feature)
                
                if i >= X.shape[1]:
                    continue
                
                # Get values
                values = X[:, i]
                pos_values = values[y == 1]
                neg_values = values[y == 0]
                
                plt.figure(figsize=(10, 6))
                
                # Create histograms by class
                if len(pos_values) > 0:
                    # Check for sufficient variance before using KDE
                    if np.var(pos_values) > 0:
                        sns.histplot(pos_values, color='green', alpha=0.5, label='Positive Class', 
                                    kde=True, stat='density')
                    else:
                        # Just create a histogram without KDE
                        sns.histplot(pos_values, color='green', alpha=0.5, label='Positive Class', 
                                    kde=False, stat='count')
                        
                if len(neg_values) > 0:
                    # Check for sufficient variance before using KDE
                    if np.var(neg_values) > 0:
                        sns.histplot(neg_values, color='red', alpha=0.5, label='Negative Class', 
                                    kde=True, stat='density')
                    else:
                        # Just create a histogram without KDE
                        sns.histplot(neg_values, color='red', alpha=0.5, label='Negative Class', 
                                    kde=False, stat='count')
                
                # Add means as vertical lines
                if len(pos_values) > 0:
                    plt.axvline(x=row['positive_mean'], color='green', linestyle='--', 
                            label=f"Positive Mean: {row['positive_mean']:.4f}")
                if len(neg_values) > 0:
                    plt.axvline(x=row['negative_mean'], color='red', linestyle='--', 
                            label=f"Negative Mean: {row['negative_mean']:.4f}")
                
                plt.title(f'Distribution of {feature} by Class (Separation: {row["separation"]:.4f})', 
                        fontsize=16)
                plt.xlabel('Feature Value', fontsize=14)
                plt.ylabel('Density', fontsize=14)
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                
                plt.savefig(self.visualizations_dir / f'feature_distribution_{feature}.png', dpi=300)
                plt.close()
            
            # Create combined scatter plot of top 2 features
            if len(top_features) >= 2:
                feature1 = top_features.iloc[0]['feature']
                feature2 = top_features.iloc[1]['feature']
                
                feature1_idx = feature_names.index(feature1)
                feature2_idx = feature_names.index(feature2)
                
                if feature1_idx < X.shape[1] and feature2_idx < X.shape[1]:
                    plt.figure(figsize=(10, 8))
                    
                    f1_values = X[:, feature1_idx]
                    f2_values = X[:, feature2_idx]
                    
                    # Scatter plot
                    plt.scatter(f1_values[y == 0], f2_values[y == 0], 
                            color='red', alpha=0.6, label='Negative Class')
                    plt.scatter(f1_values[y == 1], f2_values[y == 1], 
                            color='green', alpha=0.6, label='Positive Class')
                    
                    plt.title(f'Class Separation: {feature1} vs. {feature2}', fontsize=16)
                    plt.xlabel(feature1, fontsize=14)
                    plt.ylabel(feature2, fontsize=14)
                    plt.grid(alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    
                    plt.savefig(self.visualizations_dir / 'top_features_scatter.png', dpi=300)
                    plt.close()
                    
                    # Create interactive plots if bokeh is available
                    try:
                        from bokeh.plotting import figure, output_file, save
                        from bokeh.models import ColumnDataSource, HoverTool
                        
                        # Create dataset
                        source_pos = ColumnDataSource(data={
                            'x': f1_values[y == 1],
                            'y': f2_values[y == 1],
                            'label': ['Positive']*sum(y == 1)
                        })
                        
                        source_neg = ColumnDataSource(data={
                            'x': f1_values[y == 0],
                            'y': f2_values[y == 0],
                            'label': ['Negative']*sum(y == 0)
                        })
                        
                        # Create plot
                        output_file(str(self.visualizations_dir / 'interactive_scatter.html'))
                        
                        p = figure(width=800, height=600, title=f'Class Separation: {feature1} vs. {feature2}')
                        p.circle('x', 'y', size=8, color='green', alpha=0.6, legend_label='Positive Class', source=source_pos)
                        p.circle('x', 'y', size=8, color='red', alpha=0.6, legend_label='Negative Class', source=source_neg)
                        
                        p.xaxis.axis_label = feature1
                        p.yaxis.axis_label = feature2
                        
                        # Add hover tool
                        hover = HoverTool()
                        hover.tooltips = [
                            ('Class', '@label'),
                            (feature1, '@x{0.000}'),
                            (feature2, '@y{0.000}')
                        ]
                        p.add_tools(hover)
                        
                        save(p)
                    except ImportError:
                        logger.info("Bokeh not available, skipping interactive visualizations")
            
            logger.info("Feature distribution reports generated")
        
        except Exception as e:
            logger.error(f"Error generating feature distribution reports: {e}", exc_info=True)
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten a nested dictionary for CSV output"""
        items = {}
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
