"""
Enhanced Reporting module for entity resolution
Handles generation of comprehensive reports and visualizations
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

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Reporter:
    """Enhanced reporter for entity resolution pipeline"""
    
    def __init__(self, config):
        """Initialize the reporter with configuration"""
        self.config = config
        self.output_formats = config['reporting']['output_format']
        self.detailed_metrics = config['reporting']['detailed_metrics']
        self.save_misclassified = config['reporting']['save_misclassified']
        
        # Output directories
        self.output_dir = Path(config['general']['output_dir'])
        self.detailed_dir = self.output_dir / 'detailed'
        self.visualizations_dir = self.output_dir / 'visualizations'
        
        # Ensure directories exist
        ensure_dir(self.output_dir)
        ensure_dir(self.detailed_dir)
        ensure_dir(self.visualizations_dir)
        
        logger.info("Enhanced reporter initialized")
    
    def generate_reports(self, preprocessor, embedder, indexer, classifier, clusterer, output_dir=None):
        """Generate comprehensive reports for pipeline results"""
        with Timer() as timer:
            # Update output directory if provided
            if output_dir:
                self.output_dir = Path(output_dir)
                self.detailed_dir = self.output_dir / 'detailed'
                self.visualizations_dir = self.output_dir / 'visualizations'
                ensure_dir(self.output_dir)
                ensure_dir(self.detailed_dir)
                ensure_dir(self.visualizations_dir)
            
            logger.info(f"Generating comprehensive reports in {self.output_dir}")
            
            # Generate summary report
            self._generate_summary_report(
                preprocessor, embedder, indexer, classifier, clusterer
            )
            
            # Generate detailed reports
            self._generate_detailed_reports(
                preprocessor, embedder, indexer, classifier, clusterer
            )
            
            # Generate visualization reports
            self._generate_visualization_reports(
                preprocessor, embedder, indexer, classifier, clusterer
            )
            
            # Generate feature analysis reports
            self._generate_feature_analysis_reports(classifier)
            
            # Generate misclassification analysis
            if self.save_misclassified:
                self._generate_misclassification_reports(classifier)
            
            logger.info("Comprehensive report generation completed")
        
        logger.info(f"Report generation completed in {timer.elapsed:.2f} seconds")
    
    def _generate_summary_report(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate comprehensive summary report"""
        logger.info("Generating summary report")
        
        # Collect summary data
        summary = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing': {
                'unique_strings': len(preprocessor.get_unique_strings()),
                'records': len(preprocessor.get_record_field_hashes()),
                'person_ids': len(preprocessor.get_all_person_ids())
            },
            'embedding': {
                'embeddings': len(embedder.get_embeddings()),
                'dimension': self.config['openai']['embedding_dim']
            },
            'indexing': {
                'indexed': indexer.is_indexed()
            },
            'classification': {
                'matches': len(classifier.get_match_pairs()) if hasattr(classifier, 'get_match_pairs') else 0,
                'threshold': self.config['classification']['match_threshold']
            },
            'clustering': {
                'clusters': len(clusterer.get_clusters()) if hasattr(clusterer, 'get_clusters') else 0,
                'algorithm': self.config['clustering']['algorithm']
            },
            'config': {
                'mode': self.config['general']['mode'],
                'openai_model': self.config['openai']['embedding_model']
            }
        }
        
        # Add classifier metrics if available
        if hasattr(classifier, 'metrics') and classifier.metrics:
            metrics = classifier.metrics
            if isinstance(metrics, dict):
                summary['classification']['metrics'] = {
                    'precision': metrics.get('precision', None),
                    'recall': metrics.get('recall', None),
                    'f1': metrics.get('f1', None)
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
                        writer.writerow([key, value])
        
        logger.info("Summary report generated")
    
    def _generate_detailed_reports(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate detailed reports with full data"""
        logger.info("Generating detailed reports")
        
        # 1. Preprocessing details
        # Generate report of unique strings with frequencies
        unique_strings = preprocessor.get_unique_strings()
        string_counts = preprocessor.get_string_counts()
        field_mapping = preprocessor.get_field_mapping()
        
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
        
        # Generate field-specific reports
        for field in set(field for mapping in field_mapping.values() for field in mapping.keys()):
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
        
        # 2. Match pairs details
        if hasattr(classifier, 'get_match_pairs'):
            match_pairs = classifier.get_match_pairs()
            
            if match_pairs:
                match_data = []
                for e1, e2, conf in match_pairs:
                    match_data.append({
                        'entity1': e1,
                        'entity2': e2,
                        'confidence': conf
                    })
                
                match_df = pd.DataFrame(match_data)
                match_df.sort_values('confidence', ascending=False, inplace=True)
                match_df.to_csv(self.detailed_dir / 'match_pairs.csv', index=False)
        
        # 3. Clustering details
        if hasattr(clusterer, 'get_clusters'):
            clusters = clusterer.get_clusters()
            
            if clusters:
                # Full cluster assignments CSV
                cluster_data = []
                for cluster_id, entities in clusters.items():
                    for entity_idx, entity in enumerate(entities):
                        cluster_data.append({
                            'cluster_id': cluster_id,
                            'entity_id': entity,
                            'position_in_cluster': entity_idx + 1,
                            'cluster_size': len(entities)
                        })
                
                cluster_df = pd.DataFrame(cluster_data)
                cluster_df.to_csv(self.detailed_dir / 'clusters.csv', index=False)
                
                # Cluster summary CSV
                summary_data = []
                for cluster_id, entities in clusters.items():
                    summary_data.append({
                        'cluster_id': cluster_id,
                        'size': len(entities),
                        'entities_sample': ','.join(entities[:5]) + ('...' if len(entities) > 5 else '')
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
        
        logger.info("Detailed reports generated")
    
    def _generate_visualization_reports(self, preprocessor, embedder, indexer, classifier, clusterer):
        """Generate visualization reports"""
        logger.info("Generating visualization reports")
        
        # Set Seaborn style for better plots
        sns.set(style="whitegrid")
        
        # 1. Preprocessing visualizations
        try:
            # String frequency distribution
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
                top_strings = sorted(string_counts.items(), key=lambda x: x[1], reverse=True)[:20]
                if preprocessor.unique_strings and top_strings:
                    plt.figure(figsize=(12, 8))
                    
                    labels = [preprocessor.unique_strings.get(hash_val, hash_val)[:20] + '...' 
                             for hash_val, _ in top_strings]
                    values = [count for _, count in top_strings]
                    
                    plt.barh(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
                    plt.yticks(range(len(labels)), labels)
                    plt.title('Top 20 Most Frequent Strings', fontsize=14)
                    plt.xlabel('Frequency', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(self.visualizations_dir / 'top_strings.png', dpi=300)
                    plt.close()
        except Exception as e:
            logger.error(f"Error generating preprocessing visualizations: {e}")
        
        # 2. Classification visualizations
        try:
            if hasattr(classifier, 'get_match_pairs'):
                match_pairs = classifier.get_match_pairs()
                
                if match_pairs:
                    confidences = [conf for _, _, conf in match_pairs]
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(confidences, bins=20, alpha=0.7, color='forestgreen')
                    plt.axvline(classifier.match_threshold, color='red', linestyle='--', 
                               label=f'Threshold: {classifier.match_threshold:.2f}')
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
        except Exception as e:
            logger.error(f"Error generating classification visualizations: {e}")
        
        # 3. Clustering visualizations
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
                        plt.xticks(range(len(labels)), labels, rotation=45)
                        plt.title('Top 15 Largest Clusters', fontsize=14)
                        plt.ylabel('Cluster Size', fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(self.visualizations_dir / 'largest_clusters.png', dpi=300)
                        plt.close()
        except Exception as e:
            logger.error(f"Error generating clustering visualizations: {e}")
        
        logger.info("Visualization reports generated")
    
    def _generate_feature_analysis_reports(self, classifier):
        """Generate comprehensive feature analysis reports"""
        logger.info("Generating feature analysis reports")
        
        # Check if classifier has feature data
        if not hasattr(classifier, '_get_feature_importance'):
            logger.warning("Classifier does not have feature importance method")
            return
        
        try:
            # Get feature importance
            feature_importance = classifier._get_feature_importance()
            
            if not feature_importance:
                logger.warning("No feature importance data available")
                return
            
            # Create feature importance dataframe
            importance_data = []
            for feature, importance in feature_importance.items():
                importance_data.append({
                    'feature': feature,
                    'importance': importance,
                    'normalized_importance': importance / sum(feature_importance.values()) if sum(feature_importance.values()) > 0 else 0
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_df.sort_values('importance', ascending=False, inplace=True)
            
            # Save to CSV
            importance_df.to_csv(self.detailed_dir / 'feature_importance.csv', index=False)
            
            # Create feature importance visualization
            plt.figure(figsize=(12, 10))
            
            # Show top 15 features for readability
            top_features = importance_df.head(15)
            sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
            
            plt.title('Feature Importance', fontsize=16)
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
                other_sum = importance_df.iloc[8:]['importance'].sum()
                other_row = pd.DataFrame([{'feature': 'Other', 'importance': other_sum, 
                                         'normalized_importance': other_sum / sum(feature_importance.values())}])
                top_for_pie = pd.concat([top_for_pie, other_row])
            
            plt.pie(top_for_pie['importance'], labels=top_for_pie['feature'], autopct='%1.1f%%',
                   startangle=90, shadow=True, explode=[0.05]*len(top_for_pie))
            
            plt.title('Relative Feature Importance', fontsize=16)
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'feature_importance_pie.png', dpi=300)
            plt.close()
            
            # If available, get feature correlations from classifier
            if hasattr(classifier, 'get_state') and 'feature_correlation' in classifier.get_state():
                feature_correlation = classifier.get_state()['feature_correlation']
                
                # Create correlation matrix
                features = list(feature_importance.keys())
                corr_matrix = np.zeros((len(features), len(features)))
                
                for i, feature1 in enumerate(features):
                    for j, feature2 in enumerate(features):
                        if feature1 in feature_correlation and feature2 in feature_correlation[feature1]:
                            corr_matrix[i, j] = feature_correlation[feature1][feature2]
                
                # Create and save correlation matrix visualization
                plt.figure(figsize=(12, 10))
                
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                           xticklabels=features, yticklabels=features)
                
                plt.title('Feature Correlation Matrix', fontsize=16)
                plt.tight_layout()
                
                plt.savefig(self.visualizations_dir / 'feature_correlation.png', dpi=300)
                plt.close()
                
                # Create CSV report of highest correlations
                corr_data = []
                for i, feature1 in enumerate(features):
                    for j, feature2 in enumerate(features):
                        if i < j:  # Only include upper triangle to avoid duplicates
                            corr_data.append({
                                'feature1': feature1,
                                'feature2': feature2,
                                'correlation': corr_matrix[i, j],
                                'abs_correlation': abs(corr_matrix[i, j])
                            })
                
                corr_df = pd.DataFrame(corr_data)
                corr_df.sort_values('abs_correlation', ascending=False, inplace=True)
                corr_df.to_csv(self.detailed_dir / 'feature_correlations.csv', index=False)
            
            logger.info("Feature analysis reports generated")
        
        except Exception as e:
            logger.error(f"Error generating feature analysis reports: {e}")
    
    def _generate_misclassification_reports(self, classifier):
        """Generate detailed misclassification analysis reports"""
        logger.info("Generating misclassification reports")
        
        # Check if classifier has test data and predictions
        if not hasattr(classifier, 'get_state') or 'test_predictions' not in classifier.get_state():
            logger.warning("Classifier does not have test predictions available")
            return
        
        try:
            # Get test data and predictions
            state = classifier.get_state()
            
            if 'test_predictions' not in state or 'test_data' not in state or 'test_labels' not in state:
                logger.warning("Missing test data components")
                return
            
            predictions = state['test_predictions']
            X_test = state['test_data']
            y_test = state['test_labels']
            feature_names = state.get('feature_names', [f"feature_{i}" for i in range(X_test.shape[1])])
            
            # Identify misclassified examples
            misclassified = []
            
            for i in range(len(y_test)):
                if predictions[i] != y_test[i]:
                    misclassified_item = {
                        'instance_id': i,
                        'true_label': int(y_test[i]),
                        'predicted_label': int(predictions[i]),
                        'error_type': 'False Positive' if predictions[i] == 1 and y_test[i] == 0 
                                     else 'False Negative'
                    }
                    
                    # Add feature values
                    for j, name in enumerate(feature_names):
                        misclassified_item[f"feature_{name}"] = float(X_test[i, j])
                    
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
            
            sns.barplot(x='Error Type', y='Count', data=error_counts, palette='deep')
            
            plt.title('Misclassification Types', fontsize=16)
            plt.ylabel('Count', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'misclassification_types.png', dpi=300)
            plt.close()
            
            # 2. Feature distribution for misclassified vs. correctly classified examples
            if hasattr(classifier, 'get_feature_importance'):
                feature_importance = classifier._get_feature_importance()
                
                if feature_importance:
                    # Select top 5 features by importance
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_feature_names = [name for name, _ in top_features]
                    
                    # Create feature distribution visualizations
                    for feature in top_feature_names:
                        feature_col = f"feature_{feature}"
                        
                        if feature_col in misclassified_df.columns:
                            plt.figure(figsize=(12, 6))
                            
                            # Split by error type
                            false_pos = misclassified_df[misclassified_df['error_type'] == 'False Positive'][feature_col]
                            false_neg = misclassified_df[misclassified_df['error_type'] == 'False Negative'][feature_col]
                            
                            # Create density plot
                            if len(false_pos) > 0:
                                sns.kdeplot(false_pos, label='False Positive', shade=True, alpha=0.5)
                            if len(false_neg) > 0:
                                sns.kdeplot(false_neg, label='False Negative', shade=True, alpha=0.5)
                            
                            plt.title(f'Distribution of {feature} in Misclassified Examples', fontsize=16)
                            plt.xlabel('Feature Value', fontsize=14)
                            plt.ylabel('Density', fontsize=14)
                            plt.legend()
                            plt.grid(alpha=0.3)
                            plt.tight_layout()
                            
                            plt.savefig(self.visualizations_dir / f'misclassified_{feature}_distribution.png', dpi=300)
                            plt.close()
            
            logger.info("Misclassification reports generated")
        
        except Exception as e:
            logger.error(f"Error generating misclassification reports: {e}")
    
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
    
    def generate_test_set_csv_report(self, X_test, y_test, y_pred, feature_names, preprocessor=None):
        """Generate comprehensive CSV report with test set results, features, and predictions
        
        Args:
            X_test: Test set feature vectors
            y_test: True labels
            y_pred: Predicted labels
            feature_names: Feature names
            preprocessor: Optional preprocessor for additional metadata
        """
        logger.info("Generating comprehensive test set CSV report")
        
        try:
            # Create dataframe with all test data
            test_data = []
            
            for i in range(len(y_test)):
                item = {
                    'instance_id': i,
                    'true_label': int(y_test[i]),
                    'predicted_label': int(y_pred[i]),
                    'correctly_classified': y_test[i] == y_pred[i]
                }
                
                # Add all feature values
                for j, name in enumerate(feature_names):
                    item[f"feature_{name}"] = float(X_test[i][j])
                
                test_data.append(item)
            
            # Convert to dataframe
            test_df = pd.DataFrame(test_data)
            
            # Save to CSV
            test_df.to_csv(self.detailed_dir / 'test_set_with_features.csv', index=False)
            
            # Create a separate misclassified examples report
            misclassified_df = test_df[test_df['correctly_classified'] == False]
            misclassified_df.to_csv(self.detailed_dir / 'test_set_misclassified.csv', index=False)
            
            # Create summary
            summary = {
                'total_test_samples': len(test_df),
                'correctly_classified': len(test_df[test_df['correctly_classified'] == True]),
                'misclassified': len(misclassified_df),
                'accuracy': len(test_df[test_df['correctly_classified'] == True]) / len(test_df) if len(test_df) > 0 else 0,
                'false_positives': len(misclassified_df[(misclassified_df['true_label'] == 0) & (misclassified_df['predicted_label'] == 1)]),
                'false_negatives': len(misclassified_df[(misclassified_df['true_label'] == 1) & (misclassified_df['predicted_label'] == 0)])
            }
            
            # Save summary to JSON
            with open(self.detailed_dir / 'test_set_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Test set report generated with {len(test_df)} samples")
        
        except Exception as e:
            logger.error(f"Error generating test set CSV report: {e}")
    
    def generate_feature_correlation_reports(self, feature_correlation_matrix, feature_names=None):
        """Generate detailed feature correlation reports and visualizations
        
        Args:
            feature_correlation_matrix: Matrix of feature correlations
            feature_names: Optional list of feature names
        """
        logger.info("Generating feature correlation reports")
        
        try:
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
                for j in range(len(feature_names)):
                    if i != j:  # Skip self-correlations
                        all_correlations.append({
                            'feature1': feature_names[i],
                            'feature2': feature_names[j],
                            'correlation': feature_correlation_matrix[i, j],
                            'abs_correlation': abs(feature_correlation_matrix[i, j])
                        })
            
            corr_df = pd.DataFrame(all_correlations)
            corr_df.sort_values('abs_correlation', ascending=False, inplace=True)
            
            # Save to CSV
            corr_df.to_csv(self.detailed_dir / 'all_feature_correlations.csv', index=False)
            
            # Generate top correlations report
            top_correlations = corr_df.head(100)  # Top 100 correlations
            top_correlations.to_csv(self.detailed_dir / 'top_feature_correlations.csv', index=False)
            
            # Generate correlation heatmap
            plt.figure(figsize=(15, 12))
            
            mask = np.zeros_like(feature_correlation_matrix, dtype=bool)
            mask[np.triu_indices_from(mask)] = True  # Hide upper triangle for clarity
            
            sns.heatmap(feature_correlation_matrix, mask=mask, annot=True,
                       cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f',
                       xticklabels=feature_names, yticklabels=feature_names)
            
            plt.title('Feature Correlation Matrix', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(self.visualizations_dir / 'feature_correlation_heatmap.png', dpi=300)
            plt.close()
            
            # Generate feature clusters based on correlation
            from scipy.cluster import hierarchy
            from scipy.spatial.distance import squareform
            
            # Convert correlation to distance (1 - abs(correlation))
            corr_condensed = squareform(1 - np.abs(feature_correlation_matrix))
            
            z = hierarchy.linkage(corr_condensed, method='average')
            
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
            
            logger.info("Feature correlation reports generated")
        
        except Exception as e:
            logger.error(f"Error generating feature correlation reports: {e}")
    
    def generate_feature_distribution_reports(self, X, y, feature_names=None):
        """Generate detailed feature distribution reports and visualizations
        
        Args:
            X: Feature vectors
            y: Class labels
            feature_names: Optional list of feature names
        """
        logger.info("Generating feature distribution reports")
        
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            # Ensure dimensions match
            if X.shape[1] != len(feature_names):
                logger.error("Feature vector dimensions do not match feature names length")
                return
            
            # Calculate statistics for each feature
            feature_stats = []
            
            for i, name in enumerate(feature_names):
                # Get values for this feature
                values = X[:, i]
                
                # Split by class
                pos_values = values[y == 1]
                neg_values = values[y == 0]
                
                # Calculate statistics
                stats = {
                    'feature': name,
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'positive_mean': np.mean(pos_values) if len(pos_values) > 0 else float('nan'),
                    'negative_mean': np.mean(neg_values) if len(neg_values) > 0 else float('nan'),
                    'positive_std': np.std(pos_values) if len(pos_values) > 0 else float('nan'),
                    'negative_std': np.std(neg_values) if len(neg_values) > 0 else float('nan'),
                    'separation': abs(np.mean(pos_values) - np.mean(neg_values)) / 
                                 (np.std(pos_values) + np.std(neg_values))
                                 if len(pos_values) > 0 and len(neg_values) > 0 and
                                    (np.std(pos_values) + np.std(neg_values)) > 0
                                 else float('nan')
                }
                
                feature_stats.append(stats)
            
            # Convert to dataframe and sort by separation
            stats_df = pd.DataFrame(feature_stats)
            stats_df.sort_values('separation', ascending=False, inplace=True, na_position='last')
            
            # Save to CSV
            stats_df.to_csv(self.detailed_dir / 'feature_distributions.csv', index=False)
            
            # Generate distribution plots for top features (by separation)
            top_features = stats_df.head(10)
            
            for _, row in top_features.iterrows():
                feature = row['feature']
                feature_idx = feature_names.index(feature)
                
                plt.figure(figsize=(10, 6))
                
                # Get values for this feature
                values = X[:, feature_idx]
                pos_values = values[y == 1]
                neg_values = values[y == 0]
                
                # Plot distributions
                sns.histplot(pos_values, color='green', alpha=0.5, label='Positive Class', 
                            kde=True, stat='density')
                sns.histplot(neg_values, color='red', alpha=0.5, label='Negative Class', 
                            kde=True, stat='density')
                
                # Add means
                plt.axvline(x=row['positive_mean'], color='green', linestyle='--', 
                           label=f"Positive Mean: {row['positive_mean']:.4f}")
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
            
            # Create combined plot of top 2 features
            if len(top_features) >= 2:
                feature1 = top_features.iloc[0]['feature']
                feature2 = top_features.iloc[1]['feature']
                
                feature1_idx = feature_names.index(feature1)
                feature2_idx = feature_names.index(feature2)
                
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
            
            logger.info("Feature distribution reports generated")
        
        except Exception as e:
            logger.error(f"Error generating feature distribution reports: {e}")
