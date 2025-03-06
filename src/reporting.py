"""
Reporting module for entity resolution
Handles generation of reports and visualizations
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

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Reporter:
    """Reporter for entity resolution pipeline"""
    
    def __init__(self, config):
        """Initialize the reporter with configuration"""
        self.config = config
        self.output_formats = config['reporting']['output_format']
        self.detailed_metrics = config['reporting']['detailed_metrics']
        self.save_misclassified = config['reporting']['save_misclassified']
    
    def generate_reports(self, preprocessor, embedder, indexer, classifier, clusterer, output_dir):
        """Generate reports for pipeline results"""
        with Timer() as timer:
            logger.info("Generating reports")
            
            # Ensure output directory exists
            ensure_dir(output_dir)
            
            # Generate summary report
            self._generate_summary_report(
                preprocessor, embedder, indexer, classifier, clusterer, output_dir
            )
            
            # Generate detailed reports if enabled
            if self.detailed_metrics:
                self._generate_detailed_reports(
                    preprocessor, embedder, indexer, classifier, clusterer, output_dir
                )
        
        logger.info(f"Report generation completed in {timer.elapsed:.2f} seconds")
    
    def _generate_summary_report(self, preprocessor, embedder, indexer, classifier, clusterer, output_dir):
        """Generate summary report"""
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
                'matches': len(classifier.get_match_pairs()),
                'threshold': self.config['classification']['match_threshold']
            },
            'clustering': {
                'clusters': len(clusterer.get_clusters()),
                'algorithm': self.config['clustering']['algorithm']
            },
            'config': {
                'mode': self.config['general']['mode'],
                'openai_model': self.config['openai']['embedding_model'],
                'feature_config': self.config['features']
            }
        }
        
        # Save summary report in requested formats
        for format_type in self.output_formats:
            if format_type == 'json':
                with open(output_dir / 'summary_report.json', 'w') as f:
                    json.dump(summary, f, indent=2)
            
            elif format_type == 'csv':
                # Flatten nested dictionary for CSV
                flat_summary = self._flatten_dict(summary)
                
                with open(output_dir / 'summary_report.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['key', 'value'])
                    for key, value in flat_summary.items():
                        writer.writerow([key, value])
        
        # Generate summary visualization
        self._generate_summary_visualization(summary, output_dir)
    
    def _generate_detailed_reports(self, preprocessor, embedder, indexer, classifier, clusterer, output_dir):
        """Generate detailed reports"""
        logger.info("Generating detailed reports")
        
        # Create detailed reports directory
        detailed_dir = output_dir / 'detailed'
        ensure_dir(detailed_dir)
        
        # Entity clusters report
        self._generate_clusters_report(clusterer, detailed_dir)
        
        # Match pairs report
        self._generate_match_pairs_report(classifier, detailed_dir)
        
        # Classification metrics report
        self._generate_classification_metrics_report(classifier, detailed_dir)
        
        # If requested, save misclassified examples
        if self.save_misclassified:
            self._generate_misclassified_report(classifier, detailed_dir)
    
    def _generate_clusters_report(self, clusterer, output_dir):
        """Generate clusters report"""
        clusters = clusterer.get_clusters()
        
        # Format cluster data
        cluster_data = []
        for cluster_id, entities in clusters.items():
            cluster_data.append({
                'cluster_id': cluster_id,
                'size': len(entities),
                'entities': entities
            })
        
        # Save in requested formats
        for format_type in self.output_formats:
            if format_type == 'json':
                with open(output_dir / 'clusters.json', 'w') as f:
                    json.dump(cluster_data, f, indent=2)
            
            elif format_type == 'csv':
                # For CSV, we'll create a flattened version
                rows = []
                for cluster in cluster_data:
                    for entity in cluster['entities']:
                        rows.append([cluster['cluster_id'], entity])
                
                with open(output_dir / 'clusters.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['cluster_id', 'entity_id'])
                    writer.writerows(rows)
    
    def _generate_match_pairs_report(self, classifier, output_dir):
        """Generate match pairs report"""
        match_pairs = classifier.get_match_pairs()
        
        # Save in requested formats
        for format_type in self.output_formats:
            if format_type == 'json':
                match_data = [
                    {'entity1': e1, 'entity2': e2, 'confidence': float(conf)}
                    for e1, e2, conf in match_pairs
                ]
                
                with open(output_dir / 'match_pairs.json', 'w') as f:
                    json.dump(match_data, f, indent=2)
            
            elif format_type == 'csv':
                with open(output_dir / 'match_pairs.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['entity1', 'entity2', 'confidence'])
                    for e1, e2, conf in match_pairs:
                        writer.writerow([e1, e2, conf])
    
    def _generate_classification_metrics_report(self, classifier, output_dir):
        """Generate classification metrics report"""
        metrics = classifier.metrics
        
        if not metrics:
            logger.warning("No classification metrics available")
            return
        
        # Save in requested formats
        for format_type in self.output_formats:
            if format_type == 'json':
                with open(output_dir / 'classification_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            elif format_type == 'csv':
                # Flatten metrics for CSV
                flat_metrics = self._flatten_dict(metrics)
                
                with open(output_dir / 'classification_metrics.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric', 'value'])
                    for key, value in flat_metrics.items():
                        writer.writerow([key, value])
    
    def _generate_misclassified_report(self, classifier, output_dir):
        """Generate report of misclassified examples"""
        # Note: This would require ground truth data to actually identify misclassifications
        # For now, we'll just create a placeholder
        logger.info("Skipping misclassified report (requires evaluation with ground truth)")
    
    def _generate_summary_visualization(self, summary, output_dir):
        """Generate summary visualization"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Create subplots
            plt.subplot(2, 2, 1)
            plt.bar(['Unique Strings', 'Records', 'Person IDs'], 
                   [summary['preprocessing']['unique_strings'], 
                    summary['preprocessing']['records'], 
                    summary['preprocessing']['person_ids']])
            plt.title('Preprocessing Statistics')
            plt.yscale('log')
            
            plt.subplot(2, 2, 2)
            plt.bar(['Matches', 'Clusters'], 
                   [summary['classification']['matches'], 
                    summary['clustering']['clusters']])
            plt.title('Matching and Clustering')
            plt.yscale('log')
            
            plt.subplot(2, 2, 3)
            # If cluster sizes were included, we could visualize distribution
            # For now, just a placeholder
            plt.text(0.5, 0.5, f"Clustering Algorithm: {summary['clustering']['algorithm']}\n"
                             f"Total Clusters: {summary['clustering']['clusters']}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.title('Clustering')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, f"Mode: {summary['config']['mode']}\n"
                             f"Embedding Model: {summary['config']['openai_model']}\n"
                             f"Match Threshold: {summary['classification']['threshold']}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=plt.gca().transAxes)
            plt.title('Configuration')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'summary_report.png')
            plt.close()
        
        except Exception as e:
            logger.error(f"Error generating summary visualization: {e}")
    
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
