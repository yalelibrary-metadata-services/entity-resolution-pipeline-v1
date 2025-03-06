#!/usr/bin/env python
"""
Entity Resolution Pipeline for Yale University Library Catalog
Main entry point script for running the pipeline
"""

import os
import sys
import time
import yaml
import click
import logging
from pathlib import Path
from dotenv import load_dotenv
from prometheus_client import start_http_server

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline modules
from src.pipeline import Pipeline
from src.utils import setup_logging, setup_monitoring

# Load environment variables from .env file
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

def load_config(config_path, mode=None):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override mode if specified
    if mode:
        config['general']['mode'] = mode
    
    return config

@click.command()
@click.option('--config', default='config.yml', help='Path to the configuration file')
@click.option('--stage', default='all', 
              type=click.Choice(['preprocessing', 'embedding', 'indexing', 
                                'training', 'classification', 'clustering', 
                                'reporting', 'all']),
              help='Pipeline stage to run')
@click.option('--mode', default=None, 
              type=click.Choice(['development', 'production']),
              help='Runtime mode (development or production)')
@click.option('--reset', is_flag=True, help='Reset checkpoints and start fresh')
def main(config, stage, mode, reset):
    """Run the entity resolution pipeline"""
    start_time = time.time()
    
    # Load configuration
    config_data = load_config(config, mode)
    
    # Setup logging
    setup_logging(config_data)
    
    # Start Prometheus metrics server if enabled
    if config_data['monitoring']['prometheus']['enabled']:
        metrics_port = config_data['monitoring']['prometheus']['port']
        start_http_server(metrics_port)
        logger.info(f"Started Prometheus metrics server on port {metrics_port}")
    
    # Initialize the pipeline
    pipeline = Pipeline(config_data)
    
    # Run the requested pipeline stage
    if stage == 'all':
        logger.info("Running full pipeline...")
        pipeline.run_all(reset=reset)
    elif stage == 'preprocessing':
        logger.info("Running preprocessing stage...")
        pipeline.run_preprocessing(reset=reset)
    elif stage == 'embedding':
        logger.info("Running embedding stage...")
        pipeline.run_embedding(reset=reset)
    elif stage == 'indexing':
        logger.info("Running indexing stage...")
        pipeline.run_indexing(reset=reset)
    elif stage == 'training':
        logger.info("Running training stage...")
        pipeline.run_training(reset=reset)
    elif stage == 'classification':
        logger.info("Running classification stage...")
        pipeline.run_classification(reset=reset)
    elif stage == 'clustering':
        logger.info("Running clustering stage...")
        pipeline.run_clustering(reset=reset)
    elif stage == 'reporting':
        logger.info("Running reporting stage...")
        pipeline.run_reporting()
    
    # Log total execution time
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
