#!/usr/bin/env python
"""
Entity Resolution Pipeline for Yale University Library Catalog
Main entry point script for running the pipeline with enhanced progress tracking
and proper resource management
"""

import os
import sys
import time
import tqdm
import yaml
import click
import logging
from pathlib import Path
from dotenv import load_dotenv
from prometheus_client import start_http_server
from contextlib import ExitStack

# Configure logging - reduce noise from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline modules
from src.pipeline import Pipeline
from src.utils import setup_logging, setup_monitoring
from src.utils_enhancement import diagnose_imputation_issues

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

def print_header(title):
    """Print a formatted header for progress tracking"""
    line = "=" * 80
    print(f"\n{line}")
    print(f"  {title.upper()}")
    print(f"{line}")

def print_stage_progress(stage, status, time_taken=None):
    """Print stage progress information"""
    if time_taken:
        print(f"  → {stage}: {status} ({time_taken:.2f} seconds)")
    else:
        print(f"  → {stage}: {status}")

def print_summary(pipeline_stages, stage_times):
    """Print a summary of pipeline execution"""
    line = "=" * 80
    print(f"\n{line}")
    print("  PIPELINE EXECUTION SUMMARY")
    print(f"{line}")
    
    total_time = sum(stage_times.values())
    
    for stage in pipeline_stages:
        if stage in stage_times:
            time_taken = stage_times[stage]
            percentage = (time_taken / total_time) * 100 if total_time > 0 else 0
            print(f"  {stage.ljust(15)}: {time_taken:.2f} seconds ({percentage:.1f}%)")
    
    print(f"{line}")
    print(f"  Total execution time: {total_time:.2f} seconds")
    print(f"{line}\n")

def cleanup_resources(pipeline):
    """Ensure all resources are properly cleaned up"""
    if hasattr(pipeline, 'query_engine'):
        # Close Weaviate connection if exists
        if hasattr(pipeline.query_engine, 'close'):
            try:
                pipeline.query_engine.close()
                logger.info("Weaviate connection properly closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")
        
        # Force cleanup of any lingering connections
        try:
            import gc
            gc.collect()
            logger.debug("Garbage collection performed")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")

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
    """Run the entity resolution pipeline with enhanced progress tracking"""
    start_time = time.time()
    
    # Load configuration
    config_data = load_config(config, mode)
    
    # Setup logging
    setup_logging(config_data)
    
    print_header(f"Entity Resolution Pipeline - Stage: {stage}, Mode: {config_data['general']['mode']}")
    print(f"  Configuration: {config}")
    print(f"  Reset checkpoints: {'Yes' if reset else 'No'}")
    
    # Start Prometheus metrics server if enabled
    if config_data['monitoring']['prometheus']['enabled']:
        metrics_port = config_data['monitoring']['prometheus']['port']
        try:
            start_http_server(metrics_port)
            print_stage_progress("Monitoring", f"Started Prometheus metrics server on port {metrics_port}")
            logger.info(f"Started Prometheus metrics server on port {metrics_port}")
        except Exception as e:
            print_stage_progress("Monitoring", f"Failed to start Prometheus server: {str(e)}")
            logger.error(f"Failed to start Prometheus server: {e}")
    
    # Dictionary to track stage execution times
    stage_times = {}
    pipeline_stages = ['preprocessing', 'embedding', 'indexing', 
                      'training', 'classification', 'clustering', 'reporting']
    
    # Use ExitStack to ensure proper resource cleanup
    with ExitStack() as stack:
        # Initialize the pipeline
        print_stage_progress("Initialization", "Creating pipeline instance")
        pipeline = Pipeline(config_data)
        
        # Register cleanup to always run, even if an exception occurs
        stack.callback(lambda: cleanup_resources(pipeline))
        
        # Run the requested pipeline stage
        try:
            if stage == 'all':
                print_header("Running Full Pipeline")
                
                for current_stage in pipeline_stages:
                    stage_start_time = time.time()
                    
                    print_stage_progress(current_stage, "Starting...")
                    
                    if current_stage == 'preprocessing':
                        pipeline.run_preprocessing(reset=reset)
                    elif current_stage == 'embedding':
                        pipeline.run_embedding(reset=reset)
                    elif current_stage == 'indexing':
                        pipeline.run_indexing(reset=reset)
                    elif current_stage == 'training':
                        pipeline.run_training(reset=reset)
                    elif current_stage == 'classification':
                        pipeline.run_classification(reset=reset)
                    elif current_stage == 'clustering':
                        pipeline.run_clustering(reset=reset)
                    elif current_stage == 'reporting':
                        pipeline.run_reporting()
                    
                    stage_time = time.time() - stage_start_time
                    stage_times[current_stage] = stage_time
                    
                    print_stage_progress(current_stage, "Completed", stage_time)
            else:
                print_header(f"Running Stage: {stage}")
                stage_start_time = time.time()
                
                if stage == 'preprocessing':
                    pipeline.run_preprocessing(reset=reset)
                elif stage == 'embedding':
                    pipeline.run_embedding(reset=reset)
                elif stage == 'indexing':
                    pipeline.run_indexing(reset=reset)
                elif stage == 'training':
                    pipeline.run_training(reset=reset)
                elif stage == 'classification':
                    pipeline.run_classification(reset=reset)
                elif stage == 'clustering':
                    pipeline.run_clustering(reset=reset)
                elif stage == 'reporting':
                    pipeline.run_reporting()
                
                stage_time = time.time() - stage_start_time
                stage_times[stage] = stage_time
                
                print_stage_progress(stage, "Completed", stage_time)
        
        except Exception as e:
            print_header("PIPELINE ERROR")
            print(f"  An error occurred during execution: {str(e)}")
            logger.error(f"Pipeline execution error: {e}", exc_info=True)
            
            # Print partial summary if available
            if stage_times:
                print_summary(pipeline_stages, stage_times)
            
            sys.exit(1)
    
    # Log total execution time
    total_time = time.time() - start_time
    
    # Print execution summary
    print_summary(pipeline_stages, stage_times)
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
