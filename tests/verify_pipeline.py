#!/usr/bin/env python
"""
Verification tests for entity resolution pipeline
Used to validate the pipeline functionality on a small dataset
"""

import os
import sys
import yaml
import logging
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline modules
from src.pipeline import Pipeline
from src.preprocessing import Preprocessor
from src.embedding import Embedder
from src.indexing import Indexer
from src.imputation import Imputer
from src.querying import QueryEngine
from src.features import FeatureExtractor
from src.classification import Classifier
from src.clustering import Clusterer
from src.utils import Timer, ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_config():
    """Load test configuration"""
    config_path = Path(__file__).parent.parent / 'config.yml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings for testing
    config['general']['mode'] = 'development'
    config['development']['max_files'] = 2
    config['development']['max_records'] = 100
    
    return config

def verify_preprocessing(config):
    """Verify preprocessing functionality"""
    logger.info("Verifying preprocessing")
    
    with Timer() as timer:
        preprocessor = Preprocessor(config)
        preprocessor.process()
        
        # Verify results
        unique_strings = preprocessor.get_unique_strings()
        string_counts = preprocessor.get_string_counts()
        record_field_hashes = preprocessor.get_record_field_hashes()
        
        logger.info(f"Preprocessing results: {len(unique_strings)} unique strings, "
                  f"{len(record_field_hashes)} records")
        
        assert len(unique_strings) > 0, "No unique strings found"
        assert len(string_counts) > 0, "No string counts found"
        assert len(record_field_hashes) > 0, "No record field hashes found"
    
    logger.info(f"Preprocessing verification completed in {timer.elapsed:.2f} seconds")
    return preprocessor

def verify_embedding(config, preprocessor):
    """Verify embedding functionality"""
    logger.info("Verifying embedding")
    
    with Timer() as timer:
        embedder = Embedder(config)
        embedder.embed(preprocessor.get_unique_strings())
        
        # Verify results
        embeddings = embedder.get_embeddings()
        
        logger.info(f"Embedding results: {len(embeddings)} embeddings")
        
        assert len(embeddings) > 0, "No embeddings found"
        
        # Check embedding dimensions
        first_embedding = next(iter(embeddings.values()))
        assert len(first_embedding) == config['openai']['embedding_dim'], \
            f"Unexpected embedding dimension: {len(first_embedding)}"
    
    logger.info(f"Embedding verification completed in {timer.elapsed:.2f} seconds")
    return embedder

def verify_indexing(config, preprocessor, embedder):
    """Verify indexing functionality"""
    logger.info("Verifying indexing")
    
    with Timer() as timer:
        indexer = Indexer(config)
        indexer.index(
            preprocessor.get_unique_strings(),
            preprocessor.get_field_mapping(),
            embedder.get_embeddings()
        )
        
        # Verify results
        assert indexer.is_indexed(), "Indexing failed"
        
        # Get collection and check if it exists
        collection = indexer.get_collection()
        assert collection is not None, "Collection not found"
    
    logger.info(f"Indexing verification completed in {timer.elapsed:.2f} seconds")
    return indexer

def verify_querying(config, indexer):
    """Verify querying functionality"""
    logger.info("Verifying querying")
    
    with Timer() as timer:
        query_engine = QueryEngine(config)
        query_engine.set_collection(indexer.get_collection())
        
        # Test query on a random vector
        import numpy as np
        test_vector = np.random.randn(config['openai']['embedding_dim'])
        test_vector = test_vector / np.linalg.norm(test_vector)  # Normalize
        
        results = query_engine.query_nearest_vectors(
            test_vector,
            limit=5,
            min_similarity=0.0  # Set to 0 for testing
        )
        
        logger.info(f"Query results: {len(results)} results")
        
        # Results might be empty if no vectors are similar enough
        # but the query should execute without errors
    
    logger.info(f"Querying verification completed in {timer.elapsed:.2f} seconds")
    return query_engine

def verify_features(config, preprocessor, query_engine):
    """Verify feature extraction functionality"""
    logger.info("Verifying feature extraction")
    
    with Timer() as timer:
        feature_extractor = FeatureExtractor(config)
        
        # Get sample records
        person_ids = preprocessor.get_all_person_ids()
        if len(person_ids) >= 2:
            record1 = preprocessor.get_record(person_ids[0])
            record2 = preprocessor.get_record(person_ids[1])
            
            # Extract features
            feature_vector = feature_extractor.extract_features(record1, record2, query_engine)
            
            logger.info(f"Feature vector shape: {feature_vector.shape}")
            
            assert feature_vector.shape[0] > 0, "Empty feature vector"
        else:
            logger.warning("Not enough records to test feature extraction")
    
    logger.info(f"Feature extraction verification completed in {timer.elapsed:.2f} seconds")
    return feature_extractor

def verify_imputation(config, preprocessor, query_engine):
    """Verify imputation functionality"""
    logger.info("Verifying imputation")
    
    with Timer() as timer:
        imputer = Imputer(config)
        
        # Get sample record with null values if available
        person_ids = preprocessor.get_all_person_ids()
        null_record = None
        
        for pid in person_ids:
            record = preprocessor.get_record(pid)
            for field in config['fields']['impute']:
                if field in record and record[field] == "NULL":
                    null_record = record
                    break
            if null_record:
                break
        
        if null_record:
            # Try imputation
            imputed_record = imputer.impute_record(null_record, query_engine)
            
            logger.info(f"Imputation result: {null_record} -> {imputed_record}")
            
            # Imputation might not succeed, but it should run without errors
        else:
            logger.warning("No records with null values found for imputation testing")
    
    logger.info(f"Imputation verification completed in {timer.elapsed:.2f} seconds")
    return imputer

def verify_pipeline(config):
    """Verify complete pipeline functionality"""
    logger.info("Verifying complete pipeline")
    
    with Timer() as timer:
        pipeline = Pipeline(config)
        
        # Run pipeline stages
        pipeline.run_preprocessing(reset=True)
        pipeline.run_embedding(reset=True)
        pipeline.run_indexing(reset=True)
        
        # Training and classification require ground truth data
        # We'll skip them in the verification test unless sample data is available
        ground_truth_file = Path(config['dataset']['ground_truth_file'])
        
        if ground_truth_file.exists():
            pipeline.run_training(reset=True)
            pipeline.run_classification(reset=True)
            pipeline.run_clustering(reset=True)
            pipeline.run_reporting()
            logger.info("Complete pipeline verification successful")
        else:
            logger.warning(f"Ground truth file {ground_truth_file} not found, "
                         "skipping training and subsequent stages")
    
    logger.info(f"Pipeline verification completed in {timer.elapsed:.2f} seconds")
    return pipeline

def main():
    """Main verification function"""
    logger.info("Starting entity resolution pipeline verification")
    
    # Load test configuration
    config = load_test_config()
    
    try:
        # Verify individual components
        preprocessor = verify_preprocessing(config)
        embedder = verify_embedding(config, preprocessor)
        indexer = verify_indexing(config, preprocessor, embedder)
        query_engine = verify_querying(config, indexer)
        feature_extractor = verify_features(config, preprocessor, query_engine)
        imputer = verify_imputation(config, preprocessor, query_engine)
        
        # Verify complete pipeline
        pipeline = verify_pipeline(config)
        
        logger.info("All verification tests passed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
