"""
Pipeline orchestration module for entity resolution
Optimized version with batch processing and parallel feature extraction
"""

import os
import logging
import time
import tqdm
import numpy as np
from pathlib import Path

# Import pipeline modules
from .preprocessing import Preprocessor
from .embedding import Embedder
from .indexing import Indexer
from .imputation import Imputer
from .classification import Classifier
from .clustering import Clusterer
from .reporting import Reporter
from .analysis import Analyzer

# Import optimized components
from .optimized_query import OptimizedQueryEngine
from .batch_processor import BatchProcessor
from .parallel_features import ParallelFeatureExtractor
from .utils_enhancement import monitor_memory_usage

# Import original utilities
from .utils import Timer, ensure_dir, load_checkpoint, save_checkpoint

# Configure logger
logger = logging.getLogger(__name__)

class Pipeline:
    """Pipeline orchestrator for entity resolution"""
    
    def __init__(self, config):
        """Initialize the pipeline with configuration"""
        self.config = config
        self.mode = config['general']['mode']
        self.checkpoint_dir = Path(config['general']['checkpoint_dir'])
        self.output_dir = Path(config['general']['output_dir'])
        
        # Ensure directories exist
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.output_dir)
        
        # Initialize original pipeline components
        self.preprocessor = Preprocessor(config)
        self.embedder = Embedder(config)
        self.indexer = Indexer(config)
        self.imputer = Imputer(config)
        self.classifier = Classifier(config)
        self.clusterer = Clusterer(config)
        self.reporter = Reporter(config)
        self.analyzer = Analyzer(config)
        
        # Initialize optimized components
        self.query_engine = OptimizedQueryEngine(config)
        self.feature_extractor = ParallelFeatureExtractor(config)
        self.batch_processor = BatchProcessor(config)
        
        logger.info(f"Optimized pipeline initialized in {self.mode} mode")
    
    def run_all(self, reset=False):
        """Run the complete pipeline"""
        try:
            with Timer() as timer:
                self.run_preprocessing(reset=reset)
                self.run_embedding(reset=reset)
                self.run_indexing(reset=reset)
                self.run_training(reset=reset)
                self.run_classification(reset=reset)
                self.run_clustering(reset=reset)
                self.run_reporting()
            
            logger.info(f"Complete pipeline executed in {timer.elapsed:.2f} seconds")
        finally:
            # Ensure Weaviate connection is closed
            if hasattr(self, 'query_engine') and hasattr(self.query_engine, 'close'):
                self.query_engine.close()
    
    def run_preprocessing(self, reset=False):
        """Run the preprocessing stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "preprocessing.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading preprocessing results from checkpoint")
                preprocessing_data = load_checkpoint(checkpoint_path)
                self.preprocessor.load_state(preprocessing_data)
            else:
                logger.info("Running preprocessing")
                self.preprocessor.process()
                save_checkpoint(checkpoint_path, self.preprocessor.get_state())
        
        logger.info(f"Preprocessing completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on preprocessing results
        self.analyzer.analyze_preprocessing(self.preprocessor)
    
    def run_embedding(self, reset=False):
        """Run the embedding stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "embedding.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading embedding results from checkpoint")
                embedding_data = load_checkpoint(checkpoint_path)
                self.embedder.load_state(embedding_data)
            else:
                logger.info("Running embedding generation")
                # Ensure preprocessor data is available
                if not self.preprocessor.is_processed():
                    self.run_preprocessing()
                
                self.embedder.embed(self.preprocessor.get_unique_strings())
                save_checkpoint(checkpoint_path, self.embedder.get_state())
        
        logger.info(f"Embedding completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on embedding results
        self.analyzer.analyze_embedding(self.embedder)
    
    def run_indexing(self, reset=False):
        """Run the indexing stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "indexing.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading indexing results from checkpoint")
                indexing_data = load_checkpoint(checkpoint_path)
                self.indexer.load_state(indexing_data)
            else:
                logger.info("Running indexing")
                # Ensure embedder data is available
                if not self.embedder.is_processed():
                    self.run_embedding()
                
                self.indexer.index(
                    self.preprocessor.get_unique_strings(),
                    self.preprocessor.get_field_mapping(),
                    self.embedder.get_embeddings()
                )
                save_checkpoint(checkpoint_path, self.indexer.get_state())
            
            # Important: Set the collection in the query engine
            logger.info("Setting collection in query engine after indexing")
            self.query_engine.set_collection(self.indexer.get_collection())
            
            # Run diagnostics if imputation is enabled
            if self.config['imputation']['enabled']:
                from .utils_enhancement import diagnose_imputation_issues
                logger.info("Running imputation diagnostics after indexing...")
                diagnose_imputation_issues(self.query_engine, "subjects")
        
        logger.info(f"Indexing completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on indexing results
        self.analyzer.analyze_indexing(self.indexer)
    
    def run_training(self, reset=False):
        """Run the training stage for classification with optimized processing and improved progress tracking"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "training.pkl"
            
            print("\n=== STARTING TRAINING STAGE ===")
            
            if not reset and checkpoint_path.exists():
                print("Loading training results from checkpoint")
                logger.info("Loading training results from checkpoint")
                training_data = load_checkpoint(checkpoint_path)
                self.classifier.load_state(training_data)
            else:
                print("Running classifier training with optimized batch processing")
                logger.info("Running classifier training with optimized batch processing")
                
                # Ensure indexing is complete
                if not self.indexer.is_indexed():
                    print("Indexing not complete, running indexing first")
                    self.run_indexing()
                
                # Set collection in optimized query engine
                print("Setting collection in optimized query engine")
                self.query_engine.set_collection(self.indexer.get_collection())
                
                # Load ground truth data
                print("Loading ground truth data")
                ground_truth_file = self.config['dataset']['ground_truth_file']
                record_pairs, labels = self.preprocessor.load_ground_truth(ground_truth_file)
                print(f"Loaded {len(record_pairs)} labeled pairs: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
                
                # Prepare record pairs with progress tracking
                print("Preparing record pairs...")
                prepared_pairs = []
                valid_count = 0
                invalid_count = 0
                
                # Import tqdm properly
                from tqdm import tqdm as tqdm_func
                
                for idx, (left_id, right_id) in enumerate(tqdm_func(record_pairs, desc="Preparing record pairs")):
                    left_record = self.preprocessor.get_record(left_id)
                    right_record = self.preprocessor.get_record(right_id)
                    
                    if left_record and right_record:
                        # Add personId to the records for reference
                        left_record = left_record.copy()  # Make a copy to avoid modifying original
                        right_record = right_record.copy()
                        left_record['personId'] = left_id
                        right_record['personId'] = right_id
                        prepared_pairs.append((left_record, right_record))
                        valid_count += 1
                    else:
                        invalid_count += 1
                        
                    # Log progress periodically
                    if (idx + 1) % 5000 == 0 or idx == len(record_pairs) - 1:
                        logger.info(f"Prepared {valid_count} valid pairs, {invalid_count} invalid pairs " +
                                f"({(idx+1)}/{len(record_pairs)} processed)")
                
                print(f"Prepared {len(prepared_pairs)} valid record pairs for feature extraction")
                
                # Extract features using batch processor with memory monitoring
                monitor_enabled = self.config.get('batch_processing', {}).get('monitor_memory', False)
                if monitor_enabled:
                    print("Memory monitoring enabled during processing")
                    interval = self.config.get('batch_processing', {}).get('monitoring_interval', 30)
                    memory_context = monitor_memory_usage(interval)
                else:
                    # Create a no-op context manager
                    from contextlib import nullcontext
                    memory_context = nullcontext()
                
                with memory_context:
                    print("Starting optimized batch feature extraction")
                    feature_vectors = self.batch_processor.process_record_pairs(
                        prepared_pairs,
                        self.preprocessor,
                        self.query_engine,
                        self.feature_extractor,
                        self.imputer
                    )
                
                print(f"Feature extraction complete: {len(feature_vectors)} vectors")
                
                
                # Make sure we align labels with the successfully extracted features
                if len(feature_vectors) < len(prepared_pairs):
                    print(f"Warning: {len(prepared_pairs) - len(feature_vectors)} feature extractions failed")
                    print("Realigning labels with successfully extracted feature vectors...")
                    
                    # We need to create a mapping from the successful extractions back to the original labels
                    # Since we don't know which extractions failed, we'll need to re-extract features
                    # just to identify which ones succeeded
                    valid_indices = []
                    
                    for idx, (left_record, right_record) in enumerate(tqdm_func(prepared_pairs, desc="Identifying successful extractions")):
                        # Quick check if features can be extracted (without actually extracting them)
                        valid = True
                        for field in self.feature_extractor.fields_to_embed:
                            left_key = (left_record.get(field), field) if field in left_record and left_record[field] != "NULL" else None
                            right_key = (right_record.get(field), field) if field in right_record and right_record[field] != "NULL" else None
                            
                            if (left_key is not None and left_key not in self.query_engine.vector_memory_cache) or \
                            (right_key is not None and right_key not in self.query_engine.vector_memory_cache):
                                valid = False
                                break
                        
                        if valid:
                            valid_indices.append(idx)
                    
                    if len(valid_indices) == len(feature_vectors):
                        print("Successfully identified failing extractions")
                        adjusted_labels = [labels[i] for i in valid_indices]
                        
                        # Create record pairs list for successfully extracted features
                        successful_pairs = [record_pairs[valid_indices[i]] for i in range(len(valid_indices))]
                    else:
                        print("Warning: Could not precisely identify failing extractions.")
                        print(f"Expected {len(feature_vectors)} valid indices but found {len(valid_indices)}")
                        print("Using first N labels as a fallback (may introduce label misalignment)")
                        adjusted_labels = labels[:len(feature_vectors)]
                        successful_pairs = record_pairs[:len(feature_vectors)]
                else:
                    adjusted_labels = labels
                    successful_pairs = record_pairs
                
                # Convert to numpy array for training
                X = np.array(feature_vectors)
                y = np.array(adjusted_labels[:len(feature_vectors)])  # Match labels to feature vectors
                
                print(f"Feature matrix shape: {X.shape}")
                
                # Train RFE if enabled
                if self.config['features']['recursive_feature_elimination']['enabled']:
                    print("Training Recursive Feature Elimination model")
                    logger.info("Training Recursive Feature Elimination model")
                    self.feature_extractor.train_rfe(X, y)
                    
                    # Get updated feature vectors if RFE is enabled
                    if self.feature_extractor.selected_feature_indices is not None:
                        X = X[:, self.feature_extractor.selected_feature_indices]
                        print(f"Using {X.shape[1]} features selected by RFE")
                        logger.info(f"Using {X.shape[1]} features selected by RFE")
                
                # Train classifier with record pairs for enhanced reporting
                print("Starting classifier training")
                self.classifier.train(X, y, self.feature_extractor.get_feature_names(), successful_pairs)
                
                print("Saving checkpoint")
                save_checkpoint(checkpoint_path, self.classifier.get_state())
                
                # Clear caches to free memory
                self.feature_extractor.clear_caches()
                self.query_engine.clear_cache()
            
            print("=== TRAINING STAGE COMPLETED ===\n")
        
        logger.info(f"Training completed in {timer.elapsed:.2f} seconds")
    
    def run_classification(self, reset=False):
        """Run the classification stage with optimized batch processing"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "classification.pkl"
            
            print("\n=== STARTING CLASSIFICATION STAGE ===")
            
            if not reset and checkpoint_path.exists():
                print("Loading classification results from checkpoint")
                logger.info("Loading classification results from checkpoint")
                classification_data = load_checkpoint(checkpoint_path)
                self.classifier.load_classification_results(classification_data)
            else:
                print("Running classification with optimized batch processing")
                logger.info("Running classification with optimized batch processing")
                
                # Ensure classifier is trained
                if not self.classifier.is_trained():
                    print("Classifier not trained, running training first")
                    self.run_training()
                
                # Ensure query engine has collection set
                if self.query_engine.collection is None:
                    print("Setting collection in optimized query engine")
                    self.query_engine.set_collection(self.indexer.get_collection())
                
                # Get all personIds
                person_ids = self.preprocessor.get_all_person_ids()
                print(f"Classifying {len(person_ids)} entities")
                
                # Enable memory monitoring if configured
                monitor_enabled = self.config.get('batch_processing', {}).get('monitor_memory', False)
                if monitor_enabled:
                    print("Memory monitoring enabled during processing")
                    interval = self.config.get('batch_processing', {}).get('monitoring_interval', 30)
                    memory_context = monitor_memory_usage(interval)
                else:
                    # Create a no-op context manager
                    from contextlib import nullcontext
                    memory_context = nullcontext()
                
                # Process classification with memory monitoring
                with memory_context:
                    print("Starting batch classification")
                    results = self.batch_processor.classify_dataset(
                        person_ids,
                        self.preprocessor,
                        self.query_engine,
                        self.feature_extractor,
                        self.classifier,
                        self.imputer
                    )
                
                # Store match pairs
                self.classifier.match_pairs = results['match_pairs']
                print(f"Classification complete: found {len(results['match_pairs'])} matches")
                
                # Save checkpoint
                print("Saving classification results checkpoint")
                save_checkpoint(checkpoint_path, self.classifier.get_classification_results())
                
                # Clear caches to free memory
                self.feature_extractor.clear_caches()
                self.query_engine.clear_cache()
            
            print("=== CLASSIFICATION STAGE COMPLETED ===\n")

            # if self.imputer.enabled:
            #     imputation_stats = self.imputer.get_imputation_stats()
            #     print("\n=== IMPUTATION STATISTICS ===")
            #     print(f"Total imputation attempts: {imputation_stats['total_attempts']}")
            #     print(f"Successful imputations: {imputation_stats['successful_imputations']}")
            #     print(f"Cache hits: {imputation_stats['cache_hits']}")
            #     print(f"Overall success rate: {imputation_stats['overall_success_rate']:.2f}%")
                
            #     print("\nBy Field:")
            #     for field, metrics in imputation_stats['by_field'].items():
            #         if metrics['attempts'] > 0:
            #             print(f"  {field}: {metrics['success']}/{metrics['attempts']} ({metrics['success_rate']:.2f}%)")
        
        logger.info(f"Classification completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on classification results - FIXED VERSION
        # Pass the feature_extractor as an additional parameter
        self.analyzer.analyze_classification(self.classifier, self.feature_extractor)
    
    def run_clustering(self, reset=False):
        """Run the clustering stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "clustering.pkl"
            
            print("\n=== STARTING CLUSTERING STAGE ===")
            
            if not reset and checkpoint_path.exists():
                print("Loading clustering results from checkpoint")
                logger.info("Loading clustering results from checkpoint")
                clustering_data = load_checkpoint(checkpoint_path)
                self.clusterer.load_state(clustering_data)
            else:
                print("Running clustering")
                logger.info("Running clustering")
                
                # Ensure classification is complete
                if not self.classifier.has_classification_results():
                    print("Classification not complete, running classification first")
                    self.run_classification()
                
                # Process clustering
                match_pairs = self.classifier.get_match_pairs()
                print(f"Clustering {len(match_pairs)} match pairs")
                self.clusterer.cluster(match_pairs)
                
                # Save checkpoint
                print("Saving clustering results checkpoint")
                save_checkpoint(checkpoint_path, self.clusterer.get_state())
                
                # Save final clustering results
                output_path = self.output_dir / "entity_clusters.jsonl"
                self.clusterer.save_clusters(output_path)
                print(f"Saved {len(self.clusterer.get_clusters())} clusters to {output_path}")
            
            print("=== CLUSTERING STAGE COMPLETED ===\n")
        
        logger.info(f"Clustering completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on clustering results
        self.analyzer.analyze_clustering(self.clusterer)
    
    def run_reporting(self):
        """Run the reporting stage"""
        with Timer() as timer:
            logger.info("Generating reports")
            print("\n=== STARTING REPORTING STAGE ===")
            
            # Ensure all previous stages are complete
            if not self.clusterer.is_clustered():
                print("Clustering not complete, running clustering first")
                self.run_clustering()
            
            # Generate reports
            print("Generating analysis reports and visualizations")
            self.reporter.generate_reports(
                self.preprocessor,
                self.embedder,
                self.indexer,
                self.classifier,
                self.clusterer,
                self.output_dir
            )
            
            print("=== REPORTING STAGE COMPLETED ===\n")
        
        logger.info(f"Reporting completed in {timer.elapsed:.2f} seconds")
        
        # Print performance statistics
        batch_stats = self.batch_processor.get_statistics()
        cache_stats = self.query_engine.get_cache_stats() if hasattr(self.query_engine, 'get_cache_stats') else None
        
        print("\n=== PERFORMANCE STATISTICS ===")
        print(f"Batch Processing:")
        print(f"  - Processed batches: {batch_stats['processed_batches']}")
        print(f"  - Processed items: {batch_stats['processed_items']}")
        print(f"  - Items per second: {batch_stats['items_per_second']:.2f}")
        print(f"  - Total processing time: {batch_stats['total_processing_time']:.2f} seconds")
        
        if cache_stats:
            print("\nCache Statistics:")
            print(f"  - Hits: {sum(cache_stats['hits'].values())}")
            print(f"  - Misses: {sum(cache_stats['misses'].values())}")
            if sum(cache_stats['hits'].values()) + sum(cache_stats['misses'].values()) > 0:
                hit_rate = sum(cache_stats['hits'].values()) / (sum(cache_stats['hits'].values()) + sum(cache_stats['misses'].values()))
                print(f"  - Hit rate: {hit_rate:.2%}")
        
        print("=================================\n")
