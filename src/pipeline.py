"""
Pipeline orchestration module for entity resolution
"""

import os
import logging
import time
import numpy as np
from pathlib import Path

# Import pipeline modules
from .preprocessing import Preprocessor
from .embedding import Embedder
from .indexing import Indexer
from .imputation import Imputer
from .querying import QueryEngine
from .features import FeatureExtractor
from .classification import Classifier
from .clustering import Clusterer
from .reporting import Reporter
from .analysis import Analyzer
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
        
        # Initialize pipeline components
        self.preprocessor = Preprocessor(config)
        self.embedder = Embedder(config)
        self.indexer = Indexer(config)
        self.imputer = Imputer(config)
        self.query_engine = QueryEngine(config)
        self.feature_extractor = FeatureExtractor(config)
        self.classifier = Classifier(config)
        self.clusterer = Clusterer(config)
        self.reporter = Reporter(config)
        self.analyzer = Analyzer(config)
        
        logger.info(f"Pipeline initialized in {self.mode} mode")
    
    def run_all(self, reset=False):
        """Run the complete pipeline"""
        with Timer() as timer:
            self.run_preprocessing(reset=reset)
            self.run_embedding(reset=reset)
            self.run_indexing(reset=reset)
            self.run_training(reset=reset)
            self.run_classification(reset=reset)
            self.run_clustering(reset=reset)
            self.run_reporting()
        
        logger.info(f"Complete pipeline executed in {timer.elapsed:.2f} seconds")
    
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
        
        logger.info(f"Indexing completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on indexing results
        self.analyzer.analyze_indexing(self.indexer)
    
    def run_training(self, reset=False):
        """Run the training stage for classification"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "training.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading training results from checkpoint")
                training_data = load_checkpoint(checkpoint_path)
                self.classifier.load_state(training_data)
            else:
                logger.info("Running classifier training")
                # Ensure indexing is complete
                if not self.indexer.is_indexed():
                    self.run_indexing()
                
                # Set collection in query engine
                self.query_engine.set_collection(self.indexer.get_collection())
                
                # Load ground truth data
                ground_truth_file = self.config['dataset']['ground_truth_file']
                record_pairs, labels = self.preprocessor.load_ground_truth(ground_truth_file)
                
                # Extract features for training
                feature_vectors = []
                for left_id, right_id in record_pairs:
                    # Get record data
                    left_record = self.preprocessor.get_record(left_id)
                    right_record = self.preprocessor.get_record(right_id)
                    
                    # Impute missing values if necessary
                    if self.config['imputation']['enabled']:
                        # In pipeline.py, modify the imputation calls
                        left_record = self.imputer.impute_record(left_record, self.query_engine, self.preprocessor)
                        right_record = self.imputer.impute_record(right_record, self.query_engine, self.preprocessor)
                    
                    # Extract features
                    feature_vector = self.feature_extractor.extract_features(
                        left_record, right_record, self.query_engine
                    )
                    feature_vectors.append(feature_vector)
                
                # Convert to numpy array for RFE
                X = np.array(feature_vectors)
                y = np.array(labels)
                
                # Train RFE if enabled
                if self.config['features']['recursive_feature_elimination']['enabled']:
                    logger.info("Training Recursive Feature Elimination model")
                    self.feature_extractor.train_rfe(X, y)
                    
                    # Get updated feature vectors if RFE is enabled
                    if self.feature_extractor.selected_feature_indices is not None:
                        X = X[:, self.feature_extractor.selected_feature_indices]
                        logger.info(f"Using {X.shape[1]} features selected by RFE")
                
                # Get feature names (which will be filtered by RFE if enabled)
                feature_names = self.feature_extractor.get_feature_names()
                
                # Train classifier
                self.classifier.train(X, labels, feature_names)
                save_checkpoint(checkpoint_path, self.classifier.get_state())
        
        logger.info(f"Training completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on training results
        self.analyzer.analyze_training(self.classifier)
    
    def run_classification(self, reset=False):
        """Run the classification stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "classification.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading classification results from checkpoint")
                classification_data = load_checkpoint(checkpoint_path)
                self.classifier.load_classification_results(classification_data)
            else:
                logger.info("Running classification")
                # Ensure classifier is trained
                if not self.classifier.is_trained():
                    self.run_training()
                
                # Ensure query engine has collection set
                if self.query_engine.collection is None:
                    self.query_engine.set_collection(self.indexer.get_collection())
                
                # Get all personIds
                person_ids = self.preprocessor.get_all_person_ids()
                
                # Process classification
                self.classifier.classify_dataset(
                    person_ids,
                    self.preprocessor,
                    self.query_engine,
                    self.feature_extractor,
                    self.imputer
                )
                save_checkpoint(checkpoint_path, self.classifier.get_classification_results())
        
        logger.info(f"Classification completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on classification results
        self.analyzer.analyze_classification(self.classifier)
    
    def run_clustering(self, reset=False):
        """Run the clustering stage"""
        with Timer() as timer:
            checkpoint_path = self.checkpoint_dir / "clustering.pkl"
            
            if not reset and checkpoint_path.exists():
                logger.info("Loading clustering results from checkpoint")
                clustering_data = load_checkpoint(checkpoint_path)
                self.clusterer.load_state(clustering_data)
            else:
                logger.info("Running clustering")
                # Ensure classification is complete
                if not self.classifier.has_classification_results():
                    self.run_classification()
                
                # Process clustering
                match_pairs = self.classifier.get_match_pairs()
                self.clusterer.cluster(match_pairs)
                save_checkpoint(checkpoint_path, self.clusterer.get_state())
                
                # Save final clustering results
                output_path = self.output_dir / "entity_clusters.jsonl"
                self.clusterer.save_clusters(output_path)
        
        logger.info(f"Clustering completed in {timer.elapsed:.2f} seconds")
        
        # Run analysis on clustering results
        self.analyzer.analyze_clustering(self.clusterer)
    
    def run_reporting(self):
        """Run the reporting stage"""
        with Timer() as timer:
            logger.info("Generating reports")
            
            # Ensure all previous stages are complete
            if not self.clusterer.is_clustered():
                self.run_clustering()
            
            # Generate reports
            self.reporter.generate_reports(
                self.preprocessor,
                self.embedder,
                self.indexer,
                self.classifier,
                self.clusterer,
                self.output_dir
            )
        
        logger.info(f"Reporting completed in {timer.elapsed:.2f} seconds")