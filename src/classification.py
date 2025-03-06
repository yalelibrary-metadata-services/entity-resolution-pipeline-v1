"""
Classification module for entity resolution
Handles classifier training and prediction
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing as mp
from joblib import Parallel, delayed

from .utils import Timer

# Configure logger
logger = logging.getLogger(__name__)

class Classifier:
    """Classifier for entity resolution"""
    
    def __init__(self, config):
        """Initialize the classifier with configuration"""
        self.config = config
        self.match_threshold = config['classification']['match_threshold']
        self.max_iterations = config['classification']['max_iterations']
        self.learning_rate = config['classification']['learning_rate']
        self.batch_size = config['classification']['batch_size']
        self.num_workers = min(
            config['general']['num_workers'], 
            mp.cpu_count()
        )
        
        # Regularization settings
        self.reg_type = config['classification']['regularization']['type']
        self.reg_lambda = config['classification']['regularization']['lambda']
        
        # Early stopping settings
        self.early_stopping = config['classification']['early_stopping']['enabled']
        self.patience = config['classification']['early_stopping']['patience']
        self.min_delta = config['classification']['early_stopping']['min_delta']
        
        # Train-test split
        self.train_test_split = config['dataset']['train_test_split']
        self.validation_split = config['dataset']['validation_split']
        
        # Exact name prefilter
        self.exact_name_prefilter = config['features']['exact_name_prefilter']['enabled']
        self.override_threshold = config['features']['exact_name_prefilter']['override_threshold']
        
        # Initialize model weights
        self.weights = None
        self.bias = 0.0
        
        # Training metrics
        self.metrics = None
        
        # Classification results
        self.match_pairs = []
        
        # Processing state
        self.trained = False
    
    def train(self, feature_vectors, labels):
        """Train the classifier"""
        with Timer() as timer:
            logger.info("Starting classifier training")
            
            # Convert to numpy arrays
            X = np.array(feature_vectors)
            y = np.array(labels)
            
            logger.info(f"Training with {len(X)} samples ({sum(y)} positive, {len(y) - sum(y)} negative)")
            
            # Normalize features
            X = self._normalize_features(X)
            
            # Split data into train, validation, and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-self.train_test_split, random_state=42
            )
            
            if self.early_stopping:
                # Further split training data to create a validation set
                validation_ratio = self.validation_split / self.train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=validation_ratio, random_state=42
                )
                logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
            else:
                logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Train using logistic regression with gradient descent
            self._train_logistic_regression(X_train, y_train, X_val, y_val if self.early_stopping else None)
            
            # Evaluate on test set
            test_metrics = self._evaluate(X_test, y_test)
            logger.info(f"Test set metrics: Precision: {test_metrics['precision']:.4f}, "
                      f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            
            # Store metrics
            self.metrics = {
                'train_metrics': self.metrics,
                'test_metrics': test_metrics,
                'feature_importance': self._get_feature_importance()
            }
            
            self.trained = True
        
        logger.info(f"Training completed in {timer.elapsed:.2f} seconds")
        return self
    
    def _train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Train logistic regression model with gradient descent"""
        # Initialize weights randomly
        n_features = X_train.shape[1]
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # Training settings
        best_val_loss = float('inf')
        patience_counter = 0
        metrics_history = []
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred_prob = self._sigmoid(X_train.dot(self.weights) + self.bias)
            
            # Compute loss
            train_loss = self._binary_cross_entropy(y_train, y_pred_prob)
            
            # Compute metrics for tracking
            y_pred = (y_pred_prob >= 0.5).astype(int)
            train_metrics = {
                'iteration': iteration,
                'loss': train_loss,
                'precision': precision_score(y_train, y_pred, zero_division=0),
                'recall': recall_score(y_train, y_pred, zero_division=0),
                'f1': f1_score(y_train, y_pred, zero_division=0)
            }
            
            # Validation if enabled
            if self.early_stopping and X_val is not None and y_val is not None:
                val_pred_prob = self._sigmoid(X_val.dot(self.weights) + self.bias)
                val_loss = self._binary_cross_entropy(y_val, val_pred_prob)
                
                # Check for early stopping
                if val_loss < best_val_loss - self.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = self.weights.copy()
                    best_bias = self.bias
                else:
                    patience_counter += 1
                
                train_metrics['val_loss'] = val_loss
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at iteration {iteration}")
                    # Restore best weights
                    self.weights = best_weights
                    self.bias = best_bias
                    break
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Loss: {train_loss:.4f}, "
                          f"Precision: {train_metrics['precision']:.4f}, "
                          f"Recall: {train_metrics['recall']:.4f}, "
                          f"F1: {train_metrics['f1']:.4f}")
            
            metrics_history.append(train_metrics)
            
            # Backward pass (gradient descent)
            dw = X_train.T.dot(y_pred_prob - y_train) / len(y_train)
            db = np.mean(y_pred_prob - y_train)
            
            # Add regularization
            if self.reg_type == 'l2':
                dw += self.reg_lambda * self.weights
            elif self.reg_type == 'l1':
                dw += self.reg_lambda * np.sign(self.weights)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        # Store training metrics
        self.metrics = metrics_history[-1] if metrics_history else None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -709, 709)))  # Clip to avoid overflow
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """Binary cross entropy loss"""
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization
        if self.reg_type == 'l2':
            loss += 0.5 * self.reg_lambda * np.sum(self.weights**2) / len(y_true)
        elif self.reg_type == 'l1':
            loss += self.reg_lambda * np.sum(np.abs(self.weights)) / len(y_true)
        
        return loss
    
    def _evaluate(self, X, y):
        """Evaluate the model"""
        # Get predictions
        y_pred_prob = self._sigmoid(X.dot(self.weights) + self.bias)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # Compute metrics
        metrics = {
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        
        return metrics
    
    def _normalize_features(self, X):
        """Normalize features"""
        # Compute min and max for each feature
        self.feature_min = np.min(X, axis=0)
        self.feature_max = np.max(X, axis=0)
        
        # Avoid division by zero
        self.feature_range = self.feature_max - self.feature_min
        self.feature_range[self.feature_range == 0] = 1.0
        
        # Min-max normalization
        X_norm = (X - self.feature_min) / self.feature_range
        
        return X_norm
    
    def _normalize_feature_vector(self, X):
        """Normalize a single feature vector"""
        if not hasattr(self, 'feature_min') or not hasattr(self, 'feature_range'):
            logger.warning("Normalization parameters not set")
            return X
        
        # Min-max normalization
        X_norm = (X - self.feature_min) / self.feature_range
        
        return X_norm
    
    def predict(self, feature_vector):
        """Predict whether a record pair matches"""
        if not self.trained:
            logger.error("Classifier not trained")
            return 0.0
        
        # Normalize feature vector
        X = self._normalize_feature_vector(feature_vector)
        
        # Compute probability
        probability = self._sigmoid(X.dot(self.weights) + self.bias)
        
        return probability
    
    def classify_dataset(self, person_ids, preprocessor, query_engine, feature_extractor, imputer):
        """Classify all person pairs in the dataset"""
        if not self.trained:
            logger.error("Classifier not trained")
            return
        
        with Timer() as timer:
            logger.info(f"Starting classification of {len(person_ids)} entities")
            
            # Set Weaviate collection in query engine if not already set
            if query_engine.collection is None:
                # Assuming indexer.get_collection() is available
                from .indexing import Indexer
                indexer = Indexer(self.config)
                query_engine.set_collection(indexer.get_collection())
            
            # Process in parallel
            match_pairs = Parallel(n_jobs=self.num_workers)(
                delayed(self._process_person)(
                    person_id, person_ids, preprocessor, query_engine, feature_extractor, imputer
                ) for person_id in tqdm(person_ids, desc="Classifying entities")
            )
            
            # Flatten results and filter out None
            self.match_pairs = [pair for sublist in match_pairs if sublist for pair in sublist]
            
            logger.info(f"Classification completed: {len(self.match_pairs)} matches found")
        
        logger.info(f"Classification time: {timer.elapsed:.2f} seconds")
        return self
    
    def _process_person(self, person_id, all_person_ids, preprocessor, query_engine, feature_extractor, imputer):
        """Process a single person and find matches"""
        matches = []
        
        try:
            # Get record and person hash
            record = preprocessor.get_record(person_id)
            if not record or 'person' not in record or record['person'] == "NULL":
                return matches
            
            person_hash = record['person']
            
            # Find candidate matches
            candidates = query_engine.find_candidates(
                person_hash,
                min_similarity=self.config['querying']['min_similarity'],
                limit=self.config['querying']['max_candidates']
            )
            
            # Process candidates
            for candidate_hash, similarity in candidates:
                # Find person IDs with this candidate hash
                candidate_person_ids = [
                    pid for pid in all_person_ids
                    if pid != person_id and preprocessor.get_record(pid) and 
                    preprocessor.get_record(pid).get('person') == candidate_hash
                ]
                
                for candidate_id in candidate_person_ids:
                    # Get candidate record
                    candidate_record = preprocessor.get_record(candidate_id)
                    
                    # Apply exact name prefilter if enabled
                    if self.exact_name_prefilter:
                        # TODO: Implement exact name matching with birth/death years logic
                        pass
                    
                    # Impute missing values if needed
                    if self.config['imputation']['enabled']:
                        record = imputer.impute_record(record, query_engine)
                        candidate_record = imputer.impute_record(candidate_record, query_engine)
                    
                    # Extract features
                    feature_vector = feature_extractor.extract_features(
                        record, candidate_record, query_engine
                    )
                    
                    # Predict match probability
                    probability = self.predict(feature_vector)
                    
                    # Check if it's a match
                    if probability >= self.match_threshold:
                        matches.append((person_id, candidate_id, float(probability)))
            
            return matches
        
        except Exception as e:
            logger.error(f"Error processing person {person_id}: {e}")
            return matches
    
    def get_match_pairs(self):
        """Get all match pairs"""
        return self.match_pairs
    
    def _get_feature_importance(self):
        """Get feature importance based on model weights"""
        if not self.trained or self.weights is None:
            return {}
        
        # Get feature names from the feature extractor
        # This is a placeholder - in practice, you'd need to
        # get feature names from the feature extractor
        feature_names = [f"feature_{i}" for i in range(len(self.weights))]
        
        # Compute absolute weight values
        importance = np.abs(self.weights)
        
        # Normalize to sum to 1
        importance = importance / np.sum(importance)
        
        # Create a dictionary of feature importance
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
        
        return importance_dict
    
    def is_trained(self):
        """Check if the classifier is trained"""
        return self.trained
    
    def has_classification_results(self):
        """Check if classification results are available"""
        return len(self.match_pairs) > 0
    
    def get_state(self):
        """Get the current state for checkpointing"""
        return {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': float(self.bias),
            'metrics': self.metrics,
            'feature_min': self.feature_min.tolist() if hasattr(self, 'feature_min') else None,
            'feature_max': self.feature_max.tolist() if hasattr(self, 'feature_max') else None,
            'feature_range': self.feature_range.tolist() if hasattr(self, 'feature_range') else None,
            'trained': self.trained
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        if state['weights'] is not None:
            self.weights = np.array(state['weights'])
        
        self.bias = state['bias']
        self.metrics = state['metrics']
        
        if state['feature_min'] is not None:
            self.feature_min = np.array(state['feature_min'])
        
        if state['feature_max'] is not None:
            self.feature_max = np.array(state['feature_max'])
        
        if state['feature_range'] is not None:
            self.feature_range = np.array(state['feature_range'])
        
        self.trained = state['trained']
        
        logger.info(f"Loaded classifier state: trained={self.trained}")
    
    def get_classification_results(self):
        """Get classification results for checkpointing"""
        return {
            'match_pairs': self.match_pairs
        }
    
    def load_classification_results(self, results):
        """Load classification results from checkpoint"""
        self.match_pairs = results['match_pairs']
        logger.info(f"Loaded classification results: {len(self.match_pairs)} matches")
