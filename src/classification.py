"""
Modified classification.py with progress bar for training stage
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing as mp
from joblib import Parallel, delayed
from weaviate.classes.query import Filter

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

        # Person cosine prefilter (new)
        self.person_cosine_prefilter = config['features'].get('person_cosine_prefilter', {}).get('enabled', False)
        self.person_cosine_threshold = config['features'].get('person_cosine_prefilter', {}).get('threshold', 0.70)
        
        # Composite cosine prefilter (new)
        self.composite_cosine_prefilter = config['features'].get('composite_cosine_prefilter', {}).get('enabled', False)
        self.composite_cosine_threshold = config['features'].get('composite_cosine_prefilter', {}).get('threshold', 0.65)
        self.composite_override_threshold = config['features'].get('composite_cosine_prefilter', {}).get('override_threshold', 0.90)
        
        # Initialize model weights
        self.weights = None
        self.bias = 0.0
        self.feature_names = None
        
        # Training metrics
        self.metrics = None
        
        # Classification results
        self.match_pairs = []
        
        # Processing state
        self.trained = False
    
    def train(self, feature_vectors, labels, feature_names=None, record_pairs=None):
        """Train the classifier with enhanced reporting capabilities
        
        Args:
            feature_vectors: Array of feature vectors
            labels: Array of binary labels
            feature_names: Optional list of feature names
            record_pairs: Optional list of record pair IDs corresponding to feature vectors
        """
        with Timer() as timer:
            print("\n==== Starting Classifier Training ====")
            logger.info("Starting classifier training")
            
            # Store feature names
            if feature_names is None:
                logger.warning("No feature names provided, using generic names")
                self.feature_names = [f"feature_{i}" for i in range(len(feature_vectors[0]))]
            else:
                self.feature_names = feature_names
                logger.info(f"Using {len(self.feature_names)} feature names: {self.feature_names}")
            
            # Convert to numpy arrays
            X = np.array(feature_vectors)
            y = np.array(labels)
            
            print(f"Training with {len(X)} samples ({sum(y)} positive, {len(y) - sum(y)} negative)")
            logger.info(f"Training with {len(X)} samples ({sum(y)} positive, {len(y) - sum(y)} negative)")
            
            # Normalize features
            X = self._normalize_features(X)
            
            # Split data into train, validation, and test sets
            # Keep track of indices for later reference
            indices = np.arange(len(X))
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X, y, indices, test_size=1-self.train_test_split, random_state=42
            )
            
            # Store record pair IDs for the test set if provided
            if record_pairs is not None:
                self.pair_ids = [record_pairs[i] for i in test_indices]
                logger.info(f"Stored {len(self.pair_ids)} pair IDs for test set")
            else:
                self.pair_ids = None
                
            if self.early_stopping:
                # Further split training data to create a validation set
                validation_ratio = self.validation_split / self.train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=validation_ratio, random_state=42
                )
                print(f"Data split: Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
                logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")
            else:
                print(f"Data split: Training set: {len(X_train)}, Test set: {len(X_test)}")
                logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Train using logistic regression with gradient descent
            self._train_logistic_regression(X_train, y_train, X_val, y_val if self.early_stopping else None)
            
            # Store test data for reporting
            self.X_test = X_test
            self.y_test = y_test
            
            # Get predictions and scores for test set
            y_pred_prob = self._sigmoid(X_test.dot(self.weights) + self.bias)
            y_pred = (y_pred_prob >= self.match_threshold).astype(int)
            
            # Store predictions and scores
            self.test_scores = y_pred_prob
            self.test_predictions = y_pred
            
            # Calculate feature correlation matrix for reporting
            self.feature_correlation = {}
            corr_matrix = np.corrcoef(X_test, rowvar=False)
            for i, name1 in enumerate(self.feature_names):
                self.feature_correlation[name1] = {}
                for j, name2 in enumerate(self.feature_names):
                    if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                        self.feature_correlation[name1][name2] = float(corr_matrix[i, j])
            
            # Evaluate on test set
            test_metrics = self._evaluate(X_test, y_test)
            print(f"Test set metrics: Precision: {test_metrics['precision']:.4f}, "
                f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            logger.info(f"Test set metrics: Precision: {test_metrics['precision']:.4f}, "
                    f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
            
            # Store all metrics
            self.metrics = {
                'train_metrics': self.metrics,
                'test_metrics': test_metrics,
                'feature_importance': self._get_feature_importance()
            }

            # Analyze feature distribution among true matches
            if sum(y_test) > 0:  # If we have any positive examples
                match_features = X_test[y_test == 1]
                feature_means = np.mean(match_features, axis=0)
                logger.info("True match feature means:")
                for i, name in enumerate(self.feature_names):
                    logger.info(f"  {name}: {feature_means[i]:.4f}")

            # Log misclassified examples count
            misclassified_count = np.sum(y_pred != y_test)
            print(f"Misclassified examples: {misclassified_count} out of {len(y_test)} ({misclassified_count/len(y_test):.2%})")
            logger.info(f"Misclassified examples: {misclassified_count} out of {len(y_test)} ({misclassified_count/len(y_test):.2%})")
            
            self.trained = True
            
            print("==== Classifier Training Complete ====\n")
        
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
        
        print(f"Starting logistic regression training ({self.max_iterations} max iterations)")
        print(f"Early stopping: {'Enabled' if self.early_stopping else 'Disabled'}")
        print(f"Regularization: {self.reg_type}, lambda={self.reg_lambda}")
        
        # Progress bar setup
        pbar = tqdm(total=self.max_iterations, desc="Training", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        # Training loop
        for iteration in range(self.max_iterations):
            # Forward pass
            y_pred_prob = self._sigmoid(X_train.dot(self.weights) + self.bias)
            
            # Compute loss
            train_loss = self._binary_cross_entropy(y_train, y_pred_prob)
            
            # Compute metrics for tracking
            y_pred = (y_pred_prob >= 0.5).astype(int)
            precision = precision_score(y_train, y_pred, zero_division=0)
            recall = recall_score(y_train, y_pred, zero_division=0)
            f1 = f1_score(y_train, y_pred, zero_division=0)
            
            train_metrics = {
                'iteration': iteration,
                'loss': train_loss,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Validation if enabled
            if self.early_stopping and X_val is not None and y_val is not None:
                val_pred_prob = self._sigmoid(X_val.dot(self.weights) + self.bias)
                val_loss = self._binary_cross_entropy(y_val, val_pred_prob)
                
                # Update progress bar description with metrics
                pbar.set_description(f"Training [Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}]")
                
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
                    pbar.set_description(f"Early stopping at iteration {iteration} [Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {f1:.4f}]")
                    logger.info(f"Early stopping at iteration {iteration}")
                    # Restore best weights
                    self.weights = best_weights
                    self.bias = best_bias
                    pbar.update(self.max_iterations - iteration - 1)  # Update progress bar to completion
                    break
            else:
                # Update progress bar description with metrics (no validation)
                pbar.set_description(f"Training [Loss: {train_loss:.4f}, F1: {f1:.4f}]")
            
            # Log progress periodically
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Loss: {train_loss:.4f}, "
                          f"Precision: {precision:.4f}, "
                          f"Recall: {recall:.4f}, "
                          f"F1: {f1:.4f}")
            
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
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Final log of best metrics
        final_metrics = metrics_history[-1] if metrics_history else None
        if final_metrics:
            print(f"Final training metrics: Loss: {final_metrics['loss']:.4f}, "
                 f"Precision: {final_metrics['precision']:.4f}, "
                 f"Recall: {final_metrics['recall']:.4f}, "
                 f"F1: {final_metrics['f1']:.4f}")
        
        # Store training metrics
        self.metrics = final_metrics
    
    # Rest of the Classifier class remains unchanged...
    
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
        # Check if X is empty
        if X.shape[0] == 0 or X.shape[1] == 0:
            logger.error(f"Cannot normalize features: Empty array with shape {X.shape}")
            # Return the original array if it's empty
            return X
        
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
        """Predict whether a record pair matches with dimension compatibility"""
        if not self.trained:
            logger.error("Classifier not trained")
            return 0.0
        
        # Handle dimension mismatch between feature vector and weights
        if len(feature_vector) != len(self.weights):
            logger.warning(f"Feature vector dimension ({len(feature_vector)}) doesn't match weights ({len(self.weights)})")
            
            # If we have feature names, we can match by name
            if hasattr(self, 'feature_names') and self.feature_names:
                logger.info("Attempting to align features by name")
                aligned_vector = np.zeros(len(self.weights))
                
                # Get feature extractor's feature names if available
                feature_extractor_names = None
                if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'get_feature_names'):
                    feature_extractor_names = self.feature_extractor.get_feature_names()
                
                # Try to align features
                for i, name in enumerate(self.feature_names):
                    # First check if the name exists directly in feature_extractor_names
                    if feature_extractor_names and name in feature_extractor_names:
                        idx = feature_extractor_names.index(name)
                        if idx < len(feature_vector):
                            aligned_vector[i] = feature_vector[idx]
                    # Also check for the feature_ prefix version
                    elif feature_extractor_names and f"feature_{name}" in feature_extractor_names:
                        idx = feature_extractor_names.index(f"feature_{name}")
                        if idx < len(feature_vector):
                            aligned_vector[i] = feature_vector[idx]
                
                # Use the aligned vector
                feature_vector = aligned_vector
            else:
                # If we can't align, just use the first N features where N is the length of weights
                logger.warning("No feature names available, using first N features")
                feature_vector = feature_vector[:len(self.weights)]
        
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
        """Process a single person and find matches with improved prefilter handling
        
        This method examines potential matches for a person_id, applying both prefilters
        and regular classification. Prefiltered pairs are still fully processed to ensure
        complete feature extraction and proper test metrics.
        
        Args:
            person_id: ID of the person to process
            all_person_ids: List of all person IDs to check against
            preprocessor: Preprocessor instance for record lookup
            query_engine: Query engine for vector retrieval
            feature_extractor: Feature extractor for feature calculation
            imputer: Optional imputer for null values
            
        Returns:
            List of match tuples (person_id, candidate_id, confidence)
        """
        matches = []
        
        try:
            # Get record and person hash
            record = preprocessor.get_record(person_id)
            if not record or 'person' not in record or record['person'] == "NULL":
                logger.debug(f"Skipping person {person_id}: Invalid record or missing person field")
                return matches
            
            person_hash = record['person']
            
            # Find candidate matches
            candidates = query_engine.find_candidates(
                person_hash,
                min_similarity=self.config['querying']['min_similarity'],
                limit=self.config['querying']['max_candidates']
            )
            
            # Get feature names once for this method execution
            feature_names = feature_extractor.get_feature_names()
            feature_indices = {name: idx for idx, name in enumerate(feature_names)}
            
            # Process candidates
            for candidate_hash, similarity in candidates:
                # Find person IDs with this candidate hash
                candidate_person_ids = [
                    pid for pid in all_person_ids
                    if pid != person_id and preprocessor.get_record(pid) and 
                    preprocessor.get_record(pid).get('person') == candidate_hash
                ]
                
                for candidate_id in candidate_person_ids:
                    try:
                        # Get candidate record
                        candidate_record = preprocessor.get_record(candidate_id)
                        
                        # Skip invalid records
                        if not candidate_record:
                            logger.debug(f"Skipping candidate {candidate_id}: Invalid record")
                            continue
                        
                        # Impute missing values if needed
                        if self.config['imputation']['enabled'] and imputer:
                            # Create copies to avoid modifying originals
                            record_copy = record.copy()
                            candidate_copy = candidate_record.copy()
                            
                            # Apply imputation and log changes
                            logger.debug(f"Before imputation - record: {record_copy.get('provision', 'NULL')}")
                            record_copy = imputer.impute_record(record_copy, query_engine)
                            logger.debug(f"After imputation - record: {record_copy.get('provision', 'NULL')}")
                            
                            logger.debug(f"Before imputation - candidate: {candidate_copy.get('provision', 'NULL')}")
                            candidate_copy = imputer.impute_record(candidate_copy, query_engine)
                            logger.debug(f"After imputation - candidate: {candidate_copy.get('provision', 'NULL')}")
                            
                            # Use imputed records for feature extraction
                            imputed_record = record_copy
                            imputed_candidate = candidate_copy
                        else:
                            imputed_record = record
                            imputed_candidate = candidate_record
                        
                        # Extract features - always do this regardless of prefilters
                        feature_vector = feature_extractor.extract_features(
                            imputed_record, imputed_candidate, query_engine
                        )
                        
                        # Initialize prefilter decision tracking
                        apply_prefilter = False
                        prefilter_type = None
                        prefilter_confidence = 0.0
                        prefilter_threshold_used = 0.0
                        
                        # Check for composite_cosine prefilter - high value means automatic match
                        if self.composite_cosine_prefilter:
                            composite_cosine_found = False
                            composite_cosine = 0.0
                            
                            try:
                                if 'composite_cosine' in feature_indices:
                                    composite_cosine_idx = feature_indices['composite_cosine']
                                    if composite_cosine_idx < len(feature_vector):
                                        composite_cosine = feature_vector[composite_cosine_idx]
                                        composite_cosine_found = True
                                        logger.debug(f"Found composite_cosine = {composite_cosine:.4f}")
                            except Exception as e:
                                logger.warning(f"Error getting composite_cosine: {e}")
                            
                            # Apply the prefilter - auto accept if above threshold
                            if composite_cosine_found and composite_cosine >= self.composite_cosine_threshold:
                                apply_prefilter = True
                                prefilter_type = "composite_cosine"
                                prefilter_confidence = self.composite_override_threshold
                                
                                logger.info(f"AutoMatch: {person_id} - {candidate_id} with composite_cosine={composite_cosine:.4f}")
                        
                        # Check for exact name prefilter if not already decided
                        if self.exact_name_prefilter and not apply_prefilter:
                            try:
                                # Get original strings
                                person_obj = query_engine.collection.query.fetch_objects(
                                    filters=Filter.by_property("hash").equal(person_hash),
                                    limit=1
                                )
                                
                                candidate_obj = query_engine.collection.query.fetch_objects(
                                    filters=Filter.by_property("hash").equal(candidate_hash),
                                    limit=1
                                )
                                
                                if person_obj.objects and candidate_obj.objects:
                                    person_string = person_obj.objects[0].properties.get('text', '')
                                    candidate_string = candidate_obj.objects[0].properties.get('text', '')
                                    
                                    # Use the feature extractor's birth/death year matching
                                    birth_death_match = feature_extractor._check_birth_death_year_match(
                                        person_string, candidate_string
                                    )
                                    
                                    # If birth/death years match exactly, it's a strong signal
                                    if birth_death_match > 0:
                                        apply_prefilter = True
                                        prefilter_type = "birth_death_year"
                                        prefilter_confidence = 0.95  # High confidence
                                        logger.info(f"AutoMatch: {person_id} - {candidate_id} with exact birth/death year match")
                            except Exception as e:
                                logger.warning(f"Error in birth/death prefilter: {e}")
                        
                        # Check for person_cosine prefilter if not already decided - low value means automatic reject
                        if self.person_cosine_prefilter and not apply_prefilter:
                            person_cosine_found = False
                            person_cosine = 0.0
                            
                            try:
                                if 'person_cosine' in feature_indices:
                                    person_cosine_idx = feature_indices['person_cosine']
                                    if person_cosine_idx < len(feature_vector):
                                        person_cosine = feature_vector[person_cosine_idx]
                                        person_cosine_found = True
                                        logger.debug(f"Found person_cosine = {person_cosine:.4f}")
                            except Exception as e:
                                logger.warning(f"Error getting person_cosine: {e}")
                            
                            # Apply the prefilter - reject if below threshold
                            if person_cosine_found and person_cosine < self.person_cosine_threshold:
                                apply_prefilter = True
                                prefilter_type = "person_cosine"
                                prefilter_confidence = 0.25  # Low confidence
                                prefilter_threshold_used = self.person_cosine_threshold
                                logger.debug(f"Rejecting pair due to low person_cosine: {person_cosine:.4f} < {self.person_cosine_threshold}")
                        
                        # Make classification decision
                        if apply_prefilter:
                            # Use prefilter decision
                            matches.append((person_id, candidate_id, prefilter_confidence))
                            logger.debug(f"Prefilter decision ({prefilter_type}): {person_id} - {candidate_id} with confidence {prefilter_confidence:.4f}")
                        else:
                            # Use regular classifier
                            probability = self.predict(feature_vector)
                            
                            # Check if it's a match based on probability
                            if probability >= self.match_threshold:
                                matches.append((person_id, candidate_id, float(probability)))
                                logger.debug(f"Regular classification: {person_id} - {candidate_id} with probability {probability:.4f}")
                            
                            # Log potential missed matches with high similarity but low probability
                            elif 'composite_cosine' in feature_indices and 'person_cosine' in feature_indices:
                                composite_idx = feature_indices['composite_cosine']
                                person_idx = feature_indices['person_cosine']
                                
                                if (composite_idx < len(feature_vector) and 
                                    person_idx < len(feature_vector) and
                                    feature_vector[composite_idx] >= 0.65 and 
                                    feature_vector[person_idx] >= 0.7):
                                    
                                    logger.warning(f"POTENTIAL MISSED MATCH: {person_id} - {candidate_id} with " +
                                                f"composite_cosine={feature_vector[composite_idx]:.4f}, " +
                                                f"person_cosine={feature_vector[person_idx]:.4f} but " +
                                                f"probability={probability:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error processing candidate pair {person_id} - {candidate_id}: {e}")
                        continue  # Skip this candidate but continue with others
            
            # Log summary for debugging
            if matches:
                logger.debug(f"Found {len(matches)} matches for person {person_id}")
            
            return matches  # Return all matches
            
        except Exception as e:
            logger.error(f"Error processing person {person_id}: {e}", exc_info=True)
            return matches
    
    def get_match_pairs(self):
        """Get all match pairs"""
        return self.match_pairs
    
    def _get_feature_importance(self):
        """Get feature importance based on model weights"""
        if not self.trained or self.weights is None:
            return {}
        
        # Use feature names from initialization or training
        if not self.feature_names:
            logger.warning("No feature names available, using generic names")
            feature_names = [f"feature_{i}" for i in range(len(self.weights))]
        else:
            feature_names = self.feature_names
        
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
        """Get the current state for checkpointing with enhanced test data reporting"""
        state = {
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': float(self.bias),
            'metrics': self.metrics,
            'feature_min': self.feature_min.tolist() if hasattr(self, 'feature_min') else None,
            'feature_max': self.feature_max.tolist() if hasattr(self, 'feature_max') else None,
            'feature_range': self.feature_range.tolist() if hasattr(self, 'feature_range') else None,
            'feature_names': self.feature_names,
            'trained': self.trained,
            'feature_importance': self._get_feature_importance() if self.trained else None,
            'feature_correlation': self.feature_correlation if hasattr(self, 'feature_correlation') else None
        }
        
        # Add test data components if they exist
        if hasattr(self, 'X_test'):
            state['test_data'] = self.X_test.tolist() if self.X_test is not None else None
        
        if hasattr(self, 'y_test'):
            state['test_labels'] = self.y_test.tolist() if self.y_test is not None else None
        
        if hasattr(self, 'test_predictions'):
            state['test_predictions'] = self.test_predictions.tolist() if self.test_predictions is not None else None
        
        if hasattr(self, 'test_scores'):
            state['test_scores'] = self.test_scores.tolist() if self.test_scores is not None else None
        
        if hasattr(self, 'pair_ids') and self.pair_ids is not None:
            state['pair_ids'] = self.pair_ids
        
        return state
    
    def load_state(self, state):
        """Load state from checkpoint with support for test data components"""
        if state['weights'] is not None:
            self.weights = np.array(state['weights'])
        
        self.bias = state['bias']
        self.metrics = state['metrics']
        self.feature_names = state['feature_names'] if 'feature_names' in state else None
        
        if state['feature_min'] is not None:
            self.feature_min = np.array(state['feature_min'])
        
        if state['feature_max'] is not None:
            self.feature_max = np.array(state['feature_max'])
        
        if state['feature_range'] is not None:
            self.feature_range = np.array(state['feature_range'])
        
        # Load feature correlation if available
        if 'feature_correlation' in state and state['feature_correlation'] is not None:
            self.feature_correlation = state['feature_correlation']
        
        # Load test data components if available
        if 'test_data' in state and state['test_data'] is not None:
            self.X_test = np.array(state['test_data'])
        
        if 'test_labels' in state and state['test_labels'] is not None:
            self.y_test = np.array(state['test_labels'])
        
        if 'test_predictions' in state and state['test_predictions'] is not None:
            self.test_predictions = np.array(state['test_predictions'])
        
        if 'test_scores' in state and state['test_scores'] is not None:
            self.test_scores = np.array(state['test_scores'])
        
        if 'pair_ids' in state:
            self.pair_ids = state['pair_ids']
        
        self.trained = state['trained']
        
        logger.info(f"Loaded classifier state: trained={self.trained}")
        
        # Log test data availability
        if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
            logger.info(f"Loaded test data: {len(self.X_test)} samples")
            if hasattr(self, 'test_predictions'):
                misclassified = np.sum(self.test_predictions != self.y_test)
                logger.info(f"Test set has {misclassified} misclassified examples out of {len(self.y_test)} ({misclassified/len(self.y_test):.2%})")
        else:
            logger.info("No test data available in loaded state")
    
    def get_classification_results(self):
        """Get classification results for checkpointing"""
        return {
            'match_pairs': self.match_pairs
        }
    
    def load_classification_results(self, results):
        """Load classification results from checkpoint"""
        self.match_pairs = results['match_pairs']
        logger.info(f"Loaded classification results: {len(self.match_pairs)} matches")