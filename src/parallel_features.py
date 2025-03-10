"""
Parallel Feature Extraction module for Entity Resolution
Handles optimized parallel feature extraction with batching
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
import re
import threading
from scipy.spatial.distance import cosine
import Levenshtein
from weaviate.classes.query import Filter, MetadataQuery  # Add this import
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from .utils import Timer, compute_vector_similarity, compute_levenshtein_similarity

# Configure logger
logger = logging.getLogger(__name__)

class ParallelFeatureExtractor:
    """Parallel feature extractor for entity resolution"""
    
    def __init__(self, config):
        """Initialize the parallel feature extractor with configuration"""
        self.config = config
        
        # Feature flags
        self.use_cosine_similarity = config['features']['cosine_similarity']
        self.use_levenshtein_similarity = config['features']['levenshtein_similarity']
        
        # Field weights
        self.field_weights = config['features']['field_weights']
        
        # Interaction features
        self.interaction_features = config['features']['interaction_features']
        
        # Fields configuration
        self.fields_to_embed = config['fields']['embed']
        
        # Birth/death year pattern
        self.birth_death_pattern = self._compile_birth_death_pattern()
        
        # Feature names for reference
        self.feature_names = self._get_feature_names()
        
        # RFE configuration
        self.rfe_enabled = config['features']['recursive_feature_elimination']['enabled']
        self.rfe_step = config['features']['recursive_feature_elimination']['step']
        self.rfe_cv = config['features']['recursive_feature_elimination']['cv']
        
        # RFE model and selected features
        self.rfe_model = None
        self.selected_features = None
        self.selected_feature_indices = None
        
        # Thread local storage for local caches
        self.thread_local = threading.local()
        
        # Number of workers for parallel processing
        self.num_workers = min(
            config.get('general', {}).get('num_workers', 8),
            mp.cpu_count()
        )
        
        logger.info(f"Parallel feature extractor initialized with {self.num_workers} workers")
    
    def _compile_birth_death_pattern(self):
        """Compile regular expressions for birth-death year pattern matching"""
        patterns = []
        
        # Various patterns for birth-death years
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        patterns.append(r'(?:b\.|born)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)|(?:d\.|died)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        return [re.compile(pattern) for pattern in patterns]
    
    def _get_feature_names(self):
        """Get feature names"""
        feature_names = []
        
        # Basic similarity features
        if self.use_cosine_similarity:
            for field in self.fields_to_embed:
                feature_names.append(f"{field}_cosine")
        
        if self.use_levenshtein_similarity:
            feature_names.append("person_levenshtein")
        
        # Interaction features - add all enabled features from config
        for feature_name, enabled in self.interaction_features.items():
            if enabled:
                feature_names.append(feature_name)
        
        # Birth/death year exact match
        #feature_names.append('birth_death_year_match')
        
        return feature_names
    
    def get_feature_names(self):
        """Get feature names, filtered by RFE if enabled"""
        if self.rfe_enabled and self.selected_features is not None:
            return self.selected_features
        return self.feature_names
    
    def extract_features(self, left_record, right_record, query_engine, imputer=None):
        """Extract features for a single record pair (compatibility method)
        
        This method ensures compatibility with the original feature extractor interface
        by providing a wrapper around extract_features_with_cache.
        
        Args:
            left_record: Left record dictionary
            right_record: Right record dictionary
            query_engine: Query engine instance for vector retrieval
            imputer: Optional imputer for null values
            
        Returns:
            Feature vector (numpy array)
        """
        # Initialize thread-local cache if not already done
        if not hasattr(self.thread_local, 'vector_cache'):
            self.thread_local.vector_cache = {}
        
        if not hasattr(self.thread_local, 'string_cache'):
            self.thread_local.string_cache = {}
        
        # Call the implementation method with imputer
        return self.extract_features_with_cache(left_record, right_record, query_engine, imputer)
    
    def extract_features_batch(self, record_pairs, query_engine):
        """Extract features for a batch of record pairs
        
        Args:
            record_pairs: List of (left_record, right_record) tuples
            query_engine: Query engine instance for vector retrieval
            
        Returns:
            List of feature vectors
        """
        # Initialize thread-local cache if not already done
        if not hasattr(self.thread_local, 'vector_cache'):
            self.thread_local.vector_cache = {}
        
        if not hasattr(self.thread_local, 'string_cache'):
            self.thread_local.string_cache = {}
        
        # Pre-fetch the needed vectors and strings
        self._prefetch_data_for_batch(record_pairs, query_engine)
        
        # Process in parallel
        with Timer() as timer:
            feature_vectors = Parallel(n_jobs=self.num_workers)(
                delayed(self.extract_features_with_cache)(
                    left_record, right_record, query_engine
                ) for left_record, right_record in record_pairs
            )
        
        logger.debug(f"Batch feature extraction completed in {timer.elapsed:.2f} seconds")
        return feature_vectors
    
    def _prefetch_data_for_batch(self, record_pairs, query_engine):
        """Pre-fetch needed data for a batch of record pairs"""
        # Collect all hash-field pairs needed
        hash_field_pairs = []
        person_hashes = set()
        
        for left_record, right_record in record_pairs:
            for field in self.fields_to_embed:
                # Check left record
                if left_record and field in left_record and left_record[field] != "NULL":
                    hash_field_pairs.append((left_record[field], field))
                    if field == "person":
                        person_hashes.add(left_record[field])
                
                # Check right record
                if right_record and field in right_record and right_record[field] != "NULL":
                    hash_field_pairs.append((right_record[field], field))
                    if field == "person":
                        person_hashes.add(right_record[field])
        
        # Deduplicate hash-field pairs
        hash_field_pairs = list(set(hash_field_pairs))
        person_hashes = list(person_hashes)
        
        # Fetch vectors
        vectors = query_engine.batch_get_vectors(hash_field_pairs)
        
        # Store in thread-local cache
        for (hash_val, field_type), vector in vectors.items():
            self.thread_local.vector_cache[(hash_val, field_type)] = vector
        
        # Fetch person strings if needed
        if self.use_levenshtein_similarity and person_hashes:
            strings = query_engine.batch_get_strings(person_hashes)
            
            # Store in thread-local cache
            for hash_val, text in strings.items():
                self.thread_local.string_cache[hash_val] = text
    
    def extract_features_with_cache(self, left_record, right_record, query_engine, imputer=None):
        """Extract features using thread-local cache with imputation support
        
        Args:
            left_record: Left record dictionary
            right_record: Right record dictionary
            query_engine: Query engine instance (for fallback)
            imputer: Optional imputer for null values
            
        Returns:
            Feature vector (numpy array)
        """
        # Initialize thread-local cache if not already done
        if not hasattr(self.thread_local, 'vector_cache'):
            self.thread_local.vector_cache = {}
        
        if not hasattr(self.thread_local, 'string_cache'):
            self.thread_local.string_cache = {}
        
        # Impute missing values if imputer is provided
        if imputer is not None and imputer.enabled:
            if left_record is not None:
                left_record = imputer.impute_record(left_record, query_engine)
            if right_record is not None:
                right_record = imputer.impute_record(right_record, query_engine)
        
        features = {}
        
        # Get vectors with cache priority
        left_vectors = self._get_vectors_with_cache(left_record, query_engine)
        right_vectors = self._get_vectors_with_cache(right_record, query_engine)
        
        # Compute cosine similarities
        if self.use_cosine_similarity:
            for field in self.fields_to_embed:
                if field in left_vectors and field in right_vectors:
                    cosine_sim = self._compute_cosine_similarity(left_vectors[field], right_vectors[field])
                    features[f"{field}_cosine"] = cosine_sim
                else:
                    features[f"{field}_cosine"] = 0.0
        
        # Compute Levenshtein similarity for person names
        if self.use_levenshtein_similarity:
            left_person_hash = left_record.get('person') if left_record else None
            right_person_hash = right_record.get('person') if right_record else None
            
            if left_person_hash and right_person_hash and left_person_hash != "NULL" and right_person_hash != "NULL":
                # Get strings from cache or query
                left_person = self._get_string_with_cache(left_person_hash, query_engine)
                right_person = self._get_string_with_cache(right_person_hash, query_engine)
                
                if left_person and right_person:
                    levenshtein_sim = compute_levenshtein_similarity(left_person, right_person)
                    features['person_levenshtein'] = levenshtein_sim
                else:
                    features['person_levenshtein'] = 0.0
            else:
                features['person_levenshtein'] = 0.0
        
        # Compute interaction features if enabled
        self._compute_interaction_features(features)
        
        # Check for birth/death year match
        left_person = self._get_string_with_cache(left_record.get('person'), query_engine) if left_record else None
        right_person = self._get_string_with_cache(right_record.get('person'), query_engine) if right_record else None
        features['birth_death_year_match'] = self._check_birth_death_year_match(left_person, right_person)
        
        # Convert features dictionary to numpy array
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Apply RFE if enabled and trained
        if self.rfe_enabled and self.selected_feature_indices is not None:
            feature_vector = feature_vector[self.selected_feature_indices]
        
        return feature_vector
    
    def _get_vectors_with_cache(self, record, query_engine):
        """Get vectors for a record using thread-local cache"""
        vectors = {}
        
        if record is None:
            return vectors
        
        for field in self.fields_to_embed:
            if field in record and record[field] != "NULL":
                # Try thread-local cache first
                cache_key = (record[field], field)
                if cache_key in self.thread_local.vector_cache:
                    vectors[field] = self.thread_local.vector_cache[cache_key]
                else:
                    # Fall back to query engine
                    vector = query_engine.get_vector_by_hash(record[field], field_type=field)
                    if vector is not None:
                        vectors[field] = vector
                        # Add to thread-local cache
                        self.thread_local.vector_cache[cache_key] = vector
        
        return vectors
    
    def _get_string_with_cache(self, hash_value, query_engine):
        """Get string for a hash using thread-local cache"""
        if not hash_value or hash_value == "NULL":
            return None
        
        # Try thread-local cache first
        if hash_value in self.thread_local.string_cache:
            return self.thread_local.string_cache[hash_value]
        
        # Fall back to query engine's batch_get_strings method
        try:
            strings = query_engine.batch_get_strings([hash_value])
            if hash_value in strings:
                text = strings[hash_value]
                # Add to thread-local cache
                self.thread_local.string_cache[hash_value] = text
                return text
            return None
        except Exception as e:
            logger.warning(f"Error getting string for hash {hash_value}: {e}")
            return None
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        try:
            # Use compute_vector_similarity from utils
            return compute_vector_similarity(vec1, vec2, metric='cosine')
        except Exception as e:
            logger.warning(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def _compute_interaction_features(self, features):
        """Compute all interaction features based on configuration"""
        # Harmonic means
        if self.interaction_features.get('person_title_harmonic', False):
            features['person_title_harmonic'] = self._harmonic_mean(
                features.get('person_cosine', 0.0),
                features.get('title_cosine', 0.0)
            )
        
        if self.interaction_features.get('person_provision_harmonic', False):
            features['person_provision_harmonic'] = self._harmonic_mean(
                features.get('person_cosine', 0.0),
                features.get('provision_cosine', 0.0)
            )
        
        if self.interaction_features.get('person_subjects_harmonic', False):
            features['person_subjects_harmonic'] = self._harmonic_mean(
                features.get('person_cosine', 0.0),
                features.get('subjects_cosine', 0.0)
            )
        
        if self.interaction_features.get('title_subjects_harmonic', False):
            features['title_subjects_harmonic'] = self._harmonic_mean(
                features.get('title_cosine', 0.0),
                features.get('subjects_cosine', 0.0)
            )
        
        if self.interaction_features.get('title_provision_harmonic', False):
            features['title_provision_harmonic'] = self._harmonic_mean(
                features.get('title_cosine', 0.0),
                features.get('provision_cosine', 0.0)
            )
        
        if self.interaction_features.get('provision_subjects_harmonic', False):
            features['provision_subjects_harmonic'] = self._harmonic_mean(
                features.get('provision_cosine', 0.0),
                features.get('subjects_cosine', 0.0)
            )
        
        # Other interaction features
        if self.interaction_features.get('person_subjects_product', False):
            features['person_subjects_product'] = (
                features.get('person_cosine', 0.0) * features.get('subjects_cosine', 0.0)
            )
        
        if self.interaction_features.get('composite_subjects_ratio', False):
            composite_sim = features.get('composite_cosine', 0.0)
            subjects_sim = features.get('subjects_cosine', 0.0)
            
            if subjects_sim > 0:
                features['composite_subjects_ratio'] = composite_sim / subjects_sim
            else:
                features['composite_subjects_ratio'] = 0.0
    
    def _harmonic_mean(self, a, b):
        """Compute harmonic mean with better handling of very small values"""
        # Set a minimum threshold to avoid division by near-zero
        min_threshold = 0.001
        
        if a <= min_threshold or b <= min_threshold:
            # For very small values, use a weighted average instead
            if a <= min_threshold and b <= min_threshold:
                return 0.0  # Both values are essentially zero
            elif a <= min_threshold:
                return b * 0.1  # Return a fraction of the non-zero value
            else:
                return a * 0.1  # Return a fraction of the non-zero value
        
        # Normal harmonic mean for non-zero values
        return 2 * a * b / (a + b)
    
    def _extract_birth_death_years(self, person_string):
        """Extract birth and death years from a person string"""
        if not person_string:
            return None, None
            
        # Try each pattern in order
        for pattern in self.birth_death_pattern:
            match = pattern.search(person_string)
            if match:
                # Extract matching groups depending on pattern
                if len(match.groups()) == 1:
                    # Patterns with only one capture group (death year only)
                    return None, match.group(1)
                elif len(match.groups()) == 2:
                    if match.group(1) and match.group(2):
                        # Complete birth-death range
                        return match.group(1), match.group(2)
                    elif match.group(1):
                        # Birth year only or pattern 9 birth prefix
                        return match.group(1), None
                    elif match.group(2):
                        # Pattern 9 death prefix
                        return None, match.group(2)
        
        return None, None
    
    def _check_birth_death_year_match(self, left_person, right_person):
        """Check if birth/death years match in person strings"""
        if not left_person or not right_person:
            return 0.0
        
        # Extract years from both names
        left_birth, left_death = self._extract_birth_death_years(left_person)
        right_birth, right_death = self._extract_birth_death_years(right_person)
        
        # Check if we have enough data to compare
        if (left_birth or left_death) and (right_birth or right_death):
            # Compare birth years if both available
            if left_birth and right_birth:
                # Clean up years to handle "or" cases and question marks
                left_birth_clean = left_birth.split(' or ')[0].rstrip('?')
                right_birth_clean = right_birth.split(' or ')[0].rstrip('?')
                
                if left_birth_clean == right_birth_clean:
                    return 1.0
            
            # Compare death years if both available
            if left_death and right_death:
                # Clean up years to handle "or" cases and question marks
                left_death_clean = left_death.split(' or ')[0].rstrip('?')
                right_death_clean = right_death.split(' or ')[0].rstrip('?')
                
                if left_death_clean == right_death_clean:
                    return 1.0
        
        return 0.0
    
    def normalize_features(self, feature_vectors):
        """Normalize feature vectors"""
        if not feature_vectors:
            return []
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Compute min and max for each feature
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        
        # Min-max normalization
        X_norm = (X - min_vals) / range_vals
        
        return X_norm
    
    def clear_caches(self):
        """Clear thread-local caches"""
        # This only clears the cache for the current thread
        if hasattr(self.thread_local, 'vector_cache'):
            self.thread_local.vector_cache = {}
        
        if hasattr(self.thread_local, 'string_cache'):
            self.thread_local.string_cache = {}

    def train_rfe(self, X, y):
        """Train Recursive Feature Elimination model"""
        if not self.rfe_enabled:
            logger.info("RFE is disabled in configuration")
            return
        
        logger.info("Training Recursive Feature Elimination model")
        
        # Create base estimator (LogisticRegression)
        estimator = LogisticRegression(
            solver='liblinear', 
            max_iter=1000, 
            random_state=42
        )
        
        # Create RFE model
        n_features = X.shape[1]
        n_features_to_select = max(1, int(n_features / 2))  # Select half by default
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=self.rfe_step,
            verbose=1
        )
        
        # Fit RFE model
        rfe.fit(X, y)
        
        # Get selected features
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"RFE selected {len(selected_features)} features: {selected_features}")
        
        # Store results
        self.rfe_model = rfe
        self.selected_features = selected_features
        self.selected_feature_indices = selected_indices
        
        # Log feature rankings
        feature_ranking = [(self.feature_names[i], rfe.ranking_[i]) for i in range(len(self.feature_names))]
        feature_ranking.sort(key=lambda x: x[1])
        logger.info("Feature rankings (lower is better):")
        for feature, rank in feature_ranking:
            logger.info(f"  - {feature}: {rank}")
        
        return self
    
    def get_rfe_results(self):
        """Get RFE results for analysis"""
        if not self.rfe_enabled or self.rfe_model is None:
            return None
        
        results = {
            'selected_features': self.selected_features,
            'feature_rankings': []
        }
        
        # Add rankings for all features
        if hasattr(self.rfe_model, 'ranking_'):
            for i, feature in enumerate(self.feature_names):
                results['feature_rankings'].append({
                    'feature': feature,
                    'rank': int(self.rfe_model.ranking_[i]),
                    'selected': i in self.selected_feature_indices
                })
            
            # Sort by rank
            results['feature_rankings'].sort(key=lambda x: x['rank'])
        
        return results
