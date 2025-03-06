"""
Feature engineering module for entity resolution
Handles feature extraction for record pairs
"""

import re
import logging
import numpy as np
from scipy.spatial.distance import cosine
import Levenshtein
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from weaviate.classes.query import Filter, MetadataQuery

from .utils import Timer

# Configure logger
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Feature extractor for entity resolution"""
    
    def __init__(self, config):
        """Initialize the feature extractor with configuration"""
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
    
    def _compile_birth_death_pattern(self):
        """Compile regular expressions for birth-death year pattern matching"""
        patterns = []
        
        # Pattern 1: Birth year with approximate death year - "565 - approximately 665"
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 2: Approximate birth and death years
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 3: Approximate birth with standard death
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 4: Standard birth-death range
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 5: Death year only with approximate marker
        patterns.append(r'[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 6: Death year only (simple)
        patterns.append(r'[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 7: Birth year only with approximate marker
        patterns.append(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        
        # Pattern 8: Birth year only (simple)
        patterns.append(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
        
        # Pattern 9: Explicit birth/death prefixes
        patterns.append(r'(?:b\.|born)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)|(?:d\.|died)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
        
        # Pattern 10: Single approximate year (fallback)
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
        
        # Interaction features
        if self.interaction_features.get('person_title_harmonic', False):
            feature_names.append('person_title_harmonic')
        
        if self.interaction_features.get('person_provision_harmonic', False):
            feature_names.append('person_provision_harmonic')
        
        if self.interaction_features.get('person_subjects_harmonic', False):
            feature_names.append('person_subjects_harmonic')
        
        if self.interaction_features.get('title_subjects_harmonic', False):
            feature_names.append('title_subjects_harmonic')
        
        if self.interaction_features.get('title_provision_harmonic', False):
            feature_names.append('title_provision_harmonic')
        
        if self.interaction_features.get('provision_subjects_harmonic', False):
            feature_names.append('provision_subjects_harmonic')
        
        if self.interaction_features.get('person_subjects_product', False):
            feature_names.append('person_subjects_product')
        
        if self.interaction_features.get('composite_subjects_ratio', False):
            feature_names.append('composite_subjects_ratio')
        
        # Birth/death year exact match
        feature_names.append('birth_death_year_match')
        
        return feature_names
    
    def get_feature_names(self):
        """Get feature names, filtered by RFE if enabled"""
        if self.rfe_enabled and self.selected_features is not None:
            return self.selected_features
        return self.feature_names
    
    def extract_features(self, left_record, right_record, query_engine):
        """Extract features for a record pair"""
        features = {}
        
        # Get vectors for both records
        left_vectors = self._get_vectors(left_record, query_engine)
        right_vectors = self._get_vectors(right_record, query_engine)
        
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
            left_person_hash = left_record.get('person')
            right_person_hash = right_record.get('person')
            
            if left_person_hash != "NULL" and right_person_hash != "NULL":
                # Get original strings from query engine
                left_person_obj = query_engine.collection.query.fetch_objects(
                    filters=Filter.by_property("hash").equal(left_person_hash),
                    limit=1
                )
                
                right_person_obj = query_engine.collection.query.fetch_objects(
                    filters=Filter.by_property("hash").equal(right_person_hash),
                    limit=1
                )
                
                if left_person_obj.objects and right_person_obj.objects:
                    left_person = left_person_obj.objects[0].properties.get('text', '')
                    right_person = right_person_obj.objects[0].properties.get('text', '')
                    
                    levenshtein_sim = self._compute_levenshtein_similarity(left_person, right_person)
                    features['person_levenshtein'] = levenshtein_sim
                else:
                    features['person_levenshtein'] = 0.0
            else:
                features['person_levenshtein'] = 0.0
        
        # Compute interaction features
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
        
        # Check for birth/death year match
        left_person = left_person if 'left_person' in locals() else ''
        right_person = right_person if 'right_person' in locals() else ''
        features['birth_death_year_match'] = self._check_birth_death_year_match(left_person, right_person)
        
        # Convert features dictionary to numpy array
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        # Apply RFE if enabled and trained
        if self.rfe_enabled and self.selected_feature_indices is not None:
            feature_vector = feature_vector[self.selected_feature_indices]
        
        return feature_vector
    
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
    
    def _get_vectors(self, record, query_engine):
        """Get vectors for a record"""
        vectors = {}
        
        for field in self.fields_to_embed:
            if field in record and record[field] != "NULL":
                vector = query_engine.get_vector_by_hash(record[field], field_type=field)
                if vector is not None:
                    vectors[field] = vector
        
        return vectors
    
    def _compute_cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        try:
            # Cosine distance is 1 - cosine similarity
            distance = cosine(vec1, vec2)
            similarity = 1.0 - distance
            return similarity
        except Exception as e:
            logger.warning(f"Error computing cosine similarity: {e}")
            return 0.0
    
    def _compute_levenshtein_similarity(self, str1, str2):
        """Compute Levenshtein similarity between two strings"""
        if not str1 or not str2:
            return 0.0
        
        try:
            # Normalize by max length
            max_len = max(len(str1), len(str2))
            if max_len == 0:
                return 1.0
            
            distance = Levenshtein.distance(str1, str2)
            similarity = 1.0 - (distance / max_len)
            return similarity
        except Exception as e:
            logger.warning(f"Error computing Levenshtein similarity: {e}")
            return 0.0
    
    def _harmonic_mean(self, a, b):
        """Compute harmonic mean of two values"""
        if a <= 0 or b <= 0:
            return 0.0
        
        return 2 * a * b / (a + b)
    
    def _extract_birth_death_years(self, person_string):
        """Extract birth and death years from a person string."""
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
        """Check if birth/death years match in person strings."""
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