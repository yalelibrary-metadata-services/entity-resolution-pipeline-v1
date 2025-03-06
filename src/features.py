"""
Feature engineering module for entity resolution
Handles feature extraction for record pairs
"""

import re
import logging
import numpy as np
from scipy.spatial.distance import cosine
import Levenshtein
from weaviate.classes.query import Filter

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
        # self.birth_death_pattern = re.compile(r',\s*(\d{4})-(\d{4}|\?)')
        
        # Feature names for reference
        self.feature_names = self._get_feature_names()
    
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
        """Get feature names"""
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
        features['birth_death_year_match'] = self._check_birth_death_year_match(
            left_person if 'left_person' in locals() else '',
            right_person if 'right_person' in locals() else ''
        )
        
        # Convert features dictionary to numpy array
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        
        return feature_vector
    
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

    def _check_birth_death_year_match(self, left_person, right_person):
        """Check if birth/death years match in person strings."""
        if not left_person or not right_person:
            return 0.0
        
        # Extract years from both names
        left_birth, left_death = extract_birth_death_years(left_person)
        right_birth, right_death = extract_birth_death_years(right_person)
        
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

def extract_birth_death_years(person_string):
    """Extract birth and death years from a person string."""
    
    # Pattern 1: Birth year with approximate death year
    # Must check this specific pattern first to handle "565 - approximately 665" case
    birth_approx_death = re.compile(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = birth_approx_death.search(person_string)
    if match:
        return match.group(1), match.group(2)
    
    # Pattern 2: Approximate birth and death years
    approx_birth_approx_death = re.compile(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = approx_birth_approx_death.search(person_string)
    if match:
        return match.group(1), match.group(2)
    
    # Pattern 3: Approximate birth with standard death
    approx_birth_death = re.compile(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = approx_birth_death.search(person_string)
    if match:
        return match.group(1), match.group(2)
    
    # Pattern 4: Standard birth-death range
    standard_range = re.compile(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = standard_range.search(person_string)
    if match:
        return match.group(1), match.group(2)
    
    # Pattern 5: Death year only with approximate marker
    approx_death_only = re.compile(r'[-–—]\s*(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = approx_death_only.search(person_string)
    if match:
        return None, match.group(1)
    
    # Pattern 6: Death year only (simple)
    death_only = re.compile(r'[-–—]\s*(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = death_only.search(person_string)
    if match:
        return None, match.group(1)
    
    # Pattern 7: Birth year only with approximate marker
    approx_birth_only = re.compile(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
    match = approx_birth_only.search(person_string)
    if match:
        return match.group(1), None
    
    # Pattern 8: Birth year only (simple)
    birth_only = re.compile(r'(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)\s*[-–—]')
    match = birth_only.search(person_string)
    if match:
        return match.group(1), None
    
    # Pattern 9: Explicit birth/death prefixes (b., born, d., died)
    prefix_pattern = re.compile(r'(?:b\.|born)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)|(?:d\.|died)\s+(?:(?:approximately|ca\.|circa)\s+)?(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = prefix_pattern.search(person_string)
    if match:
        return (match.group(1), None) if match.group(1) else (None, match.group(2))
    
    # Pattern 10: Single approximate year (fallback)
    approx_year = re.compile(r'(?:approximately|ca\.|circa)\s+(\d{2,4}(?:\?|\s+or\s+\d{1,4})?)')
    match = approx_year.search(person_string)
    if match:
        return match.group(1), None
    
    return None, None