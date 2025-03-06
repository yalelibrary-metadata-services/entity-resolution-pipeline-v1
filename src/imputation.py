"""
Imputation module for entity resolution
Handles null value imputation using vector-based hot deck approach
"""

import logging
import numpy as np
from weaviate.classes.query import Filter, MetadataQuery
from tqdm import tqdm

from .utils import Timer

# Configure logger
logger = logging.getLogger(__name__)

class Imputer:
    """Vector-based imputer for entity resolution"""
    
    def __init__(self, config):
        """Initialize the imputer with configuration"""
        self.config = config
        self.enabled = config['imputation']['enabled']
        self.neighbors = config['imputation']['neighbors']
        self.min_similarity = config['imputation']['min_similarity']
        self.similarity_weight_decay = config['imputation']['similarity_weight_decay']
        
        # Fields to impute
        self.fields_to_impute = config['fields']['impute']
        
        # Cache for imputed values
        self.imputed_values_cache = {}
    
    def impute_record(self, record, query_engine=None):
        """Impute missing values in a record"""
        if not self.enabled or query_engine is None:
            return record
        
        # Create a copy of the record to avoid modifying the original
        imputed_record = record.copy()
        
        # Check for null values that need imputation
        for field in self.fields_to_impute:
            if field not in record or record[field] == "NULL":
                # Check if already cached
                record_key = self._get_record_key(record)
                cache_key = f"{record_key}_{field}"
                
                if cache_key in self.imputed_values_cache:
                    imputed_record[field] = self.imputed_values_cache[cache_key]
                    continue
                
                # Need to impute
                imputed_value = self._impute_field(record, field, query_engine)
                if imputed_value:
                    imputed_record[field] = imputed_value
                    # Cache the imputed value
                    self.imputed_values_cache[cache_key] = imputed_value
        
        return imputed_record
    
    def _get_record_key(self, record):
        """Generate a key for the record for caching"""
        # Use composite key or person field as the key
        if 'composite' in record:
            return record['composite']
        elif 'person' in record:
            return record['person']
        else:
            # Fallback to concatenating all field values
            return "_".join(str(v) for v in record.values())
    
    def _impute_field(self, record, field_to_impute, query_engine):
        """Impute a specific field using vector-based hot deck approach"""
        try:
            # Use composite field for imputation if available
            if 'composite' in record and record['composite'] != "NULL":
                query_hash = record['composite']
                # Get vector for the composite field
                query_vector = query_engine.get_vector_by_hash(query_hash, "composite")
                
                if query_vector is None:
                    logger.warning(f"No vector found for composite field hash: {query_hash}")
                    return None
                
                # Query nearest neighbors for the field to impute
                results = query_engine.query_nearest_vectors(
                    query_vector,
                    field_type=field_to_impute,
                    limit=self.neighbors,
                    min_similarity=self.min_similarity
                )
                
                if not results:
                    logger.debug(f"No results found for imputing {field_to_impute}")
                    return None
                
                # Calculate weighted average vector based on similarity
                weights = []
                vectors = []
                
                for i, result in enumerate(results):
                    similarity = 1.0 - result.metadata.distance  # Convert distance to similarity
                    weight = similarity * (self.similarity_weight_decay ** i)  # Apply decay
                    weights.append(weight)
                    
                    # Get the vector for this result
                    vector = result.vector.get('text_vector')
                    if vector:
                        vectors.append(np.array(vector))
                
                if not vectors:
                    logger.warning("No valid vectors found for imputation")
                    return None
                
                # Calculate weighted average vector
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                
                weighted_vector = np.zeros_like(vectors[0])
                for i, vec in enumerate(vectors):
                    weighted_vector += weights[i] * vec
                
                # Find the most similar existing hash for this field
                most_similar_hash = query_engine.find_most_similar_hash(
                    weighted_vector,
                    field_type=field_to_impute
                )
                
                if most_similar_hash:
                    return most_similar_hash
            
            return None
        
        except Exception as e:
            logger.error(f"Error during imputation: {e}")
            return None
    
    def batch_impute(self, records, query_engine):
        """Impute missing values for a batch of records"""
        if not self.enabled or query_engine is None:
            return records
        
        imputed_records = {}
        for record_id, record in tqdm(records.items(), desc="Imputing records"):
            imputed_records[record_id] = self.impute_record(record, query_engine)
        
        return imputed_records
