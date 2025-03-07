"""
Imputation module for entity resolution
Handles null value imputation using vector-based hot deck approach
"""

import logging
import numpy as np
import hashlib
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5  # Add this import
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
        
        # Hash algorithm for computing new hashes
        self.hash_algorithm = config['preprocessing']['hash_algorithm']
        
        # Cache for imputed values
        self.imputed_values_cache = {}
    
    def impute_record(self, record, query_engine=None, preprocessor=None):
        """Impute missing values in a record"""
        if not self.enabled or query_engine is None:
            return record
        
        # Create a copy of the record to avoid modifying the original
        imputed_record = record.copy()
        
        # Check for null values that need imputation
        # In the impute_record method:
        logger.info(f"Imputing record: {record_key}, fields: {self.fields_to_impute}")
        for field in self.fields_to_impute:
            if field not in record or record[field] == "NULL":
                logger.info(f"Need to impute {field}")
                record_key = self._get_record_key(record)
                cache_key = f"{record_key}_{field}"
                
                if cache_key in self.imputed_values_cache:
                    imputed_record[field] = self.imputed_values_cache[cache_key]
                    continue
                
                # Need to impute
                imputed_hash = self._impute_field(record, field, query_engine, preprocessor)
                if imputed_hash:
                    imputed_record[field] = imputed_hash
                    # Cache the imputed value
                    self.imputed_values_cache[cache_key] = imputed_hash
        
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
    
    def _impute_field(self, record, field_to_impute, query_engine, preprocessor=None):
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
                    
                # Determine whether we have a list or a Weaviate result object
                objects_list = results if isinstance(results, list) else results.objects

                if not objects_list or len(objects_list) == 0:
                    logger.debug(f"No objects found in results for imputing {field_to_impute}")
                    return None

                # Use objects_list throughout the rest of the method
                first_match = objects_list[0]                                    
                
                # *** REQUIREMENT: Retrieve the string value for the first match ***
                first_match_string = first_match.properties.get('text', '')
                
                # *** REQUIREMENT: Compute hash for this string ***
                imputed_hash = self._compute_hash(first_match_string)
                
                # Calculate weighted average vector based on similarity
                weights = []
                vectors = []
                
                for i, result in enumerate(objects_list):
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
                
                # *** REQUIREMENT: Store the computed vector in Weaviate with the hash ***
                self._persist_imputed_vector(
                    imputed_hash, 
                    field_to_impute, 
                    first_match_string, 
                    weighted_vector, 
                    query_engine
                )
                
                # *** REQUIREMENT: Update hash mapping if preprocessor is available ***
                if preprocessor is not None:
                    record_id = record.get('personId')
                    if record_id:
                        # Update the record's field hash
                        if record_id in preprocessor.record_field_hashes:
                            preprocessor.record_field_hashes[record_id][field_to_impute] = imputed_hash
                        
                        # Update field hash mapping
                        if imputed_hash not in preprocessor.field_hash_mapping:
                            preprocessor.field_hash_mapping[imputed_hash] = {}
                        preprocessor.field_hash_mapping[imputed_hash][field_to_impute] = 1
                
                return imputed_hash
            
            return None
        
        except Exception as e:
            logger.error(f"Error during imputation: {e}")
            return None
    
    def _compute_hash(self, value):
        """Compute hash for a string value"""
        if self.hash_algorithm == 'md5':
            return hashlib.md5(value.encode('utf-8')).hexdigest()
        elif self.hash_algorithm == 'sha1':
            return hashlib.sha1(value.encode('utf-8')).hexdigest()
        elif self.hash_algorithm == 'mmh3':
            try:
                import mmh3
                # Convert to hex string to match md5/sha1 format
                hash_value = mmh3.hash128(value.encode('utf-8'), signed=False)
                return format(hash_value, 'x')
            except ImportError:
                logger.warning("mmh3 module not found, falling back to md5")
                return hashlib.md5(value.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def _persist_imputed_vector(self, hash_val, field_type, text_value, vector, query_engine):
        """Persist the imputed vector in Weaviate"""
        try:
            # Get the collection
            collection = query_engine.collection
            
            # Generate deterministic UUID for this hash and field type
            obj_uuid = generate_uuid5(f"{hash_val}_{field_type}")
            
            # Default frequency for imputed values
            frequency = 1
            
            try:
                # First check if the object exists
                existing_obj = collection.query.fetch_object_by_id(obj_uuid)
                if existing_obj:
                    # Object exists, update it
                    collection.data.update(
                        uuid=obj_uuid,
                        properties={
                            "text": text_value,
                            "hash": hash_val,
                            "frequency": frequency,
                            "field_type": field_type,
                        },
                        vector={"text_vector": vector.tolist()}
                    )
                    logger.debug(f"Updated existing vector for {hash_val} in field {field_type}")
                else:
                    # Object doesn't exist, insert it
                    collection.data.insert(
                        properties={
                            "text": text_value,
                            "hash": hash_val,
                            "frequency": frequency,
                            "field_type": field_type,
                        },
                        vector={"text_vector": vector.tolist()},
                        uuid=obj_uuid
                    )
                    logger.debug(f"Inserted new vector for {hash_val} in field {field_type}")
            except Exception as e:
                if "already exists" in str(e):
                    # If it fails because the ID already exists, try updating instead
                    collection.data.update(
                        uuid=obj_uuid,
                        properties={
                            "text": text_value,
                            "hash": hash_val,
                            "frequency": frequency,
                            "field_type": field_type,
                        },
                        vector={"text_vector": vector.tolist()}
                    )
                    logger.debug(f"Updated existing vector after insert failure for {hash_val} in field {field_type}")
                else:
                    # Re-raise if it's some other error
                    raise
            
            return True
        
        except Exception as e:
            logger.error(f"Error persisting imputed vector: {e}")
            return False
    
    def batch_impute(self, records, query_engine, preprocessor=None):
        """Impute missing values for a batch of records"""
        if not self.enabled or query_engine is None:
            return records
        
        imputed_records = {}
        for record_id, record in tqdm(records.items(), desc="Imputing records"):
            imputed_records[record_id] = self.impute_record(record, query_engine, preprocessor)
        
        return imputed_records