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
        
        # Get imputation settings with defaults
        imputation_config = config.get('imputation', {})
        self.enabled = imputation_config.get('enabled', False)
        self.neighbors = imputation_config.get('neighbors', 10)
        self.min_similarity = imputation_config.get('min_similarity', 0.7)
        self.similarity_weight_decay = imputation_config.get('similarity_weight_decay', 0.9)
        
        # Optional debug mode
        self.debug_mode = imputation_config.get('debug_mode', False)
        
        # Fields to impute
        self.fields_to_impute = config.get('fields', {}).get('impute', [])
        
        # Hash algorithm for computing new hashes
        self.hash_algorithm = config.get('preprocessing', {}).get('hash_algorithm', 'md5')
        
        # Cache for imputed values
        self.imputed_values_cache = {}
        
        # Log configuration
        logger.info(f"Imputer initialized: enabled={self.enabled}, fields={self.fields_to_impute}")
        if self.enabled:
            logger.info(f"Imputation params: neighbors={self.neighbors}, min_similarity={self.min_similarity}")
    
    def diagnose_and_fix_imputation(self, record, field_to_impute, query_engine):
        """Run a comprehensive diagnostic on why imputation is failing"""
        logger.info(f"\n===== IMPUTATION DIAGNOSTIC =====")
        logger.info(f"Diagnosing imputation failure for field: {field_to_impute}")
        
        # 1. Check if the field exists at all in Weaviate
        try:
            from weaviate.classes.aggregate import GroupByAggregate
            
            # Get field type distribution
            result = query_engine.collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="field_type"),
                total_count=True
            )
            
            field_counts = {}
            for group in result.groups:
                field_type = group.grouped_by.value
                count = group.total_count
                field_counts[field_type] = count
            
            logger.info("Field counts in Weaviate:")
            for field, count in field_counts.items():
                logger.info(f"  - {field}: {count}")
            
            if field_to_impute not in field_counts or field_counts[field_to_impute] == 0:
                logger.error(f"CRITICAL ISSUE: No '{field_to_impute}' data exists in Weaviate!")
                logger.error("This explains the imputation failure - there's no data to draw from.")
                logger.error("You need to index records with 'subjects' field data first.")
                return None
            
            # 2. Check if we can find ANY subjects with relaxed similarity
            logger.info("Testing query with very low similarity threshold...")
            
            # Get composite vector for testing
            if 'composite' in record and record['composite'] != "NULL":
                composite_vector = query_engine.get_vector_by_hash(record['composite'], "composite")
                
                if composite_vector is None:
                    logger.error("Cannot get composite vector for record!")
                    return None
                
                # Try with lower threshold
                from weaviate.classes.query import Filter, MetadataQuery
                
                field_filter = Filter.by_property("field_type").equal(field_to_impute)
                
                # Try progressively lower thresholds
                for threshold in [0.5, 0.3, 0.1, 0.0]:
                    logger.info(f"Testing with threshold: {threshold}")
                    
                    results = query_engine.collection.query.near_vector(
                        near_vector=composite_vector.tolist(),
                        limit=5,
                        return_metadata=MetadataQuery(distance=True),
                        filters=field_filter
                    )
                    
                    if results.objects:
                        # Found results! Show their similarities
                        for i, obj in enumerate(results.objects):
                            similarity = 1.0 - obj.metadata.distance
                            logger.info(f"  Result {i+1}: Similarity = {similarity:.4f}")
                        
                        # We found results but they don't meet the threshold
                        best_similarity = 1.0 - results.objects[0].metadata.distance
                        logger.info(f"SOLUTION: Lower min_similarity from {self.min_similarity} to {max(0.1, best_similarity - 0.05)}")
                        
                        # Dynamically adjust threshold for this field
                        self.min_similarity = max(0.1, best_similarity - 0.05)
                        logger.info(f"Adjusted similarity threshold to {self.min_similarity}")
                        
                        # Try imputation again with adjusted threshold
                        logger.info("Retrying imputation with adjusted threshold...")
                        return self._impute_field(record, field_to_impute, query_engine)
                    
                    logger.info(f"No results found with threshold {threshold}")
                
                # If we reach here, we couldn't find any results even with threshold 0
                logger.error("CRITICAL: No results found even with threshold 0!")
                logger.error("The 'subjects' field exists in the database but appears to be completely unrelated to your query vectors.")
                logger.error("This could indicate a data quality issue or improper indexing.")
            
            # End of diagnostics
            logger.info("===== END OF DIAGNOSTIC =====\n")
            return None
            
        except Exception as e:
            logger.error(f"Error during diagnostic: {e}", exc_info=True)
            return None

    def impute_record(self, record, query_engine=None, preprocessor=None):
        """Impute missing values in a record"""
        if not self.enabled or query_engine is None:
            return record
    
        # Create a copy to avoid modifying original
        imputed_record = record.copy()
        
        # Get record key first before logging
        record_key = self._get_record_key(record)
        logger.debug(f"Checking imputation for record: {record_key}")  # Use debug level
        
        # Check for null values that need imputation
        for field in self.fields_to_impute:
            if field not in record or record[field] == "NULL":
                logger.info(f"Need to impute {field} for record {record_key}")
                cache_key = f"{record_key}_{field}"
                
                if cache_key in self.imputed_values_cache:
                    imputed_record[field] = self.imputed_values_cache[cache_key]
                    logger.info(f"Used cached imputation for {field}")
                    continue
                
                # Need to impute
                imputed_hash = self._impute_field(record, field, query_engine, preprocessor)
                if imputed_hash:
                    imputed_record[field] = imputed_hash
                    # Cache the imputed value
                    self.imputed_values_cache[cache_key] = imputed_hash
                    logger.info(f"Successfully imputed {field} with hash {imputed_hash}")
                else:
                    logger.warning(f"Failed to impute {field} for record {record_key}")
        
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
        try:
            # Use composite field for imputation if available
            if 'composite' in record and record['composite'] != "NULL":
                query_hash = record['composite']
                logger.debug(f"Using composite hash {query_hash} for imputation of {field_to_impute}")
                
                # Get vector for the composite field
                query_vector = query_engine.get_vector_by_hash(query_hash, "composite")
                
                if query_vector is None:
                    logger.warning(f"No vector found for composite field hash: {query_hash}")
                    return None
                
                # Query nearest neighbors for the field to impute
                logger.debug(f"Querying nearest vectors for {field_to_impute}")
                results = query_engine.query_nearest_vectors(
                    query_vector,
                    field_type=field_to_impute,
                    limit=self.neighbors,
                    min_similarity=self.min_similarity
                )
                
                if not results:
                    logger.warning(f"No results found for imputing {field_to_impute}")
                    
                    # Only run diagnostic on the first failure to avoid spam
                    if not hasattr(self, '_diagnostic_run'):
                        self._diagnostic_run = True
                        return self.diagnose_and_fix_imputation(record, field_to_impute, query_engine)
                    
                    return None
                    
                # Determine whether we have a list or a Weaviate result object
                objects_list = results if isinstance(results, list) else results.objects

                if not objects_list or len(objects_list) == 0:
                    logger.warning(f"No objects found in results for imputing {field_to_impute}")
                    return None

                # Use objects_list throughout the rest of the method
                first_match = objects_list[0]                                    
                
                # Get the string value from the first match
                first_match_string = first_match.properties.get('text', '')
                if not first_match_string:
                    logger.warning(f"No text property found in first match")
                    return None
                    
                # Compute hash for this string
                imputed_hash = self._compute_hash(first_match_string)
                
                # Calculate weighted average vector based on similarity
                weights = []
                vectors = []
                
                for i, result in enumerate(objects_list):
                    try:
                        similarity = 1.0 - result.metadata.distance
                        weight = similarity * (self.similarity_weight_decay ** i)  # Apply decay
                        weights.append(weight)
                        
                        # Get the vector for this result
                        vector = result.vector.get('text_vector')
                        if vector:
                            vectors.append(np.array(vector))
                    except Exception as e:
                        logger.warning(f"Error processing result {i}: {e}")
                
                if not vectors:
                    logger.warning("No valid vectors found for imputation")
                    return None
                
                # Calculate weighted average vector
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize weights
                
                weighted_vector = np.zeros_like(vectors[0])
                for i, vec in enumerate(vectors):
                    weighted_vector += weights[i] * vec
                
                # Store the computed vector in Weaviate with the hash
                if not self._persist_imputed_vector(
                    imputed_hash, 
                    field_to_impute, 
                    first_match_string, 
                    weighted_vector, 
                    query_engine
                ):
                    logger.warning(f"Failed to persist imputed vector in Weaviate")
                    return None
                
                # Update hash mapping if preprocessor is available
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
            else:
                logger.warning(f"No composite field available for imputation")
                return None
        
        except Exception as e:
            logger.error(f"Error during imputation: {e}", exc_info=True)
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
                    return True
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
                    return True
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
                    return True
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