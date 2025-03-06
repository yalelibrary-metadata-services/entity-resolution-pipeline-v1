"""
Querying module for entity resolution
Handles vector queries and candidate retrieval
"""

import os
import time
import pickle
import logging
import numpy as np
from pathlib import Path
from weaviate.classes.query import Filter, MetadataQuery

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class QueryEngine:
    """Query engine for entity resolution"""
    
    def __init__(self, config):
        """Initialize the query engine with configuration"""
        self.config = config
        self.max_candidates = config['querying']['max_candidates']
        self.min_similarity = config['querying']['min_similarity']
        self.batch_size = config['querying']['batch_size']
        
        # Cache settings
        self.cache_enabled = config['querying']['cache_enabled']
        if self.cache_enabled:
            self.cache_dir = Path(config['querying']['cache_dir'])
            ensure_dir(self.cache_dir)
            self.vector_cache = {}
            self.query_cache = {}
        
        # Reference to Weaviate collection
        self.collection = None
    
    def set_collection(self, collection):
        """Set the Weaviate collection reference"""
        self.collection = collection
        return self
    
    def get_vector_by_hash(self, hash_value, field_type=None):
        """Get vector for a hash value"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return None
        
        # Check cache first
        cache_key = f"{hash_value}_{field_type}"
        if self.cache_enabled and cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        try:
            # Build filter
            hash_filter = Filter.by_property("hash").equal(hash_value)
            
            if field_type:
                # Updated for Weaviate client v4
                combined_filter = Filter.by_property("hash").equal(hash_value) & Filter.by_property("field_type").equal(field_type)
            else:
                combined_filter = hash_filter
            
            # if field_type:
            #     field_filter = Filter.by_property("field_type").equal(field_type)
            #     combined_filter = Filter.and_operator([hash_filter, field_filter])
            # else:
            #     combined_filter = hash_filter
            
            # Execute query
            result = self.collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            if not result.objects:
                logger.debug(f"No vector found for hash: {hash_value}")
                return None
            
            # Extract vector
            vector = result.objects[0].vector.get('text_vector')
            if vector:
                vector = np.array(vector)
                
                # Cache result
                if self.cache_enabled:
                    self.vector_cache[cache_key] = vector
                
                return vector
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting vector for hash {hash_value}: {e}")
            return None
    
    def query_nearest_vectors(self, query_vector, field_type=None, limit=None, min_similarity=None):
        """Query nearest vectors to a query vector"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return []
        
        if limit is None:
            limit = self.max_candidates
        
        if min_similarity is None:
            min_similarity = self.min_similarity
        
        # Generate cache key
        if self.cache_enabled:
            # Convert vector to bytes for hashing
            vector_bytes = query_vector.tobytes()
            cache_key = f"{hash(vector_bytes)}_{field_type}_{limit}_{min_similarity}"
            
            # Check cache
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        results = pickle.load(f)
                    self.query_cache[cache_key] = results
                    return results
                except Exception as e:
                    logger.warning(f"Error loading query cache: {e}")
        
        try:
            # Build filter if field_type is specified
            query_filter = None
            if field_type:
                query_filter = Filter.by_property("field_type").equal(field_type)
            
            # Execute near vector query
            results = self.collection.query.near_vector(
                near_vector=query_vector.tolist(),
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                include_vector=True,
                filters=query_filter
            )
            
            # Filter by similarity threshold
            filtered_results = [
                obj for obj in results.objects
                if 1.0 - obj.metadata.distance >= min_similarity
            ]
            
            # Cache results
            if self.cache_enabled:
                self.query_cache[cache_key] = filtered_results
                
                # Save to disk cache
                try:
                    with open(self.cache_dir / f"{cache_key}.pkl", 'wb') as f:
                        pickle.dump(filtered_results, f)
                except Exception as e:
                    logger.warning(f"Error saving query cache: {e}")
            
            return filtered_results
        
        except Exception as e:
            logger.error(f"Error querying nearest vectors: {e}")
            return []
    
    def find_candidates(self, person_hash, min_similarity=None, limit=None):
        """Find candidate matches for a person"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return []
        
        # Get vector for person
        person_vector = self.get_vector_by_hash(person_hash, field_type="person")
        if person_vector is None:
            logger.warning(f"No vector found for person hash: {person_hash}")
            return []
        
        # Query nearest vectors
        results = self.query_nearest_vectors(
            person_vector,
            field_type="person",
            limit=limit,
            min_similarity=min_similarity
        )
        
        # Extract candidate hashes
        candidates = []
        for result in results:
            candidate_hash = result.properties.get('hash')
            if candidate_hash and candidate_hash != person_hash:
                similarity = 1.0 - result.metadata.distance
                candidates.append((candidate_hash, similarity))
        
        return candidates
    
    def find_most_similar_hash(self, query_vector, field_type, limit=10):
        """Find the most similar hash for a field type"""
        results = self.query_nearest_vectors(
            query_vector,
            field_type=field_type,
            limit=limit
        )
        
        if not results:
            return None
        
        # Return the hash of the most similar result
        return results[0].properties.get('hash')
    
    def clear_cache(self):
        """Clear query cache"""
        if self.cache_enabled:
            self.vector_cache = {}
            self.query_cache = {}
            logger.info("Query cache cleared")
