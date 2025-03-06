"""
Optimized Query Engine for Entity Resolution
Handles efficient vector retrieval and caching for large-scale operations
"""

import os
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional, Any
from weaviate.classes.query import Filter, MetadataQuery
import threading
from collections import defaultdict

from .utils import Timer, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class OptimizedQueryEngine:
    """Optimized query engine with advanced caching for entity resolution"""
    
    def __init__(self, config):
        """Initialize the optimized query engine with configuration"""
        self.config = config
        self.max_candidates = config['querying']['max_candidates']
        self.min_similarity = config['querying']['min_similarity']
        self.batch_size = config['querying']['batch_size']
        
        # Thread lock for cache access
        self._lock = threading.RLock()
        
        # Memory cache settings
        self.memory_cache_enabled = config['querying'].get('memory_cache_enabled', True)
        if self.memory_cache_enabled:
            self.max_memory_cache_size = config['querying'].get('max_memory_cache_size', 100000)
            self.vector_memory_cache = {}  # (hash, field_type) -> vector
            self.string_memory_cache = {}  # hash -> string
        
        # Disk cache settings
        self.disk_cache_enabled = config['querying'].get('disk_cache_enabled', False)
        if self.disk_cache_enabled:
            self.cache_dir = Path(config['querying']['cache_dir'])
            ensure_dir(self.cache_dir)
            
            try:
                # Try to import diskcache, but fall back to simple dict if not available
                from diskcache import Cache
                self.vector_disk_cache = Cache(directory=str(self.cache_dir / 'vectors'), size_limit=int(10e9))
                self.string_disk_cache = Cache(directory=str(self.cache_dir / 'strings'), size_limit=int(5e9))
                self.query_disk_cache = Cache(directory=str(self.cache_dir / 'queries'), size_limit=int(5e9))
            except ImportError:
                logger.warning("diskcache not available, using in-memory cache only")
                self.disk_cache_enabled = False
                self.vector_disk_cache = {}
                self.string_disk_cache = {}
                self.query_disk_cache = {}
        
        # Batch caching
        self.batch_cache = {}  # Temporary cache for current batch
        
        # Statistics for cache performance
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        
        # Reference to Weaviate collection
        self.collection = None
        
        logger.info("Optimized query engine initialized")
    
    def set_collection(self, collection):
        """Set the Weaviate collection reference"""
        self.collection = collection
        return self
    
    def get_vector_by_hash(self, hash_value, field_type=None):
        """Get vector for a hash value with multi-level caching"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return None
        
        # Create cache key
        cache_key = (hash_value, field_type)
        
        # Check memory cache first (fastest)
        if self.memory_cache_enabled:
            with self._lock:
                if cache_key in self.vector_memory_cache:
                    self.cache_hits['memory_vector'] += 1
                    return self.vector_memory_cache[cache_key]
        
        # Check batch cache (for current operation)
        if cache_key in self.batch_cache:
            self.cache_hits['batch_vector'] += 1
            return self.batch_cache[cache_key]
        
        # Check disk cache
        if self.disk_cache_enabled:
            disk_key = f"{hash_value}_{field_type}"
            try:
                vector = self.vector_disk_cache.get(disk_key)
                if vector is not None:
                    self.cache_hits['disk_vector'] += 1
                    # Add to memory cache for faster future access
                    if self.memory_cache_enabled:
                        with self._lock:
                            self.vector_memory_cache[cache_key] = vector
                    return vector
            except Exception as e:
                logger.debug(f"Error accessing disk cache: {e}")
        
        # Not in cache, fetch from Weaviate
        self.cache_misses['vector'] += 1
        vector = self._fetch_vector_from_weaviate(hash_value, field_type)
        
        # Cache the result
        if vector is not None:
            # Add to memory cache
            if self.memory_cache_enabled:
                with self._lock:
                    # Implement simple LRU eviction if cache is full
                    if len(self.vector_memory_cache) >= self.max_memory_cache_size:
                        # Remove a random entry (simple but effective)
                        self.vector_memory_cache.pop(next(iter(self.vector_memory_cache)))
                    self.vector_memory_cache[cache_key] = vector
            
            # Add to disk cache
            if self.disk_cache_enabled:
                disk_key = f"{hash_value}_{field_type}"
                try:
                    self.vector_disk_cache[disk_key] = vector
                except Exception as e:
                    logger.debug(f"Error writing to disk cache: {e}")
        
        return vector
    
    def _fetch_vector_from_weaviate(self, hash_value, field_type=None):
        """Fetch a vector from Weaviate (without caching)"""
        try:
            # Build filter
            if field_type:
                combined_filter = Filter.by_property("hash").equal(hash_value) & Filter.by_property("field_type").equal(field_type)
            else:
                combined_filter = Filter.by_property("hash").equal(hash_value)
            
            # Execute query
            result = self.collection.query.fetch_objects(
                filters=combined_filter,
                limit=1,
                include_vector=True
            )
            
            if not result.objects:
                return None
            
            # Extract vector
            vector = result.objects[0].vector.get('text_vector')
            if vector:
                return np.array(vector)
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting vector for hash {hash_value}: {e}")
            return None
    
    def batch_get_vectors(self, hash_field_pairs):
        """Efficiently fetch multiple vectors in a single batch
        
        Args:
            hash_field_pairs: List of (hash, field_type) tuples
        
        Returns:
            Dictionary mapping (hash, field_type) to vector
        """
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return {}
        
        # Initialize batch cache for this operation
        self.batch_cache = {}
        
        # Check which vectors are already cached
        missing_pairs = []
        for hash_value, field_type in hash_field_pairs:
            cache_key = (hash_value, field_type)
            
            # Check memory cache
            if self.memory_cache_enabled:
                with self._lock:
                    if cache_key in self.vector_memory_cache:
                        self.batch_cache[cache_key] = self.vector_memory_cache[cache_key]
                        self.cache_hits['memory_vector'] += 1
                        continue
            
            # Check disk cache
            if self.disk_cache_enabled:
                disk_key = f"{hash_value}_{field_type}"
                try:
                    vector = self.vector_disk_cache.get(disk_key)
                    if vector is not None:
                        self.batch_cache[cache_key] = vector
                        self.cache_hits['disk_vector'] += 1
                        continue
                except Exception as e:
                    logger.debug(f"Error accessing disk cache: {e}")
            
            # Not cached, need to fetch
            missing_pairs.append((hash_value, field_type))
        
        # If all vectors were cached, return early
        if not missing_pairs:
            return self.batch_cache
        
        # Group by field_type for efficient querying
        field_to_hashes = defaultdict(list)
        for hash_value, field_type in missing_pairs:
            field_to_hashes[field_type].append(hash_value)
        
        # Fetch each field type in batches
        for field_type, hashes in field_to_hashes.items():
            self._batch_fetch_vectors_by_field(hashes, field_type)
        
        # Return the combined batch cache
        return self.batch_cache
    
    def _batch_fetch_vectors_by_field(self, hashes, field_type, batch_size=50):
        """Fetch vectors for a specific field type in batches"""
        # Process in smaller batches to avoid filter complexity issues
        for i in range(0, len(hashes), batch_size):
            batch_hashes = hashes[i:i+batch_size]
            
            try:
                # Instead of building a complex OR filter, fetch vectors one by one
                # This is less efficient but more reliable across Weaviate versions
                for hash_value in batch_hashes:
                    try:
                        # Simple filter for a single hash value
                        hash_filter = Filter.by_property("hash").equal(hash_value)
                        
                        if field_type:
                            # Combine with field_type filter
                            combined_filter = hash_filter & Filter.by_property("field_type").equal(field_type)
                        else:
                            combined_filter = hash_filter
                        
                        # Execute single query
                        result = self.collection.query.fetch_objects(
                            filters=combined_filter,
                            limit=1,
                            include_vector=True
                        )
                        
                        # Process result if found
                        if result.objects:
                            obj = result.objects[0]
                            if hasattr(obj, 'properties') and hasattr(obj, 'vector'):
                                vector = obj.vector.get('text_vector')
                                if vector:
                                    # Convert to numpy array
                                    vector_array = np.array(vector)
                                    
                                    # Add to batch cache and results
                                    cache_key = (hash_value, field_type)
                                    self.batch_cache[cache_key] = vector_array
                                    
                                    # Add to memory cache
                                    if self.memory_cache_enabled:
                                        with self._lock:
                                            if len(self.vector_memory_cache) >= self.max_memory_cache_size:
                                                self.vector_memory_cache.pop(next(iter(self.vector_memory_cache)))
                                            self.vector_memory_cache[cache_key] = vector_array
                                    
                                    # Add to disk cache
                                    if self.disk_cache_enabled:
                                        disk_key = f"{hash_value}_{field_type}"
                                        try:
                                            self.vector_disk_cache[disk_key] = vector_array
                                        except Exception as e:
                                            logger.debug(f"Error writing to disk cache: {e}")
                    
                    except Exception as e:
                        logger.debug(f"Error fetching vector for hash {hash_value}: {e}")
            
            except Exception as e:
                logger.error(f"Error in batch fetching vectors for field {field_type}: {e}")
    
    def batch_get_strings(self, hash_values):
        """Efficiently fetch multiple original strings
        
        Args:
            hash_values: List of hash values
        
        Returns:
            Dictionary mapping hash to original string
        """
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return {}
        
        # Initialize result dictionary
        string_dict = {}
        
        # Check which strings are already cached
        missing_hashes = []
        for hash_value in hash_values:
            # Check memory cache
            if self.memory_cache_enabled:
                with self._lock:
                    if hash_value in self.string_memory_cache:
                        string_dict[hash_value] = self.string_memory_cache[hash_value]
                        self.cache_hits['memory_string'] += 1
                        continue
            
            # Check disk cache
            if self.disk_cache_enabled:
                try:
                    text = self.string_disk_cache.get(hash_value)
                    if text is not None:
                        string_dict[hash_value] = text
                        self.cache_hits['disk_string'] += 1
                        continue
                except Exception as e:
                    logger.debug(f"Error accessing disk cache: {e}")
            
            # Not cached, need to fetch
            missing_hashes.append(hash_value)
        
        # If all strings were cached, return early
        if not missing_hashes:
            return string_dict
        
        # Fetch missing strings in batches (smaller batches to reduce complexity)
        batch_size = min(50, self.batch_size)
        for i in range(0, len(missing_hashes), batch_size):
            batch_hashes = missing_hashes[i:i+batch_size]
            
            # Process each hash individually to avoid filter complexity issues
            for hash_value in batch_hashes:
                try:
                    # Simple filter for a single hash
                    hash_filter = Filter.by_property("hash").equal(hash_value)
                    
                    # Execute query
                    result = self.collection.query.fetch_objects(
                        filters=hash_filter,
                        limit=1,
                        include_vector=False  # Don't need vectors here
                    )
                    
                    # Process result if found
                    if result.objects:
                        obj = result.objects[0]
                        if hasattr(obj, 'properties'):
                            text = obj.properties.get('text')
                            
                            if text:
                                # Add to result
                                string_dict[hash_value] = text
                                
                                # Add to memory cache
                                if self.memory_cache_enabled:
                                    with self._lock:
                                        if len(self.string_memory_cache) >= self.max_memory_cache_size:
                                            self.string_memory_cache.pop(next(iter(self.string_memory_cache)))
                                        self.string_memory_cache[hash_value] = text
                                
                                # Add to disk cache
                                if self.disk_cache_enabled:
                                    try:
                                        self.string_disk_cache[hash_value] = text
                                    except Exception as e:
                                        logger.debug(f"Error writing to disk cache: {e}")
                
                except Exception as e:
                    logger.warning(f"Error fetching string for hash {hash_value}: {e}")
        
        # Update cache miss statistics
        self.cache_misses['string'] += len(missing_hashes)
        
        return string_dict
    
    def query_nearest_vectors(self, query_vector, field_type=None, limit=None, min_similarity=None):
        """Query nearest vectors to a query vector with caching"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return []
        
        if limit is None:
            limit = self.max_candidates
        
        if min_similarity is None:
            min_similarity = self.min_similarity
        
        # Generate cache key
        if self.disk_cache_enabled:
            # Create a deterministic hash of the vector
            vector_hash = hash(tuple(query_vector.flatten()))
            cache_key = f"query_{vector_hash}_{field_type}_{limit}_{min_similarity}"
            
            # Check disk cache
            try:
                cached_result = self.query_disk_cache.get(cache_key)
                if cached_result is not None:
                    self.cache_hits['query'] += 1
                    return cached_result
            except Exception as e:
                logger.debug(f"Error accessing query cache: {e}")
        
        self.cache_misses['query'] += 1
        
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
            if self.disk_cache_enabled:
                try:
                    self.query_disk_cache[cache_key] = filtered_results
                except Exception as e:
                    logger.debug(f"Error writing to query cache: {e}")
            
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
    
    def batch_find_candidates(self, person_hashes, min_similarity=None, limit=None):
        """Find candidate matches for multiple persons in batch"""
        if not self.collection:
            logger.error("Collection not set in QueryEngine")
            return {}
        
        # Initialize result dictionary
        candidates_dict = {}
        
        # Process person hashes in smaller batches to avoid memory issues
        batch_size = min(50, len(person_hashes))
        for i in range(0, len(person_hashes), batch_size):
            batch = person_hashes[i:i+batch_size]
            
            # Process each person individually to avoid complex filters
            for person_hash in batch:
                try:
                    # Find candidates using the standard method
                    candidates = self.find_candidates(
                        person_hash, 
                        min_similarity=min_similarity, 
                        limit=limit
                    )
                    candidates_dict[person_hash] = candidates
                except Exception as e:
                    logger.error(f"Error finding candidates for hash {person_hash}: {e}")
                    candidates_dict[person_hash] = []
        
        return candidates_dict
    
    def warm_cache_for_batch(self, record_pairs, fields_to_fetch):
        """Pre-warm cache for a batch of record pairs"""
        # Collect all unique hashes needed
        hash_field_pairs = []
        person_hashes = set()
        
        for left_record, right_record in record_pairs:
            for field in fields_to_fetch:
                if left_record and field in left_record and left_record[field] != "NULL":
                    hash_field_pairs.append((left_record[field], field))
                    if field == "person":
                        person_hashes.add(left_record[field])
                
                if right_record and field in right_record and right_record[field] != "NULL":
                    hash_field_pairs.append((right_record[field], field))
                    if field == "person":
                        person_hashes.add(right_record[field])
        
        # Deduplicate
        hash_field_pairs = list(set(hash_field_pairs))
        
        # Batch fetch vectors
        logger.info(f"Warming cache with {len(hash_field_pairs)} hash-field pairs")
        self.batch_get_vectors(hash_field_pairs)
        
        # Also fetch original strings for person hashes (for Levenshtein)
        self.batch_get_strings(list(person_hashes))
        
        return len(hash_field_pairs)
    
    def get_cache_stats(self):
        """Get cache statistics"""
        stats = {
            'hits': dict(self.cache_hits),
            'misses': dict(self.cache_misses),
            'memory_cache_size': 0,
            'disk_cache_size': 0
        }
        
        # Add memory cache sizes
        if self.memory_cache_enabled:
            stats['memory_cache_size'] = {
                'vectors': len(self.vector_memory_cache),
                'strings': len(self.string_memory_cache)
            }
        
        # Add disk cache sizes if possible
        if self.disk_cache_enabled:
            try:
                stats['disk_cache_size'] = {
                    'vectors': len(self.vector_disk_cache),
                    'strings': len(self.string_disk_cache),
                    'queries': len(self.query_disk_cache)
                }
            except:
                stats['disk_cache_size'] = "Not available"
        
        return stats
    
    def clear_cache(self, cache_type=None):
        """Clear specified cache or all caches"""
        if cache_type == 'memory' or cache_type is None:
            with self._lock:
                if self.memory_cache_enabled:
                    self.vector_memory_cache = {}
                    self.string_memory_cache = {}
                    logger.info("Memory cache cleared")
        
        if cache_type == 'batch' or cache_type is None:
            self.batch_cache = {}
            logger.info("Batch cache cleared")
        
        if cache_type == 'disk' or cache_type is None:
            if self.disk_cache_enabled:
                try:
                    self.vector_disk_cache.clear()
                    self.string_disk_cache.clear()
                    self.query_disk_cache.clear()
                    logger.info("Disk cache cleared")
                except:
                    logger.warning("Error clearing disk cache")
        
        if cache_type == 'stats' or cache_type is None:
            self.cache_hits = defaultdict(int)
            self.cache_misses = defaultdict(int)
            logger.info("Cache statistics reset")

    def close(self):
        """Close the Weaviate client connection properly"""
        if hasattr(self, 'client') and self.client:
            try:
                # Get all active connections from client
                self.client.close()
                logger.info("Weaviate connection closed")
            except Exception as e:
                logger.error(f"Error closing Weaviate connection: {e}")

        # Force garbage collection to release any lingering connections
        try:
            import gc
            gc.collect()
            logger.debug("Garbage collection performed")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
