"""
Batch Processing module for Entity Resolution
Handles efficient batch processing of records for feature extraction and classification
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Callable
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from collections import defaultdict
import math

from .utils import Timer, ensure_dir
from .utils_enhancement import chunk_iterable

# Configure logger
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processor for entity resolution"""
    
    def __init__(self, config):
        """Initialize the batch processor with configuration"""
        self.config = config
        self.batch_size = config.get('batch_processing', {}).get('batch_size', 1000)
        self.num_workers = min(
            config.get('batch_processing', {}).get('num_workers', 
                     config.get('general', {}).get('num_workers', 8)),
            mp.cpu_count()
        )
        self.memory_limit = config.get('batch_processing', {}).get('memory_limit_gb', 16) * 1024 * 1024 * 1024  # Convert to bytes
        
        # Performance tracking
        self.processed_batches = 0
        self.processed_items = 0
        self.total_processing_time = 0
        
        logger.info(f"Batch processor initialized with batch size {self.batch_size} and {self.num_workers} workers")
    
    def process_in_batches(self, items, process_func, desc="Processing", max_items=None, **kwargs):
        """Process items in batches with progress tracking
        
        Args:
            items: List or iterable of items to process
            process_func: Function to process each batch (must accept batch as first arg)
            desc: Description for progress bar
            max_items: Optional limit on number of items to process (for development mode)
            **kwargs: Additional keyword arguments to pass to process_func
            
        Returns:
            List of results from all batches
        """
        with Timer() as timer:
            # Apply optional limit for development mode
            if max_items is not None and max_items > 0:
                items = list(items)[:max_items]
                logger.info(f"Limited processing to {max_items} items (development mode)")
            
            # Create batches
            batches = list(chunk_iterable(items, self.batch_size))
            total_items = len(items) if hasattr(items, '__len__') else "unknown"
            
            logger.info(f"Processing {total_items} items in {len(batches)} batches")
            
            # Process batches with progress tracking
            results = []
            for batch_idx, batch in enumerate(tqdm(batches, desc=desc)):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} items)")
                
                # Process batch
                batch_result = process_func(batch, **kwargs)
                
                # Aggregate results
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
                
                # Update tracking
                self.processed_batches += 1
                self.processed_items += len(batch)
            
            logger.info(f"Batch processing completed: {self.processed_items} items in {self.processed_batches} batches")
        
        # Update total processing time
        self.total_processing_time += timer.elapsed
        
        return results
    
    def process_in_parallel(self, items, process_func, desc="Processing in parallel", 
                          max_items=None, show_progress=True, **kwargs):
        """Process items in parallel with batch chunking for memory efficiency
        
        Args:
            items: List or iterable of items to process
            process_func: Function to process each item (must accept item as first arg)
            desc: Description for progress bar
            max_items: Optional limit on number of items to process
            show_progress: Whether to show progress bar
            **kwargs: Additional keyword arguments to pass to process_func
            
        Returns:
            List of results from all items
        """
        with Timer() as timer:
            # Apply optional limit
            if max_items is not None and max_items > 0:
                items = list(items)[:max_items]
                logger.info(f"Limited processing to {max_items} items (development mode)")
            
            # Estimate memory per item and adjust batch size if needed
            estimated_memory_per_item = self._estimate_memory_per_item()
            if estimated_memory_per_item > 0:
                # Calculate maximum items that can fit in memory
                max_memory_items = max(1, int(self.memory_limit / (estimated_memory_per_item * self.num_workers)))
                adaptive_batch_size = min(self.batch_size, max_memory_items)
                
                if adaptive_batch_size < self.batch_size:
                    logger.info(f"Adjusting batch size from {self.batch_size} to {adaptive_batch_size} based on memory constraints")
                    batches = list(chunk_iterable(items, adaptive_batch_size))
                else:
                    batches = list(chunk_iterable(items, self.batch_size))
            else:
                batches = list(chunk_iterable(items, self.batch_size))
            
            total_items = len(items) if hasattr(items, '__len__') else "unknown"
            logger.info(f"Processing {total_items} items in {len(batches)} batches with {self.num_workers} workers")
            
            # Process each batch in parallel
            all_results = []
            
            # Setup progress tracking
            if show_progress:
                batches_iter = tqdm(batches, desc=desc)
            else:
                batches_iter = batches
            
            for batch_idx, batch in enumerate(batches_iter):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} items)")
                
                # Process batch items in parallel
                batch_results = Parallel(n_jobs=self.num_workers)(
                    delayed(process_func)(item, **kwargs) for item in batch
                )
                
                # Aggregate results
                all_results.extend(batch_results)
                
                # Update tracking
                self.processed_batches += 1
                self.processed_items += len(batch)
            
            logger.info(f"Parallel processing completed: {self.processed_items} items in {self.processed_batches} batches")
        
        # Update total processing time
        self.total_processing_time += timer.elapsed
        
        return all_results
    
    def process_batch_in_parallel(self, batch, batch_process_func, item_process_func, **kwargs):
        """Process a single batch with parallelization
        
        This method is useful when you need to do batch preparation first,
        then process individual items in parallel.
        
        Args:
            batch: Batch of items to process
            batch_process_func: Function to process the batch before parallelization
            item_process_func: Function to process each item in parallel
            **kwargs: Additional keyword arguments to pass to functions
            
        Returns:
            List of results from all items in the batch
        """
        # First process the batch
        batch_result = batch_process_func(batch, **kwargs)
        
        # Then process individual items in parallel
        return Parallel(n_jobs=self.num_workers)(
            delayed(item_process_func)(item, batch_result, **kwargs) for item in batch
        )
    
    def process_record_pairs(self, record_pairs, preprocessor, query_engine, feature_extractor):
        """Process record pairs for feature extraction in optimized batches
        
        This specialized method handles the most common entity resolution use case:
        extracting features from pairs of records.
        
        Args:
            record_pairs: List of (left_record, right_record) tuples
            preprocessor: Preprocessor instance for record lookup
            query_engine: Query engine for vector retrieval
            feature_extractor: Feature extractor for feature calculation
            
        Returns:
            List of feature vectors
        """
        with Timer() as timer:
            logger.info(f"Processing {len(record_pairs)} record pairs for feature extraction")
            
            # Define batch preparation function
            def prepare_batch(batch_pairs):
                # Collect all records in the batch
                all_records = {}
                for left_record, right_record in batch_pairs:
                    # Store records by their personId
                    if isinstance(left_record, dict) and 'personId' in left_record:
                        all_records[left_record['personId']] = left_record
                    if isinstance(right_record, dict) and 'personId' in right_record:
                        all_records[right_record['personId']] = right_record
                
                # Collect all field hashes that will be needed
                hash_field_pairs = []
                for record in all_records.values():
                    if record is None:
                        continue
                    for field in feature_extractor.fields_to_embed:
                        if field in record and record[field] != "NULL":
                            hash_field_pairs.append((record[field], field))
                
                # Deduplicate hash-field pairs
                hash_field_pairs = list(set(hash_field_pairs))
                
                # Warm up cache with batch fetch
                query_engine.warm_cache_for_batch(
                    [(left_record, right_record) 
                    for left_record, right_record in batch_pairs 
                    if left_record is not None and right_record is not None],
                    feature_extractor.fields_to_embed
                )
                
                return {
                    'records': all_records,
                    'hash_field_pairs': hash_field_pairs
                }
            
            # Define item processing function
            def process_pair(pair, batch_data):
                left_record, right_record = pair
                
                # Skip if either record is missing
                if left_record is None or right_record is None:
                    logger.warning(f"Missing record in pair")
                    return None
                
                # Extract features
                try:
                    return feature_extractor.extract_features(left_record, right_record, query_engine)
                except Exception as e:
                    logger.error(f"Error extracting features: {e}")
                    return None
            
            # Process in batches with parallelization
            feature_vectors = []
            
            # Create batches
            batches = list(chunk_iterable(record_pairs, self.batch_size))
            
            for batch_idx, batch in enumerate(tqdm(batches, desc="Extracting features")):
                if batch_idx % 10 == 0:
                    logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
                
                # Process batch
                batch_data = prepare_batch(batch)
                
                # Process items in parallel
                batch_results = Parallel(n_jobs=self.num_workers)(
                    delayed(process_pair)(pair, batch_data) for pair in batch
                )
                
                # Filter out None results and add to feature vectors
                batch_features = [vec for vec in batch_results if vec is not None]
                feature_vectors.extend(batch_features)
                
                # Update tracking
                self.processed_batches += 1
                self.processed_items += len(batch)
            
            logger.info(f"Feature extraction completed: {len(feature_vectors)} vectors extracted")
        
        # Update total processing time
        self.total_processing_time += timer.elapsed
        
        return feature_vectors
    
    def classify_dataset(self, person_ids, preprocessor, query_engine, feature_extractor, classifier, clusterer=None):
        """Classify dataset with optimized batch processing
        
        This specialized method handles entity classification and optional clustering.
        
        Args:
            person_ids: List of person IDs to classify
            preprocessor: Preprocessor instance for record lookup
            query_engine: Query engine for vector retrieval
            feature_extractor: Feature extractor for feature calculation
            classifier: Classifier for match prediction
            clusterer: Optional clusterer for entity clustering
            
        Returns:
            Dictionary with classification and optional clustering results
        """
        with Timer() as timer:
            logger.info(f"Classifying {len(person_ids)} entities")
            
            # Define batch processing function
            def process_batch(batch_ids):
                logger.info(f"Processing batch of {len(batch_ids)} entities")
                
                # Get records for all persons in batch
                records = {pid: preprocessor.get_record(pid) for pid in batch_ids}
                
                # Find candidate matches for all persons in batch
                batch_candidates = {}
                
                # Collect person hashes for all valid records
                person_hashes = {}
                for pid, record in records.items():
                    if record is not None and 'person' in record and record['person'] != "NULL":
                        person_hashes[pid] = record['person']
                
                # Batch query for candidates
                candidate_results = query_engine.batch_find_candidates(
                    list(person_hashes.values()),
                    min_similarity=classifier.config['querying']['min_similarity'],
                    limit=classifier.config['querying']['max_candidates']
                )
                
                # Map candidate hashes to person IDs
                reverse_mapping = {}
                for pid, hash_val in person_hashes.items():
                    if hash_val not in reverse_mapping:
                        reverse_mapping[hash_val] = []
                    reverse_mapping[hash_val].append(pid)
                
                # Process each person's candidates
                batch_match_pairs = []
                
                for pid in batch_ids:
                    # Skip if no record or no person hash
                    if pid not in person_hashes:
                        continue
                    
                    person_hash = person_hashes[pid]
                    candidates = candidate_results.get(person_hash, [])
                    
                    # Find all person IDs with these candidate hashes
                    for candidate_hash, similarity in candidates:
                        # Find person IDs with this hash
                        candidate_person_ids = []
                        for other_pid, other_hash in person_hashes.items():
                            if other_pid != pid and other_hash == candidate_hash:
                                candidate_person_ids.append(other_pid)
                        
                        # Process each candidate
                        for candidate_id in candidate_person_ids:
                            # Extract features
                            feature_vector = feature_extractor.extract_features(
                                records[pid], records[candidate_id], query_engine
                            )
                            
                            # Predict match probability
                            probability = classifier.predict(feature_vector)
                            
                            # Check if it's a match
                            if probability >= classifier.match_threshold:
                                batch_match_pairs.append((pid, candidate_id, float(probability)))
                
                return batch_match_pairs
            
            # Process in batches
            all_match_pairs = []
            
            # Create batches with dynamic sizing based on dataset size
            # For large datasets, use larger batches to reduce overhead
            if len(person_ids) > 100000:
                adaptive_batch_size = min(10000, self.batch_size * 5)
                logger.info(f"Large dataset detected, using larger batch size: {adaptive_batch_size}")
                batches = list(chunk_iterable(person_ids, adaptive_batch_size))
            else:
                batches = list(chunk_iterable(person_ids, self.batch_size))
            
            for batch_idx, batch in enumerate(tqdm(batches, desc="Classifying entities")):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} entities)")
                
                # Process batch
                batch_match_pairs = process_batch(batch)
                all_match_pairs.extend(batch_match_pairs)
                
                # Log interim progress
                logger.info(f"Batch {batch_idx+1} found {len(batch_match_pairs)} matches")
                
                # Update tracking
                self.processed_batches += 1
                self.processed_items += len(batch)
            
            # Deduplicate match pairs (some might be found from both sides)
            unique_match_pairs = {}
            for e1, e2, conf in all_match_pairs:
                # Ensure consistent ordering for deduplication
                if e1 > e2:
                    e1, e2 = e2, e1
                
                pair_key = (e1, e2)
                if pair_key not in unique_match_pairs or unique_match_pairs[pair_key] < conf:
                    unique_match_pairs[pair_key] = conf
            
            final_match_pairs = [(e1, e2, conf) for (e1, e2), conf in unique_match_pairs.items()]
            
            logger.info(f"Classification completed: {len(final_match_pairs)} unique matches found")
            
            results = {'match_pairs': final_match_pairs}
            
            # Cluster if requested
            if clusterer is not None:
                logger.info("Performing clustering on match pairs")
                clusterer.cluster(final_match_pairs)
                results['clusters'] = clusterer.get_clusters()
                logger.info(f"Clustering completed: {len(results['clusters'])} clusters found")
        
        # Update total processing time
        self.total_processing_time += timer.elapsed
        
        return results
    
    def _estimate_memory_per_item(self):
        """Estimate memory required per item based on embedding dimension"""
        try:
            embedding_dim = self.config['openai']['embedding_dim']
            # Rough estimate: embedding vector + overhead
            bytes_per_vector = embedding_dim * 4  # float32
            bytes_per_item = bytes_per_vector * 5  # Assume ~5 vectors per item + overhead
            
            return bytes_per_item
        except:
            # If we can't estimate, return -1
            return -1
    
    def get_statistics(self):
        """Get processing statistics"""
        return {
            'processed_batches': self.processed_batches,
            'processed_items': self.processed_items,
            'total_processing_time': self.total_processing_time,
            'items_per_second': self.processed_items / self.total_processing_time if self.total_processing_time > 0 else 0,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers
        }
