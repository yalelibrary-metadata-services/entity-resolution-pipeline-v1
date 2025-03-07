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
    
    def process_record_pairs(self, record_pairs, preprocessor, query_engine, feature_extractor, imputer=None):  
        """Process record pairs for feature extraction in batches with sequential processing and improved progress tracking"""
        with Timer() as timer:
            total_pairs = len(record_pairs)
            logger.info(f"Processing {total_pairs} record pairs for feature extraction")
            
            # Import necessary utilities at the method level to avoid circular imports
            from src.utils import compute_vector_similarity, compute_levenshtein_similarity
            
            # Import tqdm properly
            from tqdm import tqdm as tqdm_func
            import time
            
            # Create smaller sub-batches for better progress tracking
            sub_batch_size = min(self.batch_size, 1000)  # Use smaller sub-batches for better tracking
            batches = list(chunk_iterable(record_pairs, sub_batch_size))
            
            logger.info(f"Split into {len(batches)} sub-batches of about {sub_batch_size} pairs each")
            
            # Collect all hash-field pairs and person hashes that will be needed
            logger.info("Collecting hash-field pairs for prefetching...")
            hash_field_pairs = []
            person_hashes = set()
            
            for left_record, right_record in tqdm_func(record_pairs, desc="Preparing data", leave=False):
                for field in feature_extractor.fields_to_embed:
                    if left_record and field in left_record and left_record[field] != "NULL":
                        hash_field_pairs.append((left_record[field], field))
                        if field == "person":
                            person_hashes.add(left_record[field])
                    
                    if right_record and field in right_record and right_record[field] != "NULL":
                        hash_field_pairs.append((right_record[field], field))
                        if field == "person":
                            person_hashes.add(right_record[field])
            
            # Deduplicate hash-field pairs
            hash_field_pairs = list(set(hash_field_pairs))
            person_hashes = list(person_hashes)
            
            logger.info(f"Prefetching {len(hash_field_pairs)} vectors and {len(person_hashes)} strings")
            
            # Batch fetch all vectors and strings
            vectors_dict = query_engine.batch_get_vectors(hash_field_pairs)
            strings_dict = query_engine.batch_get_strings(person_hashes)
            
            logger.info(f"Successfully prefetched {len(vectors_dict)} vectors and {len(strings_dict)} strings")
            
            # Process in batches with progress bar
            all_feature_vectors = []
            
            # Create the main progress bar with a more accurate description
            pbar = tqdm_func(total=total_pairs, desc="Extracting features")
            
            # Track sub-batch success rates for monitoring
            successful_extractions = 0
            total_attempted = 0
            
            # Define a local sequential version of extract_features_for_pair
            def extract_features_for_pair(pair_data):
                """Local sequential version of feature extraction function"""
                import numpy as np
                from scipy.spatial.distance import cosine
                import traceback
                
                # Unpack the additional imputer parameter
                left_record, right_record, feature_names, vectors_dict, strings_dict, fields_to_embed, imputer, query_engine = pair_data
                
                # Skip if either record is missing
                if left_record is None or right_record is None:
                    return None
                
                try:
                    # Apply imputation if enabled
                    if imputer is not None and imputer.enabled:
                        if left_record:
                            left_record = imputer.impute_record(left_record, query_engine)
                        if right_record:
                            right_record = imputer.impute_record(right_record, query_engine)

                    # Extract features using only the provided data
                    features = {}
                    
                    # Get vectors for both records
                    left_vectors = {}
                    right_vectors = {}
                    
                    for field in fields_to_embed:
                        if field in left_record and left_record[field] != "NULL":
                            key = (left_record[field], field)
                            if key in vectors_dict:
                                left_vectors[field] = vectors_dict[key]
                        
                        if field in right_record and right_record[field] != "NULL":
                            key = (right_record[field], field)
                            if key in vectors_dict:
                                right_vectors[field] = vectors_dict[key]
                    
                    # Compute cosine similarities
                    for field in fields_to_embed:
                        if field in left_vectors and field in right_vectors:
                            # Compute cosine similarity
                            vec1 = left_vectors[field]
                            vec2 = right_vectors[field]
                            if vec1 is not None and vec2 is not None:
                                try:
                                    distance = cosine(vec1, vec2)
                                    cosine_sim = 1.0 - distance
                                except:
                                    cosine_sim = 0.0
                            else:
                                cosine_sim = 0.0
                            features[f"{field}_cosine"] = cosine_sim
                        else:
                            features[f"{field}_cosine"] = 0.0
                    
                    # Compute Levenshtein similarity for person names
                    left_person_hash = left_record.get('person')
                    right_person_hash = right_record.get('person')
                    
                    if left_person_hash != "NULL" and right_person_hash != "NULL":
                        left_person = strings_dict.get(left_person_hash)
                        right_person = strings_dict.get(right_person_hash)
                        
                        if left_person and right_person:
                            # Compute Levenshtein similarity
                            if not left_person or not right_person:
                                levenshtein_sim = 0.0
                            else:
                                try:
                                    import Levenshtein
                                    max_len = max(len(left_person), len(right_person))
                                    if max_len == 0:
                                        levenshtein_sim = 1.0
                                    else:
                                        distance = Levenshtein.distance(left_person, right_person)
                                        levenshtein_sim = 1.0 - (distance / max_len)
                                except:
                                    levenshtein_sim = 0.0
                            features['person_levenshtein'] = levenshtein_sim
                        else:
                            features['person_levenshtein'] = 0.0
                    else:
                        features['person_levenshtein'] = 0.0
                    
                    # Helper function for harmonic mean
                    def harmonic_mean(a, b):
                        if a <= 0 or b <= 0:
                            return 0.0
                        return 2 * a * b / (a + b)
                    
                    # Add interaction features
                    person_cosine = features.get('person_cosine', 0.0)
                    title_cosine = features.get('title_cosine', 0.0)
                    provision_cosine = features.get('provision_cosine', 0.0)
                    subjects_cosine = features.get('subjects_cosine', 0.0)
                    composite_cosine = features.get('composite_cosine', 0.0)
                    
                    # Calculate harmonic means
                    features['person_title_harmonic'] = harmonic_mean(person_cosine, title_cosine)
                    features['person_provision_harmonic'] = harmonic_mean(person_cosine, provision_cosine)
                    features['person_subjects_harmonic'] = harmonic_mean(person_cosine, subjects_cosine)
                    features['title_subjects_harmonic'] = harmonic_mean(title_cosine, subjects_cosine)
                    features['title_provision_harmonic'] = harmonic_mean(title_cosine, provision_cosine)
                    features['provision_subjects_harmonic'] = harmonic_mean(provision_cosine, subjects_cosine)
                    
                    # Other interaction features
                    features['person_subjects_product'] = person_cosine * subjects_cosine
                    
                    if subjects_cosine > 0:
                        features['composite_subjects_ratio'] = composite_cosine / subjects_cosine
                    else:
                        features['composite_subjects_ratio'] = 0.0
                    
                    # Simplified birth/death check
                    features['birth_death_year_match'] = 0.0
                    
                    # Convert features dictionary to numpy array
                    feature_vector = np.array([features.get(name, 0.0) for name in feature_names])
                    
                    return feature_vector
                except Exception as e:
                    # Capture the full traceback for better debugging
                    error_msg = traceback.format_exc()
                    # Print error but don't crash the worker
                    print(f"Error extracting features: {str(e)}\n{error_msg}")
                    return None
            
            # Process each batch sequentially
            for batch_idx, batch in enumerate(batches):
                batch_size = len(batch)
                batch_start_time = time.time()
                logger.info(f"Processing sub-batch {batch_idx+1}/{len(batches)} ({batch_size} pairs)")
                
                # Process each pair in the batch sequentially
                batch_results = []
                
                # Prepare batch data and process sequentially with progress updates
                counter = 0
                for left_record, right_record in batch:
                    # Prepare data
                    pair_data = (
                        left_record, 
                        right_record, 
                        feature_extractor.feature_names,
                        vectors_dict,
                        strings_dict,
                        feature_extractor.fields_to_embed,
                        imputer,  # Add imputer
                        query_engine  # Add query_engine for imputation
                    )
                    
                    # Extract features
                    result = extract_features_for_pair(pair_data)
                    batch_results.append(result)
                    total_attempted += 1
                    
                    # Update progress bar periodically to avoid slowdown from too many updates
                    counter += 1
                    if counter % 10 == 0 or counter == batch_size:
                        pbar.update(10 if counter != batch_size else counter % 10)
                
                # Filter out None results and add to feature vectors
                batch_features = [vec for vec in batch_results if vec is not None]
                successful_extractions += len(batch_features)
                all_feature_vectors.extend(batch_features)
                
                # Update stats
                self.processed_batches += 1
                self.processed_items += batch_size
                
                # Calculate and log batch statistics
                batch_time = time.time() - batch_start_time
                success_rate = (len(batch_features) / batch_size) * 100 if batch_size > 0 else 0
                pairs_per_second = batch_size / batch_time if batch_time > 0 else 0
                
                logger.info(f"Sub-batch {batch_idx+1} completed: {len(batch_features)}/{batch_size} " +
                            f"successful extractions ({success_rate:.1f}%), " +
                            f"{pairs_per_second:.1f} pairs/sec")
                
                # Periodically log overall progress
                if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
                    overall_success_rate = (successful_extractions / total_attempted) * 100 if total_attempted > 0 else 0
                    percent_complete = (self.processed_items / total_pairs) * 100
                    
                    logger.info(f"Overall progress: {percent_complete:.1f}% complete, " +
                                f"{successful_extractions}/{total_attempted} successful extractions " +
                                f"({overall_success_rate:.1f}%)")
            
            # Close progress bar
            pbar.close()
            
            overall_success_rate = (successful_extractions / total_attempted) * 100 if total_attempted > 0 else 0
            logger.info(f"Feature extraction completed: {len(all_feature_vectors)}/{total_pairs} feature vectors extracted " +
                        f"({overall_success_rate:.1f}% success rate)")
        
        # Update total processing time
        self.total_processing_time += timer.elapsed
        
        return all_feature_vectors
    
    def classify_dataset(self, person_ids, preprocessor, query_engine, feature_extractor, classifier, imputer=None, clusterer=None):
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
                                records[pid], records[candidate_id], query_engine, imputer
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

def _extract_features_for_pair(pair_data):
    """Standalone feature extraction function for multiprocessing with improved error handling
    
    Args:
        pair_data: Tuple containing (left_record, right_record, feature_names, 
                  vectors_dict, strings_dict, fields_to_embed)
        
    Returns:
        Feature vector or None if extraction fails
    """
    import numpy as np
    from scipy.spatial.distance import cosine
    import Levenshtein  # Make sure this is installed
    import traceback
    
    left_record, right_record, feature_names, vectors_dict, strings_dict, fields_to_embed = pair_data
    
    # Skip if either record is missing
    if left_record is None or right_record is None:
        return None
    
    # Define utility functions inside the worker function
    def compute_vector_similarity(vec1, vec2, metric='cosine'):
        """Compute similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        try:
            if metric == 'cosine':
                # Cosine distance is 1 - cosine similarity
                distance = cosine(vec1, vec2)
                return 1.0 - distance
            elif metric == 'dot':
                return np.dot(vec1, vec2)
            elif metric == 'euclidean':
                dist = np.linalg.norm(vec1 - vec2)
                return 1.0 / (1.0 + dist)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def compute_levenshtein_similarity(s1, s2):
        """Compute Levenshtein similarity between two strings"""
        if not s1 or not s2:
            return 0.0
        
        try:
            # Normalize by max length
            max_len = max(len(s1), len(s2))
            if max_len == 0:
                return 1.0
            
            distance = Levenshtein.distance(s1, s2)
            similarity = 1.0 - (distance / max_len)
            return similarity
        except Exception:
            return 0.0
    
    try:
        # Extract features using only the provided data
        features = {}
        
        # Get vectors for both records
        left_vectors = {}
        right_vectors = {}
        
        for field in fields_to_embed:
            if field in left_record and left_record[field] != "NULL":
                key = (left_record[field], field)
                if key in vectors_dict:
                    left_vectors[field] = vectors_dict[key]
            
            if field in right_record and right_record[field] != "NULL":
                key = (right_record[field], field)
                if key in vectors_dict:
                    right_vectors[field] = vectors_dict[key]
        
        # Compute cosine similarities
        for field in fields_to_embed:
            if field in left_vectors and field in right_vectors:
                cosine_sim = compute_vector_similarity(left_vectors[field], right_vectors[field])
                features[f"{field}_cosine"] = cosine_sim
            else:
                features[f"{field}_cosine"] = 0.0
        
        # Compute Levenshtein similarity for person names
        left_person_hash = left_record.get('person')
        right_person_hash = right_record.get('person')
        
        if left_person_hash != "NULL" and right_person_hash != "NULL":
            left_person = strings_dict.get(left_person_hash)
            right_person = strings_dict.get(right_person_hash)
            
            if left_person and right_person:
                levenshtein_sim = compute_levenshtein_similarity(left_person, right_person)
                features['person_levenshtein'] = levenshtein_sim
            else:
                features['person_levenshtein'] = 0.0
        else:
            features['person_levenshtein'] = 0.0
        
        # Helper function for harmonic mean
        def harmonic_mean(a, b):
            if a <= 0 or b <= 0:
                return 0.0
            return 2 * a * b / (a + b)
        
        # Add interaction features (simplified version)
        # Harmonic means
        person_cosine = features.get('person_cosine', 0.0)
        title_cosine = features.get('title_cosine', 0.0)
        provision_cosine = features.get('provision_cosine', 0.0)
        subjects_cosine = features.get('subjects_cosine', 0.0)
        composite_cosine = features.get('composite_cosine', 0.0)
        
        # Calculate harmonic means
        features['person_title_harmonic'] = harmonic_mean(person_cosine, title_cosine)
        features['person_provision_harmonic'] = harmonic_mean(person_cosine, provision_cosine)
        features['person_subjects_harmonic'] = harmonic_mean(person_cosine, subjects_cosine)
        features['title_subjects_harmonic'] = harmonic_mean(title_cosine, subjects_cosine)
        features['title_provision_harmonic'] = harmonic_mean(title_cosine, provision_cosine)
        features['provision_subjects_harmonic'] = harmonic_mean(provision_cosine, subjects_cosine)
        
        # Other interaction features
        features['person_subjects_product'] = person_cosine * subjects_cosine
        
        if subjects_cosine > 0:
            features['composite_subjects_ratio'] = composite_cosine / subjects_cosine
        else:
            features['composite_subjects_ratio'] = 0.0
        
        # Simplified birth/death check (using available strings)
        features['birth_death_year_match'] = 0.0
        
        # Convert features dictionary to numpy array
        feature_vector = np.array([features.get(name, 0.0) for name in feature_names])
        
        return feature_vector
    except Exception as e:
        # Capture the full traceback for better debugging
        error_msg = traceback.format_exc()
        # Print error but don't crash the worker
        print(f"Error extracting features: {str(e)}\n{error_msg}")
        return None