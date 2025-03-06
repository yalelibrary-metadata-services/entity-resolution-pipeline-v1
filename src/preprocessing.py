"""
Preprocessing module for entity resolution
Handles data loading, deduplication, and preparation
"""

import os
import csv
import hashlib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm

from .utils import Timer, mmap_dict

# Configure logger
logger = logging.getLogger(__name__)

class Preprocessor:
    """Preprocessor for entity resolution"""
    
    def __init__(self, config):
        """Initialize the preprocessor with configuration"""
        self.config = config
        self.input_dir = Path(config['dataset']['input_dir'])
        self.hash_algorithm = config['preprocessing']['hash_algorithm']
        self.min_string_frequency = config['preprocessing']['min_string_frequency']
        self.normalize_case = config['preprocessing']['normalize_case']
        self.batch_size = config['general']['batch_size']
        self.num_workers = min(
            config['general']['num_workers'], 
            mp.cpu_count()
        )
        self.mmap_enabled = config['preprocessing']['mmap_enabled']
        
        # Fields configuration
        self.fields_to_embed = config['fields']['embed']
        self.fields_to_impute = config['fields']['impute']
        self.required_fields = config['fields']['required']
        self.id_field = config['fields']['id_field']
        
        # Mode configuration
        self.mode = config['general']['mode']
        if self.mode == 'development':
            self.max_files = config['development']['max_files']
            self.max_records = config['development']['max_records']
        
        # Initialize data structures
        self.unique_strings = {}  # hash -> string
        self.string_counts = {}   # hash -> count
        self.record_field_hashes = {}  # record_id -> {field -> hash}
        self.field_hash_mapping = {}   # hash -> {field_type -> count}
        self.all_person_ids = set()
        
        # Processing state
        self.processed = False
    
    def process(self):
        """Process input files and deduplicate data"""
        with Timer() as timer:
            logger.info("Starting preprocessing")
            
            # Get list of CSV files
            csv_files = sorted(list(self.input_dir.glob('*.csv')))
            
            if self.mode == 'development':
                # Limit number of files in development mode
                csv_files = csv_files[:self.max_files]
                logger.info(f"Development mode: processing {len(csv_files)} files")
            
            logger.info(f"Found {len(csv_files)} CSV files in {self.input_dir}")
            
            # Process each file
            for file_idx, csv_file in enumerate(tqdm(csv_files, desc="Processing files")):
                self._process_file(csv_file)
            
            # Filter infrequent strings if needed
            if self.min_string_frequency > 1:
                self._filter_infrequent_strings()
            
            logger.info(f"Preprocessing completed: {len(self.unique_strings)} unique strings")
            logger.info(f"Total records: {len(self.record_field_hashes)}")
            
            self.processed = True
        
        logger.info(f"Preprocessing time: {timer.elapsed:.2f} seconds")
        return self
    
    def _process_file(self, file_path):
        """Process a single CSV file"""
        logger.debug(f"Processing file: {file_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Limit number of records in development mode
            if self.mode == 'development':
                df = df.head(self.max_records)
            
            # Process each record in batches
            total_records = len(df)
            batch_count = (total_records + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(batch_count):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_records)
                
                batch_df = df.iloc[start_idx:end_idx]
                self._process_batch(batch_df)
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
    
    def _process_batch(self, batch_df):
        """Process a batch of records"""
        # Process each record in the batch
        for _, record in batch_df.iterrows():
            self._process_record(record)
    
    def _process_record(self, record):
        """Process a single record"""
        record_id = record[self.id_field]
        self.all_person_ids.add(record_id)
        
        # Initialize record field hashes
        self.record_field_hashes[record_id] = {}
        
        # Process each field
        for field in self.fields_to_embed:
            if field in record and pd.notna(record[field]):
                value = str(record[field])
                
                # Normalize case if configured
                if self.normalize_case:
                    value = value.lower()
                
                # Compute hash
                value_hash = self._compute_hash(value)
                
                # Store unique string
                if value_hash not in self.unique_strings:
                    self.unique_strings[value_hash] = value
                    self.string_counts[value_hash] = 0
                
                # Increment string count
                self.string_counts[value_hash] += 1
                
                # Store record field hash
                self.record_field_hashes[record_id][field] = value_hash
                
                # Update field hash mapping
                if value_hash not in self.field_hash_mapping:
                    self.field_hash_mapping[value_hash] = defaultdict(int)
                self.field_hash_mapping[value_hash][field] += 1
            else:
                # Handle null values
                self.record_field_hashes[record_id][field] = "NULL"
    
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
    
    def _filter_infrequent_strings(self):
        """Filter out strings that appear less than min_frequency times"""
        initial_count = len(self.unique_strings)
        
        # Identify infrequent strings
        infrequent_hashes = [
            h for h, count in self.string_counts.items() 
            if count < self.min_string_frequency
        ]
        
        # Remove infrequent strings
        for h in infrequent_hashes:
            del self.unique_strings[h]
            del self.string_counts[h]
            # Note: we keep the record_field_hashes intact with references to dropped strings
        
        logger.info(f"Filtered out {initial_count - len(self.unique_strings)} infrequent strings")
    
    def get_unique_strings(self):
        """Get dictionary of unique strings keyed by hash"""
        return self.unique_strings
    
    def get_string_counts(self):
        """Get string frequency counts keyed by hash"""
        return self.string_counts
    
    def get_record_field_hashes(self):
        """Get record field hashes"""
        return self.record_field_hashes
    
    def get_field_mapping(self):
        """Get field mapping dictionary"""
        return self.field_hash_mapping
    
    def get_all_person_ids(self):
        """Get all person IDs"""
        return list(self.all_person_ids)
    
    def get_record(self, record_id):
        """Get a record by ID"""
        if record_id not in self.record_field_hashes:
            return None
        return self.record_field_hashes[record_id]
    
    def load_ground_truth(self, ground_truth_file):
        """Load ground truth data for training"""
        logger.info(f"Loading ground truth data from {ground_truth_file}")
        
        try:
            df = pd.read_csv(ground_truth_file)
            
            # Extract pairs and labels
            pairs = []
            labels = []
            
            for _, row in df.iterrows():
                left_id = row['left']
                right_id = row['right']
                match = row['match'] if 'match' in row else (row['label'] == 'true')
                
                # Convert match to boolean
                if isinstance(match, str):
                    match = match.lower() == 'true'
                
                pairs.append((left_id, right_id))
                labels.append(match)
            
            logger.info(f"Loaded {len(pairs)} labeled pairs ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
            return pairs, labels
        
        except Exception as e:
            logger.error(f"Error loading ground truth data: {e}")
            raise
    
    def is_processed(self):
        """Check if data has been processed"""
        return self.processed
    
    def get_state(self):
        """Get the current state for checkpointing"""
        return {
            'unique_strings': self.unique_strings,
            'string_counts': self.string_counts,
            'record_field_hashes': self.record_field_hashes,
            'field_hash_mapping': self.field_hash_mapping,
            'all_person_ids': self.all_person_ids,
            'processed': self.processed
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        self.unique_strings = state['unique_strings']
        self.string_counts = state['string_counts']
        self.record_field_hashes = state['record_field_hashes']
        self.field_hash_mapping = state['field_hash_mapping']
        self.all_person_ids = state['all_person_ids']
        self.processed = state['processed']
        
        logger.info(f"Loaded preprocessor state: {len(self.unique_strings)} unique strings")
