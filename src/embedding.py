"""
Embedding module for entity resolution
Handles vector embedding generation using OpenAI's API
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from .utils import Timer, chunk_dict, ensure_dir

# Configure logger
logger = logging.getLogger(__name__)

class Embedder:
    """Embedder for entity resolution"""
    
    def __init__(self, config):
        """Initialize the embedder with configuration"""
        self.config = config
        self.embedding_model = config['openai']['embedding_model']
        self.embedding_dim = config['openai']['embedding_dim']
        self.rate_limit_tpm = config['openai']['rate_limit_tpm']
        self.rate_limit_rpm = config['openai']['rate_limit_rpm']
        self.batch_size = config['openai']['batch_size']
        self.retry_attempts = config['openai']['retry_attempts']
        self.retry_delay = config['openai']['retry_delay']
        self.parallel_requests = config['embedding']['parallel_requests']
        
        # Cache settings
        self.cache_enabled = config['embedding']['cache_enabled']
        if self.cache_enabled:
            self.cache_dir = Path(config['embedding']['cache_dir'])
            ensure_dir(self.cache_dir)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get(config['openai']['api_key_env']))
        
        # Initialize data structures
        self.embeddings = {}  # hash -> embedding vector
        
        # Processing state
        self.processed = False
    
    def embed(self, unique_strings):
        """Generate embeddings for unique strings"""
        with Timer() as timer:
            logger.info(f"Starting embedding generation for {len(unique_strings)} unique strings")
            
            # Check cache first if enabled
            if self.cache_enabled:
                self._load_from_cache()
            
            # Identify strings that need embedding
            strings_to_embed = {
                hash_val: text for hash_val, text in unique_strings.items()
                if hash_val not in self.embeddings
            }
            
            logger.info(f"Generating embeddings for {len(strings_to_embed)} strings")
            
            if strings_to_embed:
                # Process in batches
                batches = list(chunk_dict(strings_to_embed, self.batch_size))
                
                with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
                    futures = {}
                    
                    for batch_idx, batch in enumerate(batches):
                        future = executor.submit(self._embed_batch, batch)
                        futures[future] = batch_idx
                    
                    # Process results as they complete
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding batches"):
                        batch_idx = futures[future]
                        try:
                            batch_embeddings = future.result()
                            self.embeddings.update(batch_embeddings)
                            
                            # Cache results if enabled
                            if self.cache_enabled:
                                self._cache_batch(batch_embeddings)
                        
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx}: {e}")
            
            logger.info(f"Embedding generation completed: {len(self.embeddings)} vectors")
            self.processed = True
        
        logger.info(f"Embedding time: {timer.elapsed:.2f} seconds")
        return self
    
    def _embed_batch(self, batch):
        """Generate embeddings for a batch of strings"""
        batch_embeddings = {}
        texts = list(batch.values())
        hash_values = list(batch.keys())
        
        # Implement retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                # Process response
                for i, embedding_data in enumerate(response.data):
                    hash_val = hash_values[i]
                    batch_embeddings[hash_val] = np.array(embedding_data.embedding)
                
                # Successful response, break retry loop
                break
            
            except Exception as e:
                logger.warning(f"Embedding API error (attempt {attempt+1}/{self.retry_attempts}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Implement exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    # Last attempt failed, raise exception
                    raise
        
        return batch_embeddings
    
    def _load_from_cache(self):
        """Load embeddings from cache"""
        if not self.cache_dir.exists():
            return
        
        cache_files = list(self.cache_dir.glob("*.npy"))
        loaded_count = 0
        
        for cache_file in cache_files:
            hash_val = cache_file.stem
            try:
                embedding = np.load(cache_file)
                self.embeddings[hash_val] = embedding
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Error loading cached embedding {cache_file}: {e}")
        
        logger.info(f"Loaded {loaded_count} embeddings from cache")
    
    def _cache_batch(self, batch_embeddings):
        """Cache a batch of embeddings"""
        for hash_val, embedding in batch_embeddings.items():
            cache_file = self.cache_dir / f"{hash_val}.npy"
            try:
                np.save(cache_file, embedding)
            except Exception as e:
                logger.warning(f"Error caching embedding {hash_val}: {e}")
    
    def get_embedding(self, hash_val):
        """Get embedding vector for a hash"""
        return self.embeddings.get(hash_val)
    
    def get_embeddings(self):
        """Get all embeddings"""
        return self.embeddings
    
    def is_processed(self):
        """Check if embeddings have been generated"""
        return self.processed
    
    def get_state(self):
        """Get the current state for checkpointing"""
        # Note: we don't include the full embeddings in the state
        # as they are stored in the cache
        return {
            'embedding_hashes': list(self.embeddings.keys()),
            'processed': self.processed
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        self.processed = state['processed']
        
        # Load embeddings from cache based on hashes
        if self.cache_enabled:
            for hash_val in state['embedding_hashes']:
                cache_file = self.cache_dir / f"{hash_val}.npy"
                if cache_file.exists():
                    try:
                        self.embeddings[hash_val] = np.load(cache_file)
                    except Exception as e:
                        logger.warning(f"Error loading cached embedding {hash_val}: {e}")
        
        logger.info(f"Loaded embedder state: {len(self.embeddings)} embeddings")
