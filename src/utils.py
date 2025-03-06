"""
Utility functions for entity resolution
"""

import os
import time
import pickle
import logging
import mmap
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
import prometheus_client as prom
from tqdm import tqdm

# Configure logger
logger = logging.getLogger(__name__)

# Prometheus metrics
EMBEDDING_REQUESTS = prom.Counter('embedding_requests_total', 'Total number of embedding API requests')
QUERY_REQUESTS = prom.Counter('query_requests_total', 'Total number of vector queries')
PROCESSING_TIME = prom.Histogram('processing_time_seconds', 'Time spent in processing stages',
                              ['stage'])
MEMORY_USAGE = prom.Gauge('memory_usage_bytes', 'Memory usage by component',
                       ['component'])

class Timer:
    """Context manager for timing code execution"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time


def ensure_dir(path):
    """Ensure directory exists"""
    if isinstance(path, str):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(path, data):
    """Save checkpoint data to file"""
    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint to {path}: {e}")
        return False


def load_checkpoint(path):
    """Load checkpoint data from file"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading checkpoint from {path}: {e}")
        return None


def setup_logging(config):
    """Set up logging configuration"""
    log_level = getattr(logging, config['general']['log_level'])
    log_file = config['monitoring']['logging']['file']
    max_size = config['monitoring']['logging']['max_size_mb'] * 1024 * 1024
    backup_count = config['monitoring']['logging']['backup_count']
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    ensure_dir(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Setup rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info("Logging configured")


def setup_monitoring(config):
    """Set up monitoring metrics"""
    if config['monitoring']['prometheus']['enabled']:
        # Nothing to do here as Prometheus server is started in main.py
        logger.info("Prometheus monitoring enabled")


def chunk_dict(dictionary, chunk_size):
    """Split a dictionary into chunks of specified size"""
    items = list(dictionary.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])


def mmap_dict(filename, mode='r'):
    """Memory-map a dictionary file for efficient access"""
    try:
        with open(filename, mode) as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = pickle.loads(mm)
            return data
    except Exception as e:
        logger.error(f"Error memory-mapping file {filename}: {e}")
        return None


def compute_vector_similarity(vec1, vec2, metric='cosine'):
    """Compute similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    try:
        if metric == 'cosine':
            # Compute cosine similarity: 1 - cosine distance
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif metric == 'dot':
            # Compute dot product
            return np.dot(vec1, vec2)
        
        elif metric == 'euclidean':
            # Compute Euclidean similarity: 1 / (1 + Euclidean distance)
            dist = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + dist)
        
        else:
            logger.warning(f"Unknown similarity metric: {metric}")
            return 0.0
    
    except Exception as e:
        logger.error(f"Error computing vector similarity: {e}")
        return 0.0


def compute_levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings"""
    if not s1 and not s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    # Create a matrix
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize first row and column
    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j
    
    # Fill matrix
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost  # substitution
            )
    
    return dist[rows-1][cols-1]


def compute_levenshtein_similarity(s1, s2):
    """Compute Levenshtein similarity between two strings"""
    if not s1 and not s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = compute_levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def extract_birth_death_years(person_string):
    """Extract birth and death years from a person string"""
    import re
    
    # Pattern for common formats like "Smith, John, 1900-1980"
    pattern = r',\s*(\d{4})-(\d{4}|\?)'
    match = re.search(pattern, person_string)
    
    if match:
        birth_year = match.group(1)
        death_year = match.group(2)
        return birth_year, death_year
    
    return None, None


def harmonic_mean(a, b):
    """Compute harmonic mean of two values"""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 * a * b / (a + b)


def parallelize_dataframe(df, func, n_cores=4):
    """Apply a function to a dataframe in parallel"""
    import numpy as np
    from joblib import Parallel, delayed
    
    df_split = np.array_split(df, n_cores)
    results = Parallel(n_jobs=n_cores)(delayed(func)(df_chunk) for df_chunk in df_split)
    return pd.concat(results)
