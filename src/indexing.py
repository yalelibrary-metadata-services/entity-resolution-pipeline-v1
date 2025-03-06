"""
Indexing module for entity resolution
Handles Weaviate integration for vector indexing and retrieval
"""

import os
import uuid
import logging
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.util import generate_uuid5
from tqdm import tqdm

from .utils import Timer, chunk_dict

# Configure logger
logger = logging.getLogger(__name__)

class Indexer:
    """Weaviate indexer for entity resolution"""
    
    def __init__(self, config):
        """Initialize the indexer with configuration"""
        self.config = config
        self.weaviate_host = config['weaviate']['host']
        self.weaviate_port = config['weaviate']['port']
        self.weaviate_scheme = config['weaviate']['scheme']
        self.collection_name = config['weaviate']['collection_name']
        self.batch_size = config['weaviate']['batch_size']
        self.timeout = config['weaviate']['timeout']
        self.recreate_collection = config['indexing']['recreate_collection']
        self.upsert_mode = config['indexing']['upsert_mode']
        self.log_frequency = config['indexing']['log_frequency']
        
        # Weaviate vector index configuration
        self.ef = config['weaviate']['ef']
        self.ef_construction = config['weaviate']['ef_construction']
        self.max_connections = config['weaviate']['max_connections']
        self.distance_metric = config['weaviate']['distance_metric']
        
        # Connect to Weaviate
        self.client = self._connect_to_weaviate()
        
        # Processing state
        self.indexed = False
    
    def _connect_to_weaviate(self):
        """Connect to Weaviate instance"""
        try:
            logger.info(f"Connecting to Weaviate at {self.weaviate_scheme}://{self.weaviate_host}:{self.weaviate_port}")
            client = weaviate.connect_to_local(
                host=self.weaviate_host,
                port=self.weaviate_port,
                scheme=self.weaviate_scheme,
                timeout=self.timeout
            )
            
            # Check connection
            client.cluster.get_nodes_status()
            logger.info("Successfully connected to Weaviate")
            return client
        
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            raise
    
    def _create_collection(self):
        """Create collection in Weaviate"""
        try:
            # Check if collection exists
            collections = self.client.collections.list_all()
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            # Delete collection if it exists and recreate flag is set
            if collection_exists and self.recreate_collection:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.collections.delete(self.collection_name)
                collection_exists = False
            
            # Create collection if it doesn't exist
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                
                self.client.collections.create(
                    self.collection_name,
                    vectorizer_config=[
                        Configure.NamedVectors.none(
                            name="text_vector",
                            vector_index_config=Configure.VectorIndex.hnsw(
                                ef=self.ef,
                                max_connections=self.max_connections,
                                ef_construction=self.ef_construction,
                                distance_metric=VectorDistances.COSINE,
                            )
                        )
                    ],
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="hash", data_type=DataType.TEXT),
                        Property(name="frequency", data_type=DataType.NUMBER),
                        Property(name="field_type", data_type=DataType.TEXT),
                    ],
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
            
            return self.client.collections.get(self.collection_name)
        
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def index(self, unique_strings, field_mapping, embeddings):
        """Index unique strings and embeddings in Weaviate"""
        with Timer() as timer:
            logger.info("Starting Weaviate indexing")
            
            # Create or get collection
            collection = self._create_collection()
            
            # Prepare objects for indexing
            objects_to_index = []
            for hash_val, text in unique_strings.items():
                # Skip if no embedding is available
                if hash_val not in embeddings:
                    continue
                
                # Get embedding vector
                vector = embeddings[hash_val]
                
                # Get field types and frequencies
                field_types = field_mapping.get(hash_val, {})
                
                # Create an object for each field type
                for field_type, frequency in field_types.items():
                    objects_to_index.append({
                        "hash": hash_val,
                        "text": text,
                        "frequency": frequency,
                        "field_type": field_type,
                        "vector": vector,
                    })
            
            logger.info(f"Indexing {len(objects_to_index)} objects in Weaviate")
            
            # Index in batches
            num_batches = (len(objects_to_index) + self.batch_size - 1) // self.batch_size
            indexed_count = 0
            
            with collection.batch.dynamic() as batch:
                for i, obj in enumerate(tqdm(objects_to_index, desc="Indexing objects")):
                    # Generate a deterministic UUID from the hash and field type
                    obj_uuid = generate_uuid5(f"{obj['hash']}_{obj['field_type']}")
                    
                    # Add object to batch
                    if self.upsert_mode:
                        batch.add_object(
                            properties={
                                "text": obj["text"],
                                "hash": obj["hash"],
                                "frequency": obj["frequency"],
                                "field_type": obj["field_type"],
                            },
                            vector={"text_vector": obj["vector"].tolist()},
                            uuid=obj_uuid
                        )
                    else:
                        batch.add_object(
                            properties={
                                "text": obj["text"],
                                "hash": obj["hash"],
                                "frequency": obj["frequency"],
                                "field_type": obj["field_type"],
                            },
                            vector={"text_vector": obj["vector"].tolist()},
                        )
                    
                    indexed_count += 1
                    
                    # Log progress
                    if (i + 1) % self.log_frequency == 0 or i == len(objects_to_index) - 1:
                        logger.info(f"Indexed {i + 1}/{len(objects_to_index)} objects")
            
            logger.info(f"Successfully indexed {indexed_count} objects in Weaviate")
            self.indexed = True
        
        logger.info(f"Indexing time: {timer.elapsed:.2f} seconds")
        return self
    
    def get_collection(self):
        """Get the Weaviate collection"""
        return self.client.collections.get(self.collection_name)
    
    def is_indexed(self):
        """Check if data has been indexed"""
        return self.indexed
    
    def get_state(self):
        """Get the current state for checkpointing"""
        return {
            'indexed': self.indexed
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        self.indexed = state['indexed']
        logger.info(f"Loaded indexer state: indexed={self.indexed}")
