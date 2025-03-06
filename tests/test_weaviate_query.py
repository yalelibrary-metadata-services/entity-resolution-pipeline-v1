#!/usr/bin/env python
"""
Weaviate v4 Query Test Script

This standalone script tests Weaviate API v4 queries to help debug filter and result handling.
It demonstrates the correct way to use filters and access query results in Weaviate client v4.
"""

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
import numpy as np
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_collection(client, collection_name="TestCollection"):
    """Create a test collection if it doesn't exist"""
    try:
        # Check if collection exists
        collections = client.collections.list_all()
        if any(c == collection_name for c in collections):
            logger.info(f"Collection {collection_name} already exists")
            return client.collections.get(collection_name)

        # Create collection
        logger.info(f"Creating collection: {collection_name}")
        collection = client.collections.create(
            collection_name,
            vectorizer_config=[
                Configure.NamedVectors.none(
                    name="text_vector",
                    vector_index_config=Configure.VectorIndex.hnsw(
                        ef=128,
                        max_connections=64,
                        ef_construction=128,
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
        return collection
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise

def add_test_data(collection):
    """Add sample data to the collection"""
    try:
        # Create sample data
        test_data = [
            {
                "text": "John Smith",
                "hash": "hash1",
                "frequency": 5,
                "field_type": "person",
                "vector": np.random.rand(1536).tolist()
            },
            {
                "text": "Jane Doe",
                "hash": "hash2",
                "frequency": 3,
                "field_type": "person",
                "vector": np.random.rand(1536).tolist()
            },
            {
                "text": "History Book",
                "hash": "hash3",
                "frequency": 2,
                "field_type": "title",
                "vector": np.random.rand(1536).tolist()
            },
            {
                "text": "New York, 2023",
                "hash": "hash4",
                "frequency": 1,
                "field_type": "provision",
                "vector": np.random.rand(1536).tolist()
            }
        ]

        # Insert data
        logger.info("Adding test data")
        with collection.batch.dynamic() as batch:
            for obj in test_data:
                batch.add_object(
                    properties={
                        "text": obj["text"],
                        "hash": obj["hash"],
                        "frequency": obj["frequency"],
                        "field_type": obj["field_type"],
                    },
                    vector={"text_vector": obj["vector"]},
                )

        logger.info("Test data added successfully")
    except Exception as e:
        logger.error(f"Error adding test data: {e}")
        raise

def test_simple_filter(collection):
    """Test a simple filter query"""
    try:
        logger.info("Testing simple filter query")
        
        # Simple filter by hash
        hash_filter = Filter.by_property("hash").equal("hash1")
        
        result = collection.query.fetch_objects(
            filters=hash_filter,
            limit=10,
            include_vector=True
        )
        
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result structure: {dir(result)}")
        
        # Print results
        if hasattr(result, 'objects'):
            # Old API style
            logger.info(f"Found {len(result.objects)} objects (accessing via .objects)")
            for obj in result.objects:
                logger.info(f"Object: {obj.properties}")
        else:
            # New API style
            logger.info(f"Found {len(result)} objects (direct list access)")
            for obj in result:
                logger.info(f"Object: {obj.properties}")
                
        return result
    except Exception as e:
        logger.error(f"Error running simple filter query: {e}")
        raise

def test_combined_filter(collection):
    """Test a combined filter query"""
    try:
        logger.info("Testing combined filter query")
        
        # Try different ways to combine filters
        hash_filter = Filter.by_property("hash").equal("hash1")
        field_filter = Filter.by_property("field_type").equal("person")
        
        # Method 1: Using & operator (recommended for v4)
        combined_filter = hash_filter & field_filter
        
        # Method 2: Using and_operator method (older versions)
        try:
            alt_combined_filter = Filter.and_operator([hash_filter, field_filter])
            logger.info("and_operator method is available")
        except AttributeError:
            logger.info("and_operator method is not available in this version")
            alt_combined_filter = None
        
        # Execute query with combined filter
        result = collection.query.fetch_objects(
            filters=combined_filter,
            limit=10,
            include_vector=True
        )
        
        # Print results
        if hasattr(result, 'objects'):
            logger.info(f"Found {len(result.objects)} objects (via .objects)")
            for obj in result.objects:
                logger.info(f"Object: {obj.properties}")
        else:
            logger.info(f"Found {len(result)} objects (direct list)")
            for obj in result:
                logger.info(f"Object: {obj.properties}")
                logger.info(f"Vector available: {obj.vector is not None}")
                
        return result
    except Exception as e:
        logger.error(f"Error running combined filter query: {e}")
        raise

def test_near_vector_query(collection):
    """Test a near vector query"""
    try:
        logger.info("Testing near vector query")
        
        # Create a random query vector
        query_vector = np.random.rand(1536).tolist()
        
        # Create a field filter
        field_filter = Filter.by_property("field_type").equal("person")
        
        # Execute near vector query
        result = collection.query.near_vector(
            near_vector=query_vector,
            limit=10,
            return_metadata=MetadataQuery(distance=True),
            include_vector=True,
            filters=field_filter
        )
        
        # Print results
        if hasattr(result, 'objects'):
            logger.info(f"Found {len(result.objects)} objects (via .objects)")
            for obj in result.objects:
                distance = obj.metadata.distance if hasattr(obj, 'metadata') else "N/A"
                logger.info(f"Object: {obj.properties}, Distance: {distance}")
        else:
            logger.info(f"Found {len(result)} objects (direct list)")
            for obj in result:
                distance = obj.metadata.distance if hasattr(obj, 'metadata') else "N/A"
                logger.info(f"Object: {obj.properties}, Distance: {distance}")
                
        return result
    except Exception as e:
        logger.error(f"Error running near vector query: {e}")
        raise

def print_result_structure(result):
    """Print detailed structure of a result object"""
    logger.info("Result structure analysis:")
    
    if result is None:
        logger.info("Result is None")
        return
        
    logger.info(f"Result type: {type(result)}")
    
    if hasattr(result, 'objects'):
        # Old style API
        logger.info("Result has 'objects' attribute - old API style")
        if len(result.objects) > 0:
            obj = result.objects[0]
            logger.info(f"First object type: {type(obj)}")
            logger.info(f"First object attributes: {dir(obj)}")
            logger.info(f"First object properties: {obj.properties}")
            if hasattr(obj, 'metadata'):
                logger.info(f"Metadata attributes: {dir(obj.metadata)}")
    else:
        # New style API
        logger.info("Result is a direct list - new API style")
        if len(result) > 0:
            obj = result[0]
            logger.info(f"First object type: {type(obj)}")
            logger.info(f"First object attributes: {dir(obj)}")
            logger.info(f"First object properties: {obj.properties}")
            if hasattr(obj, 'metadata'):
                logger.info(f"Metadata attributes: {dir(obj.metadata)}")

def main():
    """Main function to run tests"""
    try:
        # Connect to Weaviate
        logger.info("Connecting to Weaviate")
        client = weaviate.connect_to_local(
            # host="localhost",
            # port=8080,
            # scheme="http",
            # timeout=120
        )
        
        # Verify connection
        try:
            # Try client.get_meta() for simple connection check
            client.get_meta()
            logger.info("Connected to Weaviate successfully")
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            logger.info("Trying alternative connection checks...")
            try:
                client.cluster.get_nodes_status_overview()
                logger.info("get_nodes_status_overview succeeded")
            except Exception as e1:
                logger.error(f"get_nodes_status_overview failed: {e1}")
            
            try:
                client.cluster.get_nodes_status()
                logger.info("get_nodes_status succeeded")
            except Exception as e2:
                logger.error(f"get_nodes_status failed: {e2}")
            
            return
        
        # Create test collection
        collection = create_test_collection(client)
        
        # Add test data
        add_test_data(collection)
        
        # Test queries
        simple_result = test_simple_filter(collection)
        print_result_structure(simple_result)
        
        combined_result = test_combined_filter(collection)
        
        near_vector_result = test_near_vector_query(collection)
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("Weaviate connection closed")

if __name__ == "__main__":
    main()