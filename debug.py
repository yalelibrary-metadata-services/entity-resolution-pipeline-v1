# Run this in your environment to debug the specific record pair

import logging
import numpy as np
import pandas as pd
import sys
import yaml

# Configure logging to see everything
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_record_pair(entity1_id='15505854#Agent100-11', entity2_id='1271161#Agent100-13', config_path='config.optimized.yml'):
    """Debug a specific record pair to identify why it's not being matched correctly"""
    print(f"\n=== DEBUGGING RECORD PAIR {entity1_id} - {entity2_id} ===")
    
    # Step 1: Load configuration
    print("\nLoading configuration...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Successfully loaded configuration from {config_path}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Step 2: Create necessary pipeline components
    print("\nInitializing pipeline components...")
    
    # Add path to src directory if needed
    if 'src' not in sys.path:
        sys.path.append('src')
    
    # Import pipeline components
    try:
        from src.preprocessing import Preprocessor
        from src.indexing import Indexer
        from src.optimized_query import OptimizedQueryEngine
        from src.features import FeatureExtractor
        from src.classification import Classifier
        
        preprocessor = Preprocessor(config)
        indexer = Indexer(config)
        query_engine = OptimizedQueryEngine(config)
        feature_extractor = FeatureExtractor(config)
        classifier = Classifier(config)
        
        print("Successfully initialized pipeline components")
    except Exception as e:
        print(f"Error initializing pipeline components: {e}")
        return
    
    # Step 3: Load preprocessor data
    print("\nLoading preprocessor data...")
    try:
        from src.utils import load_checkpoint
        
        checkpoint_path = f"{config['general']['checkpoint_dir']}/preprocessing.pkl"
        preprocessor_data = load_checkpoint(checkpoint_path)
        
        if preprocessor_data:
            preprocessor.load_state(preprocessor_data)
            print("Successfully loaded preprocessor data")
        else:
            print("Failed to load preprocessor data")
            return
    except Exception as e:
        print(f"Error loading preprocessor data: {e}")
        return
    
    # Step 4: Set Weaviate collection
    print("\nSetting Weaviate collection...")
    try:
        collection = indexer.get_collection()
        query_engine.set_collection(collection)
        print("Successfully set Weaviate collection")
    except Exception as e:
        print(f"Error setting Weaviate collection: {e}")
        return
    
    # Step 5: Get records
    print("\nFetching records...")
    try:
        record1 = preprocessor.get_record(entity1_id)
        record2 = preprocessor.get_record(entity2_id)
        
        if not record1 or not record2:
            print(f"Error: Could not find records for entities")
            return
        
        print(f"Record 1 ({entity1_id}): {record1}")
        print(f"Record 2 ({entity2_id}): {record2}")
    except Exception as e:
        print(f"Error fetching records: {e}")
        return
    
    # Step 6: Extract features
    print("\nExtracting features...")
    try:
        feature_vector = feature_extractor.extract_features(record1, record2, query_engine)
        feature_names = feature_extractor.get_feature_names()
        
        print("Feature vector:")
        for i, name in enumerate(feature_names):
            if i < len(feature_vector):
                print(f"  {name}: {feature_vector[i]:.6f}")
        
        # Identify key features
        composite_cosine_idx = -1
        person_cosine_idx = -1
        
        for i, name in enumerate(feature_names):
            if name == 'composite_cosine':
                composite_cosine_idx = i
            elif name == 'person_cosine':
                person_cosine_idx = i
        
        composite_cosine = feature_vector[composite_cosine_idx] if composite_cosine_idx >= 0 else None
        person_cosine = feature_vector[person_cosine_idx] if person_cosine_idx >= 0 else None
        
        if composite_cosine is not None:
            print(f"\nComposite Cosine Similarity: {composite_cosine:.6f}")
            
            # Check against threshold
            threshold = config['features'].get('composite_cosine_prefilter', {}).get('threshold', 0.65)
            if composite_cosine >= threshold:
                print(f"SHOULD MATCH: Composite similarity {composite_cosine:.6f} >= threshold {threshold}")
            else:
                print(f"Would not match: Composite similarity {composite_cosine:.6f} < threshold {threshold}")
        else:
            print("Could not find composite_cosine in feature vector")
            
        if person_cosine is not None:
            print(f"\nPerson Cosine Similarity: {person_cosine:.6f}")
            
            # Check against threshold
            threshold = config['features'].get('person_cosine_prefilter', {}).get('threshold', 0.70)
            if person_cosine >= threshold:
                print(f"SHOULD PASS: Person similarity {person_cosine:.6f} >= threshold {threshold}")
            else:
                print(f"Would be filtered out: Person similarity {person_cosine:.6f} < threshold {threshold}")
        else:
            print("Could not find person_cosine in feature vector")
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
    
    # Step 7: Check classification
    print("\nTesting classification...")
    try:
        classifier_checkpoint = f"{config['general']['checkpoint_dir']}/training.pkl"
        classifier_data = load_checkpoint(classifier_checkpoint)
        
        if classifier_data:
            classifier.load_state(classifier_data)
            print("Successfully loaded classifier data")
            
            # Predict match probability
            probability = classifier.predict(feature_vector)
            print(f"Classification probability: {probability:.6f}")
            
            # Check against threshold
            threshold = classifier.match_threshold
            if probability >= threshold:
                print(f"SHOULD MATCH: Probability {probability:.6f} >= threshold {threshold}")
            else:
                print(f"Would not match: Probability {probability:.6f} < threshold {threshold}")
        else:
            print("Failed to load classifier data")
    except Exception as e:
        print(f"Error testing classification: {e}")
    
    # Step 8: Check prefilter settings
    print("\nChecking prefilter configuration...")
    try:
        composite_prefilter = config['features'].get('composite_cosine_prefilter', {})
        person_prefilter = config['features'].get('person_cosine_prefilter', {})
        
        print(f"Composite cosine prefilter:")
        print(f"  Enabled: {composite_prefilter.get('enabled', False)}")
        print(f"  Threshold: {composite_prefilter.get('threshold', 0.65)}")
        print(f"  Override threshold: {composite_prefilter.get('override_threshold', 0.0)}")
        
        print(f"Person cosine prefilter:")
        print(f"  Enabled: {person_prefilter.get('enabled', False)}")
        print(f"  Threshold: {person_prefilter.get('threshold', 0.70)}")
        
        # Check if composite prefilter is properly enabled
        if not composite_prefilter.get('enabled', False):
            print("WARNING: Composite cosine prefilter is disabled!")
        
        # Check if override threshold is too low
        if composite_prefilter.get('override_threshold', 0.0) < 0.9:
            print(f"WARNING: Override threshold is too low: {composite_prefilter.get('override_threshold', 0.0)}")
    except Exception as e:
        print(f"Error checking prefilter settings: {e}")
    
    # Final conclusion
    print("\n=== DEBUG SUMMARY ===")
    print(f"Record pair: {entity1_id} - {entity2_id}")
    
    if composite_cosine is not None and composite_cosine >= 0.65:
        print(f"This pair SHOULD MATCH based on composite similarity: {composite_cosine:.6f}")
        print("Recommended fixes:")
        print("1. Ensure composite_cosine_prefilter is enabled in the config")
        print("2. Set override_threshold to at least 0.9")
        print("3. Verify that the prefilter code is correctly checking for feature names")
    else:
        print("Further investigation needed")
    
    print("=== END DEBUG ===\n")

if __name__ == "__main__":
    debug_record_pair()