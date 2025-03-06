# Entity Resolution System for Yale University Library Catalog

## Overview

The Entity Resolution System is designed to solve the complex problem of identifying when different catalog entries in the Yale University Library refer to the same real-world person. This system represents a sophisticated approach to entity resolution in bibliographic data, where records primarily describe works rather than individuals, requiring contextual inference and advanced similarity analysis techniques.

## Business Case and Challenges

### Use Case

Library catalogs contain millions of entries with personal names appearing across multiple records. These references often vary in format, completeness, and context:

- The same person may appear with different name formats (e.g., "Smith, John" vs. "John Smith")
- Names may include varying levels of detail (middle names, initials, suffixes, birth/death years)
- Non-Latin scripts and transliterations introduce additional variability
- Historical figures may have works attributed to them spanning centuries

Without proper entity resolution, users searching for works by a specific person may miss relevant entries, duplicate entries may bloat the catalog, and relationships between works by the same creator remain obscured.

### Key Challenges

1. **Scale and Performance**: The system must process up to 50 million unique strings efficiently.

2. **Linguistic Complexity**: The solution must handle multilingual data without bias toward any specific language or script.

3. **Temporal Reasoning**: Publication dates may not align with an author's lifetime, requiring sophisticated temporal reasoning.

4. **Data Incompleteness**: Many catalog entries have missing fields that require intelligent imputation.

5. **Disambiguation**: Common names require additional contextual information to distinguish between different individuals.

6. **Precision Requirements**: In a library context, false positives (incorrectly merging different entities) are more problematic than false negatives.

## System Architecture

Our implementation uses a modular pipeline architecture that combines vector embeddings with machine learning classification:

### Data Preprocessing

- Extracts and deduplicates fields from the dataset
- Maintains efficient hash-based data structures
- Tracks frequency of duplicate strings
- Creates mappings between unique strings, embeddings, and entities

### Vector Embedding

- Generates 1,536-dimensional vector embeddings using OpenAI's text-embedding-3-small model
- Implements batch processing with rate limiting to respect API constraints
- Caches embeddings for reuse
- Processes only unique strings to minimize API calls

### Weaviate Integration

- Indexes embeddings in Weaviate for efficient similarity search
- Configures optimal HNSW parameters for approximate nearest neighbor search
- Implements named vectors for multi-field querying
- Provides efficient upsert operations to avoid duplicates

### Vector-Based Imputation

- Implements a "hot deck" approach for missing values
- Uses the composite field vector to find similar records
- Averages vectors from nearest neighbors
- Dynamically updates the index with imputed values

### Feature Engineering

- Constructs feature vectors that capture similarity across multiple dimensions
- Implements both direct similarities (cosine, Levenshtein) and interaction features
- Creates specialized features for capturing non-linear relationships
- Handles birth/death years as strong signals for matching

### Classification

- Trains a logistic regression classifier using gradient descent
- Optimizes with regularization and early stopping
- Processes candidate pairs in batches and in parallel
- Balances precision and recall based on confidence levels

### Entity Clustering

- Uses graph-based community detection algorithms
- Creates weighted edges based on match confidence
- Filters clusters by size to eliminate noise
- Serializes the complete identity graph as JSON-Lines

## Technical Approach

### Vector Representation

The system generates 1,536-dimensional vector embeddings for five key fields:
- `composite`: Combined text from all fields
- `person`: Extracted personal name
- `title`: Title of the work
- `provision`: Publication details
- `subjects`: Subject classifications

These vectors capture semantic relationships between entities, enabling sophisticated matching beyond simple string comparisons.

### Feature Engineering

The classification model uses a rich feature set combining:

1. **Vector Similarities**: Cosine similarity between embedded fields
2. **String Similarities**: Levenshtein distance for name comparisons
3. **Interaction Features**: 
   - Harmonic means between field similarities
   - Field-weighted combinations
   - Cross-field ratios and products
4. **Special Signals**: Birth/death year matching

### Null Value Imputation

For missing values in `provision` and `subjects` fields, we implement a vector-based hot deck approach:
1. Use the `composite` field vector to query similar records
2. Retrieve the top 10 nearest neighbors
3. Compute a weighted average vector
4. Find the most similar existing vector for the null field

### Pipeline Execution

The system executes in clear stages, each with checkpointing:
1. **Preprocessing**: Extract and deduplicate fields
2. **Embedding**: Generate vectors for unique strings
3. **Indexing**: Store embeddings in Weaviate
4. **Training**: Train classifier on labeled data
5. **Classification**: Apply classifier to full dataset
6. **Clustering**: Group matched entities
7. **Reporting**: Generate analysis and visualizations

## Implementation Details

### Technology Stack

- **Vector Database**: Weaviate 1.24+ (HNSW algorithm)
- **Embedding Model**: OpenAI text-embedding-3-small
- **Machine Learning**: Custom logistic regression with gradient descent
- **Parallelization**: Multi-processing for batch operations
- **Monitoring**: Prometheus for performance metrics
- **Visualization**: Matplotlib and Seaborn for analysis

### Scalability Features

- **Memory-Mapped Files**: Efficient checkpointing for large datasets
- **Batch Processing**: Optimized for API rate limits and memory constraints
- **Configurable Resources**: Adapts to available hardware
- **Parallel Execution**: Leverages multi-core processing
- **Development Mode**: Processes subset for rapid iteration

### Performance Considerations

- **Runtime Resources**:
  - Development: 8 cores, 32GB RAM
  - Production: 64 cores, 256GB RAM
- **API Constraints**:
  - 5,000,000 tokens per minute
  - 10,000 requests per minute
  - 500,000,000 tokens per day
- **Optimized Storage**: Hash-based lookups minimize memory usage
- **Efficient Querying**: Blocking strategy reduces comparison space

## Usage

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- 32GB RAM for development, 256GB for production

### Installation

1. Clone the repository
   ```
   git clone [repository-url]
   cd entity-resolution
   ```

2. Run the setup script
   ```
   ./setup.sh
   ```

3. Start Weaviate using Docker Compose
   ```
   docker-compose up -d
   ```

4. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```

### Running the Pipeline

Execute the complete pipeline:
```
python main.py --stage all
```

Run specific stages:
```
python main.py --stage preprocessing
python main.py --stage embedding
python main.py --stage indexing
python main.py --stage training
python main.py --stage classification
python main.py --stage clustering
python main.py --stage reporting
```

For development with a smaller dataset:
```
python main.py --stage all --mode development
```

## Results and Evaluation

The system provides comprehensive evaluation metrics:

- **Classification Performance**: Precision, recall, and F1 score
- **Feature Importance**: Analysis of most predictive features
- **Cluster Analysis**: Distribution of cluster sizes and characteristics
- **Execution Performance**: Runtime analysis for each pipeline stage

## Future Developments

Several areas offer potential for further enhancement:

1. **Advanced Neural Models**: Investigate transformer-based models for classification
2. **Unsupervised Learning**: Explore clustering approaches that require less labeled data
3. **Active Learning**: Implement feedback loops to improve classifier performance
4. **Specialized Embeddings**: Train domain-specific embeddings for bibliographic data
5. **Cross-lingual Matching**: Enhance performance for transliterated names
6. **Incremental Processing**: Add support for updating the index with new records
7. **User Interface**: Develop an interface for reviewing and correcting matches

## Project Structure

```
entity-resolution/
   ├── README.md               # Project documentation
   ├── config.yml              # Configuration parameters
   ├── docker-compose.yml      # Docker Compose for Weaviate
   ├── prometheus.yml          # Prometheus monitoring configuration
   ├── requirements.txt        # Python dependencies
   ├── main.py                 # Entry point script
   ├── setup.sh                # Setup script
   ├── src/                    # Source code
   │   ├── preprocessing.py    # Data preprocessing
   │   ├── embedding.py        # Vector embedding
   │   ├── indexing.py         # Weaviate integration
   │   ├── imputation.py       # Null value imputation
   │   ├── querying.py         # Querying and match candidate retrieval
   │   ├── features.py         # Feature engineering
   │   ├── classification.py   # Classifier training/evaluation
   │   ├── clustering.py       # Entity clustering
   │   ├── pipeline.py         # Pipeline orchestration
   │   ├── analysis.py         # Analysis of pipeline processes and results
   │   ├── reporting.py        # Reporting and visualization of pipeline results
   │   └── utils.py            # Utility functions
   ├── notebooks/              # Analysis notebooks
   │   ├── evaluation.ipynb    # Results evaluation
   │   └── exploration.ipynb   # Data exploration
   └── tests/                  # Testing scripts
      └── verify_pipeline.py   # Pipeline verification
```

## Conclusion

The Entity Resolution System provides a sophisticated solution to the complex problem of identifying when different catalog entries refer to the same real-world person. By combining vector embeddings, machine learning classification, and graph-based clustering, the system achieves high accuracy while accommodating the specific needs of bibliographic data.

This project demonstrates an effective approach to entity resolution that can be adapted to similar challenges in other domains requiring sophisticated inference and contextual analysis.
