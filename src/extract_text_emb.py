"""
Text Embedding Extraction Pipeline

This module extracts text embeddings from product catalog content using 
Sentence Transformers. It processes both training and test datasets to generate 
dense vector representations of product descriptions for downstream ML tasks.

The pipeline uses the 'all-MiniLM-L6-v2' model which provides a good balance 
between performance and speed for semantic text understanding tasks.

Features:
- Text preprocessing and cleaning
- Batch processing for memory efficiency
- Progress tracking during embedding generation
- Consistent embedding format for train/test datasets

Usage:
    python extract_text_emb.py

Requirements:
    - sentence-transformers
    - pandas
    - numpy
    - CSV files with 'catalog_content' column

Output:
    - train_text_emb.npy: Training text embeddings
    - test_text_emb.npy: Test text embeddings
"""

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os

# Initialize sentence transformer model for text embedding
# all-MiniLM-L6-v2: Lightweight model optimized for speed and reasonable quality
model = SentenceTransformer('all-MiniLM-L6-v2')

def prepare_text(text_content):
    """
    Clean and normalize text content for embedding generation.
    
    Performs basic text preprocessing to ensure consistent input format
    for the sentence transformer model. This includes case normalization,
    whitespace cleanup, and newline handling.
    
    Args:
        text_content: Raw text content from catalog (can be any type)
        
    Returns:
        str: Cleaned and normalized text ready for embedding
        
    Note:
        Handles None/NaN values by converting to string first.
        Minimal preprocessing to preserve semantic meaning.
    """
    return str(text_content).lower().replace('\n', ' ').strip()


def extract_embeddings_from_csv(csv_path: str, output_path: str, text_column: str = 'catalog_content'):
    """
    Extract text embeddings from CSV file and save to disk.
    
    Loads CSV, preprocesses text content, generates embeddings using
    sentence transformer, and saves the resulting numpy array.
    
    Args:
        csv_path (str): Path to input CSV file
        output_path (str): Path where embeddings will be saved (.npy format)
        text_column (str): Column name containing text content
        
    Returns:
        np.ndarray: Generated text embeddings
    """
    print(f"Processing CSV: {csv_path}")
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Preprocess text content
    texts = [prepare_text(text) for text in df[text_column].tolist()]
    
    # Generate embeddings with batch processing for memory efficiency
    print(f"Generating embeddings using {model.__class__.__name__}...")
    embeddings = model.encode(
        texts, 
        batch_size=64,                    # Process in batches to manage memory
        show_progress_bar=True,           # Display progress for user feedback
        convert_to_numpy=True             # Return numpy array for consistency
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings to disk
    np.save(output_path, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings to {output_path}")
    print(f"Embedding dimensions: {embeddings.shape}")
    
    return embeddings


def main():
    """
    Main execution function for text embedding extraction.
    
    Processes both training and test datasets to generate text embeddings
    using sentence transformers. Creates project-relative output paths and
    ensures proper directory structure.
    """
    # Project root directory for relative paths
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(ROOT, 'data')
    OUTPUT_DIR = os.path.join(ROOT, 'outputs')

    print("Starting text embedding extraction pipeline...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using model: {model.get_sentence_embedding_dimension()}-dim {model._modules['0'].__class__.__name__}")
    
    # Process training dataset
    print("\n" + "-" * 30)
    print("PROCESSING TRAINING DATA")
    print("-" * 30)
    train_csv_path = os.path.join(DATA_DIR, 'train.csv')
    train_output_path = os.path.join(OUTPUT_DIR, 'train_text_emb.npy')
    extract_embeddings_from_csv(train_csv_path, train_output_path)

    # Process test dataset  
    print("\n" + "-" * 30)
    print("PROCESSING TEST DATA")
    print("-" * 30)
    test_csv_path = os.path.join(DATA_DIR, 'test.csv')
    test_output_path = os.path.join(OUTPUT_DIR, 'test_text_emb.npy')
    extract_embeddings_from_csv(test_csv_path, test_output_path)

    # Summary
    print("\n" + "="*50)
    print("✅ Text embedding extraction completed successfully!")
    print("Generated files:")
    print(f"  📄 {train_output_path}")
    print(f"  📄 {test_output_path}")
    print("="*50)


if __name__ == '__main__':
    main()