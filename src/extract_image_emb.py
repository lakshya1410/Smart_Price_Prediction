"""
Image Embedding Extraction Pipeline

This module extracts image embeddings from product images using a pre-trained 
EfficientNet model. It processes both training and test datasets by downloading 
images from URLs and generating dense vector representations for downstream ML tasks.

The pipeline features:
- Robust image downloading with error handling and timeouts
- EfficientNet-B0 feature extraction (1280-dimensional embeddings)
- Checkpoint-based resumption for large datasets
- Fallback zero vectors for failed downloads
- Memory-efficient batch processing
- Progress tracking and status reporting

Architecture:
- Model: EfficientNet-B0 (pretrained on ImageNet)
- Input: 224x224 RGB images with ImageNet normalization
- Output: 1280-dimensional feature vectors
- Device: Auto-detection (CUDA/CPU)

Usage:
    python extract_image_emb.py

Requirements:
    - torch, torchvision, timm
    - PIL (Pillow)
    - requests
    - pandas, numpy, tqdm
    - CSV files with image URL columns

Output:
    - train_image_emb.npy: Training image embeddings
    - test_image_emb.npy: Test image embeddings
    - Checkpoint files for resumption capability
"""

import os
import requests
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from torchvision import transforms
import timm

# ===================================================================
# CONFIGURATION AND SETUP
# ===================================================================

# Project directory structure (relative to src/ folder)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_csv = os.path.join(ROOT, 'data', 'train.csv')
test_csv = os.path.join(ROOT, 'data', 'test.csv')
out_dir = os.path.join(ROOT, 'outputs')

# Ensure output directory exists for saving embeddings
os.makedirs(out_dir, exist_ok=True)

# ===================================================================
# MODEL INITIALIZATION AND IMAGE PREPROCESSING
# ===================================================================

# Auto-detect compute device (prefer GPU for faster inference)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing image embedding pipeline on device: {device}")

# Load pre-trained EfficientNet-B0 model for feature extraction
# - num_classes=0 removes classification head, returns raw features
# - Pretrained weights from ImageNet provide strong visual representations
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)

# Image preprocessing pipeline matching ImageNet training procedure
transform = transforms.Compose([
    transforms.Resize((224, 224)),                                    # Resize to model input size
    transforms.ToTensor(),                                           # Convert PIL to tensor [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],               # ImageNet mean normalization
                        std=[0.229, 0.224, 0.225]),                # ImageNet std normalization
])

print(f"Model loaded: EfficientNet-B0 (Output dim: 1280)")
print(f"Image preprocessing: 224x224 RGB with ImageNet normalization")

# ===================================================================
# IMAGE DOWNLOAD AND EMBEDDING EXTRACTION
# ===================================================================

def get_embedding_from_url(url: str, timeout: int = 10) -> np.ndarray:
    """
    Download image from URL and extract feature embedding using EfficientNet.
    
    Handles the complete pipeline from URL to embedding: downloads image,
    applies preprocessing transforms, runs inference, and returns features.
    Robust error handling ensures graceful failure for invalid URLs or images.
    
    Args:
        url (str): HTTP/HTTPS URL pointing to an image file
        timeout (int): Request timeout in seconds (default: 10)
        
    Returns:
        np.ndarray: 1280-dimensional feature vector, or None if extraction fails
        
    Raises:
        No exceptions raised - returns None on any failure for robust batch processing
        
    Note:
        - Converts images to RGB to handle grayscale/RGBA formats
        - Uses torch.no_grad() for memory efficiency during inference
        - Automatically moves tensors between CPU/GPU as needed
    """
    try:
        # Download image with timeout protection
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses
        
        # Load and convert image to RGB format
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Apply preprocessing transforms and add batch dimension
        x = transform(img).unsqueeze(0).to(device)
        
        # Extract features without gradient computation (inference mode)
        with torch.no_grad():
            features = model(x)
        
        # Move to CPU and convert to 1D numpy array
        return features.cpu().numpy().reshape(-1)
        
    except Exception as e:
        # Silently handle all failures (network, parsing, model errors)
        # This ensures batch processing continues even with bad URLs
        return None

# ===================================================================
# BATCH PROCESSING WITH CHECKPOINT RECOVERY
# ===================================================================

def process_csv(csv_path: str, output_path: str, checkpoint_path: str = None, chunk_checkpoint: int = 500):
    """
    Process CSV file to extract image embeddings with checkpoint-based resumption.
    
    Reads CSV containing image URLs, downloads images, extracts embeddings using
    EfficientNet, and saves results. Supports resumption from checkpoints for
    large datasets that may be interrupted during processing.
    
    Args:
        csv_path (str): Path to CSV file containing image URLs
        output_path (str): Path where final embeddings will be saved (.npy format)
        checkpoint_path (str, optional): Path for checkpoint file to enable resumption
        chunk_checkpoint (int): Save checkpoint every N processed images (default: 500)
        
    Returns:
        None
        
    CSV Requirements:
        - Must contain image URL column ('image_link' or 'image')  
        - Must contain ID column ('sample_id' or 'id')
        
    Output Format:
        - Numpy array of shape (n_samples, 1280) saved as .npy file
        - Failed downloads replaced with zero vectors
        
    Checkpoint Format:
        - Dictionary with 'embs' (list of embeddings) and 'done_ids' (processed IDs)
        - Enables resumption if process is interrupted
        
    Note:
        Memory usage scales with checkpoint frequency. Lower chunk_checkpoint 
        values provide more frequent saves but use more disk I/O.
    """
    print(f"Processing dataset: {csv_path}")
    
    # Load CSV dataset
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Loaded {total_samples} samples from dataset")
    
    # Initialize embedding storage and tracking
    embeddings = []
    processed_ids = set()

    # Attempt to resume from checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint: {checkpoint_path}")
        try:
            checkpoint_data = np.load(checkpoint_path, allow_pickle=True).item()
            embeddings = checkpoint_data.get("embs", [])
            processed_ids = set(checkpoint_data.get("done_ids", []))
            print(f"🔄 Resuming from checkpoint — already processed {len(processed_ids)}/{total_samples} images")
        except Exception as e:
            print(f"⚠️  Checkpoint corrupted, starting fresh: {e}")
            embeddings = []
            processed_ids = set()

    # Process each row in the dataset
    progress_bar = tqdm(df.iterrows(), total=total_samples, desc="Extracting embeddings")
    
    for _, row in progress_bar:
        # Extract sample identifier (flexible column names)
        sample_id = row.get('sample_id') if 'sample_id' in row else row.get('id')
        
        # Skip if already processed (resumption logic)
        if sample_id in processed_ids:
            continue

        # Extract image URL (flexible column names)
        image_url = row.get('image_link') or row.get('image') or ''
        
        # Generate embedding or use fallback
        if image_url:
            embedding = get_embedding_from_url(image_url)
        else:
            embedding = None
            
        # Use zero vector as fallback for failed extractions
        if embedding is None:
            embedding = np.zeros(1280, dtype=np.float32)  # EfficientNet-B0 output dimension
            
        embeddings.append(embedding)
        processed_ids.add(sample_id)

        # Periodic checkpoint saving for resumption capability
        if checkpoint_path and len(processed_ids) % chunk_checkpoint == 0:
            # Save current embeddings
            if embeddings:
                np.save(output_path, np.vstack(embeddings))
            
            # Save checkpoint data
            checkpoint_data = {
                "embs": embeddings,
                "done_ids": list(processed_ids)
            }
            np.save(checkpoint_path, checkpoint_data)
            progress_bar.set_postfix({"checkpointed": len(processed_ids)})

    progress_bar.close()

    # Final save of all embeddings
    if embeddings:
        final_embeddings = np.vstack(embeddings)
        print(f"Stacking {len(embeddings)} embeddings into shape {final_embeddings.shape}")
    else:
        # Handle edge case of no valid images
        final_embeddings = np.zeros((0, 1280), dtype=np.float32)
        print("No valid embeddings found, saving empty array")
    
    np.save(output_path, final_embeddings)
    
    # Save final checkpoint
    if checkpoint_path:
        final_checkpoint = {
            "embs": embeddings,
            "done_ids": list(processed_ids)
        }
        np.save(checkpoint_path, final_checkpoint)
    
    print(f"✅ Completed processing: {output_path}")
    print(f"   📊 Final shape: {final_embeddings.shape}")
    print(f"   💾 Saved {len(processed_ids)} embeddings")

# ===================================================================
# MAIN EXECUTION PIPELINE
# ===================================================================

def main():
    """
    Execute the complete image embedding extraction pipeline.
    
    Processes both training and test datasets to generate image embeddings
    using EfficientNet-B0. Includes checkpoint-based resumption for robustness
    against interruptions during large dataset processing.
    
    Pipeline Steps:
    1. Initialize model and preprocessing on available device (GPU/CPU)
    2. Process training dataset with checkpoint support
    3. Process test dataset with checkpoint support  
    4. Generate summary statistics and completion report
    
    Output Files:
        - train_image_emb.npy: Training embeddings (N × 1280)
        - test_image_emb.npy: Test embeddings (M × 1280)  
        - *_checkpoint.npy: Resumption checkpoints (optional)
    """
    print("="*60)
    print("🖼️  IMAGE EMBEDDING EXTRACTION PIPELINE")
    print("="*60)
    print(f"📁 Data directory: {os.path.dirname(train_csv)}")
    print(f"📁 Output directory: {out_dir}")
    print(f"🔧 Device: {device}")
    print(f"🧠 Model: EfficientNet-B0 (1280-dim features)")
    print("-"*60)

    # Process training dataset
    print("\n📚 PROCESSING TRAINING DATASET")
    print("-"*40)
    train_output = os.path.join(out_dir, 'train_image_emb.npy')
    train_checkpoint = os.path.join(out_dir, 'train_checkpoint.npy')
    
    if os.path.exists(train_csv):
        process_csv(train_csv, train_output, train_checkpoint)
    else:
        print(f"⚠️  Training CSV not found: {train_csv}")

    # Process test dataset
    print("\n🧪 PROCESSING TEST DATASET") 
    print("-"*40)
    test_output = os.path.join(out_dir, 'test_image_emb.npy')
    test_checkpoint = os.path.join(out_dir, 'test_checkpoint.npy')
    
    if os.path.exists(test_csv):
        process_csv(test_csv, test_output, test_checkpoint)
    else:
        print(f"⚠️  Test CSV not found: {test_csv}")

    # Pipeline completion summary
    print("\n" + "="*60)
    print("🎉 IMAGE EMBEDDING EXTRACTION COMPLETED")
    print("="*60)
    
    generated_files = []
    if os.path.exists(train_output):
        train_shape = np.load(train_output).shape
        generated_files.append(f"📄 {train_output} → {train_shape}")
    if os.path.exists(test_output):
        test_shape = np.load(test_output).shape  
        generated_files.append(f"📄 {test_output} → {test_shape}")
    
    if generated_files:
        print("Generated embedding files:")
        for file_info in generated_files:
            print(f"  {file_info}")
    else:
        print("⚠️  No embedding files generated - check input CSV files")
    
    print("\n💡 Note: Checkpoint files enable resumption if process is interrupted")
    print("="*60)


if __name__ == '__main__':
    main()
