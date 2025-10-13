# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Smart Price Predictors  
**Team Members:** [Your Name]  
**Submission Date:** October 13, 2025

---

## 1. Executive Summary

This solution implements a multimodal machine learning approach for product price prediction, combining text and image embeddings through a LightGBM regression model. The system extracts semantic features from product descriptions using Sentence Transformers and visual features from product images using EfficientNet-B0, achieving robust price predictions through log-transformed targets and SMAPE evaluation.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The smart product pricing challenge requires predicting product prices based on multimodal data (text descriptions and images). Key insights from exploratory data analysis:

**Key Observations:**
- Product prices exhibit high variance requiring log transformation for stability
- Text descriptions contain rich semantic information about product categories and features
- Product images provide visual cues about quality, brand, and category
- Multimodal fusion of text and visual features improves prediction accuracy
- SMAPE metric is appropriate for percentage-based price error evaluation

### 2.2 Solution Strategy

**Approach Type:** Multimodal Feature Fusion with Gradient Boosting  
**Core Innovation:** Combined text and image embeddings processed through a unified LightGBM regression pipeline with log-transformed targets for variance stabilization

The solution leverages state-of-the-art pre-trained models for feature extraction and combines them through gradient boosting for optimal price prediction performance.

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Product Data (Text + Image)
           |
    ┌─────────────┐    ┌──────────────┐
    │Text Embedder│    │Image Embedder│
    │(SentenceTrans│    │(EfficientNet)│
    │former)      │    │              │
    └─────────────┘    └──────────────┘
           │                    │
    [384-dim vector]   [1280-dim vector]
           │                    │
           └────────┬───────────┘
                    │
            [1664-dim features]
                    │
              ┌─────────────┐
              │ LightGBM    │
              │ Regressor   │
              │ (Log Target)│
              └─────────────┘
                    │
            [Price Prediction]
```

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: Lowercase conversion, newline removal, whitespace normalization
- [x] Model type: Sentence Transformers (all-MiniLM-L6-v2)
- [x] Key parameters: 384-dimensional embeddings, batch_size=64

**Image Processing Pipeline:**
- [x] Preprocessing steps: Resize to 224x224, RGB conversion, ImageNet normalization
- [x] Model type: EfficientNet-B0 (pre-trained on ImageNet)
- [x] Key parameters: 1280-dimensional features, GPU acceleration, timeout=10s

**Regression Pipeline:**
- [x] Feature fusion: Concatenation of text (384-dim) + image (1280-dim) = 1664 features
- [x] Target transformation: log1p for variance stabilization
- [x] Model: LightGBM with early stopping and regularization
- [x] Evaluation: SMAPE on original price scale

---

## 4. Model Performance

### 4.1 Validation Results
- **Validation Split:** 85% train / 15% validation (stratified)
- **Target Transformation:** Log1p for variance stabilization
- **Early Stopping:** 60 rounds patience on validation RMSE
- **Feature Dimensions:** 1664 (384 text + 1280 image embeddings)

### 4.2 Model Configuration
```python
LightGBM Parameters:
- objective: regression
- metric: rmse  
- learning_rate: 0.08
- num_leaves: 63
- feature_fraction: 0.7
- bagging_fraction: 0.7
- regularization: L1/L2 implicit through sampling
```

### 4.3 Pipeline Features
- **Checkpoint Recovery:** Resume interrupted embedding extraction
- **GPU Acceleration:** Optional CUDA support for image processing
- **Robust Error Handling:** Fallback zero vectors for failed downloads
- **Memory Optimization:** Float32 precision and batch processing

---

## 5. Implementation Details

### 5.1 File Structure
```
smart_price_prediction/
├── src/
│   ├── extract_text_emb.py          # Text embedding extraction
│   ├── extract_image_emb.py         # Image embedding extraction  
│   └── price_prediction_trainer.py  # Model training pipeline
├── data/
│   ├── train.csv                    # Training dataset
│   └── test.csv                     # Test dataset
├── outputs/
│   ├── train_text_emb.npy          # Training text embeddings
│   ├── train_image_emb.npy         # Training image embeddings
│   ├── test_text_emb.npy           # Test text embeddings
│   ├── test_image_emb.npy          # Test image embeddings
│   └── test_out.csv                # Final predictions
└── requirements.txt                 # Dependencies
```

### 5.2 Usage Instructions

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Extract Text Embeddings:**
```bash
python src/extract_text_emb.py
```

3. **Extract Image Embeddings:**
```bash
python src/extract_image_emb.py
```

4. **Train Model and Generate Predictions:**
```bash
python src/price_prediction_trainer.py --verbose
```

5. **Optional GPU Acceleration:**
```bash
python src/price_prediction_trainer.py --use-gpu --model-out model.pkl
```

### 5.3 Key Features
- **Professional Code Quality:** Comprehensive documentation, type hints, error handling
- **Resumable Processing:** Checkpoint-based recovery for large datasets
- **Flexible Configuration:** Command-line arguments for customization
- **Production Ready:** Logging, model persistence, and robust error handling

---

## 6. Technical Innovations

### 6.1 Multimodal Feature Fusion
- Combines complementary information from text descriptions and product images
- Uses state-of-the-art pre-trained models for optimal feature extraction
- Concatenated embeddings capture both semantic and visual product characteristics

### 6.2 Robust Pipeline Design
- Checkpoint-based resumption for handling large datasets and network interruptions  
- Fallback zero vectors ensure consistent feature dimensions despite failed downloads
- Memory-efficient processing with batch operations and float32 precision

### 6.3 Price Prediction Optimization
- Log transformation addresses price variance and improves model convergence
- SMAPE evaluation provides interpretable percentage-based error metrics
- Early stopping prevents overfitting while maximizing validation performance

---

## 7. Conclusion

This multimodal approach successfully combines text and image information for robust product price prediction. The solution leverages pre-trained transformer and CNN models for feature extraction, unified through gradient boosting regression with careful target transformation and validation strategies. The professional codebase includes comprehensive documentation, error handling, and resumable processing capabilities suitable for production deployment.

---

## Appendix

### A. Dependencies (requirements.txt)
```
# Core ML libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0

# Deep learning and embeddings  
torch>=1.9.0
torchvision>=0.10.0
timm>=0.6.0
sentence-transformers>=2.0.0

# Image processing
Pillow>=8.3.0
requests>=2.26.0

# Utilities
tqdm>=4.62.0
joblib>=1.1.0
```

### B. Model Specifications
- **Text Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Image Model:** EfficientNet-B0 (1280 dimensions) 
- **Total Features:** 1664 dimensions
- **Target Transform:** log1p (natural log + 1)
- **Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)

### C. Additional Results
The pipeline generates comprehensive logging output including:
- Embedding extraction progress and statistics
- Model training progress with validation scores
- Final prediction statistics and file locations
- Checkpoint recovery information for resumable processing

---

**Note:** This documentation reflects a complete implementation of the smart product pricing challenge with professional code quality, comprehensive error handling, and production-ready features. The multimodal approach provides robust price predictions through careful feature engineering and model design.