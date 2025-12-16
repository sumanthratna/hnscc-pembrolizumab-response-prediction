# Predicting Anti-PD-1 Response from FAST Multiplex Images

A Multiple Instance Learning (MIL) approach for predicting immunotherapy response in head and neck squamous cell carcinoma (HNSCC) using 21-channel FAST multiplex immunofluorescence images.

## Overview

This project implements:
- **Data preprocessing**: Downloads and preprocesses raw multiplex images from GCS
- **MIL model**: Attention-based CNN that aggregates field-of-view (FOV) level features into patient-level predictions
- **Baseline**: Logistic regression on hand-engineered marker features for comparison

## Project Structure

```
├── job_config_preprocess.yaml    # Anyscale job config for preprocessing
├── job_config_train.yaml         # Anyscale job config for training
├── bCPS_DataUpload_Ratna.csv     # Patient labels (responder/non-responder)
├── bCPS_Panel.csv                # Marker panel metadata
└── src/cpsc_5800_final_project/
    ├── config/
    │   └── marker_panel.yaml     # 21 marker definitions (cycle, channel, index)
    ├── data/                     # Preprocessing modules
    │   ├── channel_stacker.py    # Stack marker channels from TIFF files
    │   ├── enhanced_preprocessing.py  # Main preprocessing pipeline
    │   ├── illumination.py       # Illumination correction
    │   └── ...
    └── scripts/
        ├── preprocess_data.py    # Preprocessing script
        ├── train.py              # MIL training with Ray Data + Ray Train
        └── baseline_features.py  # Logistic regression baseline
```

## Setup

### Environment Variables

Set these before running any jobs:

```bash
export GCS_BUCKET="your-gcs-bucket-name"
export GCS_PREFIX="subdirectory"
```

### Dependencies

The job configs automatically install dependencies. For local development:

```bash
pip install torch torchvision numpy scikit-learn scikit-image albumentations gcsfs pyyaml tifffile ray[data,train]
```

## Running on Anyscale

### Step 1: Preprocess Data

Downloads raw images from GCS, applies illumination correction and normalization, and saves preprocessed `.npy` files to shared storage.

```bash
anyscale job submit job_config_preprocess.yaml
```

**Configuration** (edit `job_config_preprocess.yaml`):
- `--output`: Output directory for preprocessed data (default: `/mnt/shared_storage/preprocessed_data`)
- `--max-fovs`: Max FOVs per patient (default: 50)
- `--num-workers`: Parallel download workers (default: 8)

**Output**: 
- Preprocessed `.npy` files at native 1728×1728 resolution
- `metadata.json` with patient/FOV information

### Step 2: Train Model

Trains the MIL model using Leave-One-Out Cross-Validation (LOO-CV). Reads from preprocessed data on shared storage (no GCS access needed).

```bash
anyscale job submit job_config_train.yaml
```

**Model architecture**:
- **Backbone**: Channel Attention CNN (21 input channels → 64-dim embeddings)
- **Aggregation**: Attention-based MIL pooling
- **Output**: Binary classification (responder vs non-responder)

**Training configuration** (hardcoded in `train.py`):
- Image size: 256×256 (random crop from 512×512 resize)
- FOVs per patient: 20
- Epochs: 50
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-3)
- Augmentations: Random crop, horizontal/vertical flips, 90° rotations

**Output**: LOO-CV results with per-fold AUC and accuracy.

### Step 3: Run Baseline (Optional)

Computes a logistic regression baseline using hand-engineered marker features:

```bash
# Run locally or as a job
python src/cpsc_5800_final_project/scripts/baseline_features.py
```

## Model Details

### Multiple Instance Learning (MIL)

Each patient is treated as a "bag" of FOV instances. The model:
1. Extracts features from each FOV using a CNN backbone
2. Computes attention weights for each FOV
3. Aggregates FOV features into a patient-level representation
4. Predicts responder probability

### Channel Attention CNN

The backbone uses channel attention to learn marker interactions:
- 1×1 conv to mix 21 marker channels
- Standard conv layers for spatial features  
- Global average pooling → 64-dim embedding

## Data

- **Patients**: 11 HNSCC patients treated with anti-PD-1
- **Images**: ~50 FOVs per patient, 1728×1728 pixels, 21 protein markers
- **Markers**: PD-L1, CD8, CD4, FoxP3, CD163, panCK, etc. (see `marker_panel.yaml`)
- **Labels**: Binary (responder vs non-responder)

## References

- [Spatial analysis identifies DC niches as predictors of pembrolizumab therapy in HNSCC](https://www.sciencedirect.com/science/article/pii/S2666379125001739)
