#!/usr/bin/env python3
"""
Simple Baseline: Mean Marker Intensities + Logistic Regression

This baseline extracts hand-crafted features from marker intensities
and uses classical ML instead of deep learning.

Features per patient:
- Mean intensity per marker (aggregated across FOVs)
- Std intensity per marker
- Percentile features (25th, 75th)

This helps us understand if there's signal in the raw marker data.
"""

import os
from pathlib import Path
import numpy as np
import gcsfs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import LeaveOneOut

from cpsc_5800_final_project.data.channel_stacker import MarkerPanel, enumerate_fovs
from cpsc_5800_final_project.data.enhanced_preprocessing import load_fov_enhanced
from cpsc_5800_final_project.data.labels import load_labels, get_sample_ids


def find_paths():
    """Find config paths."""
    cwd = Path.cwd()
    candidates = [
        (cwd / "src" / "cpsc_5800_final_project" / "config" / "marker_panel.yaml",
         cwd / "bCPS_DataUpload_Ratna.csv"),
        (Path("/home/ray/default/src/cpsc_5800_final_project/config/marker_panel.yaml"),
         Path("/home/ray/default/bCPS_DataUpload_Ratna.csv")),
    ]
    for marker_path, labels_path in candidates:
        if marker_path.exists() and labels_path.exists():
            return marker_path, labels_path
    raise FileNotFoundError(f"Could not find config files. CWD: {cwd}")


def extract_fov_features(image: np.ndarray) -> dict:
    """
    Extract features from a single FOV image.
    
    Args:
        image: (H, W, C) normalized image
    
    Returns:
        Dict of features
    """
    n_channels = image.shape[-1]
    features = {}
    
    for c in range(n_channels):
        channel = image[:, :, c]
        
        # Basic statistics
        features[f"ch{c}_mean"] = np.mean(channel)
        features[f"ch{c}_std"] = np.std(channel)
        features[f"ch{c}_median"] = np.median(channel)
        features[f"ch{c}_p25"] = np.percentile(channel, 25)
        features[f"ch{c}_p75"] = np.percentile(channel, 75)
        features[f"ch{c}_p95"] = np.percentile(channel, 95)
        features[f"ch{c}_max"] = np.max(channel)
        
        # Fraction of "positive" pixels (above threshold)
        features[f"ch{c}_frac_pos"] = np.mean(channel > 0.1)
        features[f"ch{c}_frac_high"] = np.mean(channel > 0.5)
    
    return features


def aggregate_patient_features(fov_features_list: list[dict]) -> np.ndarray:
    """
    Aggregate FOV features to patient level.
    
    Args:
        fov_features_list: List of feature dicts from each FOV
    
    Returns:
        1D feature vector for patient
    """
    if not fov_features_list:
        return None
    
    # Get all feature names
    feature_names = sorted(fov_features_list[0].keys())
    
    # Aggregate: mean and std across FOVs
    aggregated = []
    for name in feature_names:
        values = [f[name] for f in fov_features_list]
        aggregated.append(np.mean(values))  # Mean across FOVs
        aggregated.append(np.std(values))   # Variability across FOVs
    
    return np.array(aggregated)


def main():
    print("=" * 70)
    print("  Simple Baseline: Mean Marker Intensities + Logistic Regression")
    print("=" * 70)
    print()
    
    # Config
    config = {
        "max_fovs_per_patient": 24,  # Use more FOVs for feature extraction
        "use_enhanced_preprocessing": True,
    }
    print(f"Config: {config}")
    
    # Setup
    fs = gcsfs.GCSFileSystem()
    bucket = os.environ.get("GCS_BUCKET")
    prefix = os.environ.get("GCS_PREFIX", "shared_storage")
    
    if not bucket:
        raise ValueError("GCS_BUCKET environment variable must be set")
    
    marker_panel_path, labels_csv_path = find_paths()
    marker_panel = MarkerPanel.from_yaml(str(marker_panel_path))
    labels_dict = load_labels(str(labels_csv_path))
    sample_ids = get_sample_ids(str(labels_csv_path))
    
    print(f"\nMarker panel: {len(marker_panel.markers)} markers")
    print(f"Samples: {len(sample_ids)}")
    
    # Extract features for all patients
    print("\n" + "-" * 70)
    print("Extracting features...")
    print("-" * 70)
    
    patient_features = {}
    patient_labels = {}
    
    for sample_id in sample_ids:
        label = labels_dict.get(sample_id, labels_dict.get(sample_id.split("_")[0], 0))
        fovs = enumerate_fovs(fs, bucket, sample_id, prefix)[:config["max_fovs_per_patient"]]
        
        if not fovs:
            print(f"  {sample_id}: No FOVs, skipping")
            continue
        
        print(f"  {sample_id}: extracting from {len(fovs)} FOVs (label={'R' if label else 'NR'})")
        
        fov_features = []
        for fov_id in fovs:
            try:
                image, metadata = load_fov_enhanced(
                    fs, bucket, sample_id, fov_id, marker_panel, prefix,
                    check_alignment=False,  # Skip for speed
                    illumination_correction=True,
                    normalize=True,
                )
                
                if image is not None:
                    features = extract_fov_features(image)
                    fov_features.append(features)
            except Exception as e:
                print(f"    Error {fov_id}: {e}")
        
        if fov_features:
            patient_features[sample_id] = aggregate_patient_features(fov_features)
            patient_labels[sample_id] = label
            print(f"    -> {len(fov_features)} FOVs, {len(patient_features[sample_id])} features")
    
    # Prepare data matrices
    patients = list(patient_features.keys())
    X = np.array([patient_features[p] for p in patients])
    y = np.array([patient_labels[p] for p in patients])
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"Labels: {y.sum()} responders, {len(y) - y.sum()} non-responders")
    
    # Leave-One-Out Cross-Validation
    print("\n" + "=" * 70)
    print("  Leave-One-Out Cross-Validation")
    print("=" * 70)
    
    loo = LeaveOneOut()
    predictions = []
    true_labels = []
    patient_ids_ordered = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_patient = patients[test_idx[0]]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Logistic Regression with regularization
        model = LogisticRegression(
            C=0.1,  # Strong regularization for small dataset
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        pred_proba = model.predict_proba(X_test_scaled)[0, 1]
        predictions.append(pred_proba)
        true_labels.append(y_test[0])
        patient_ids_ordered.append(test_patient)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Results
    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)
    
    auc = roc_auc_score(true_labels, predictions)
    preds_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(true_labels, preds_binary)
    
    # Inverted
    inv_predictions = 1 - predictions
    auc_inv = roc_auc_score(true_labels, inv_predictions)
    inv_binary = (inv_predictions > 0.5).astype(int)
    accuracy_inv = accuracy_score(true_labels, inv_binary)
    
    print(f"\nLogistic Regression (L2, C=0.1):")
    print(f"  Original:  AUC = {auc:.4f}, Accuracy = {accuracy:.4f} ({sum(preds_binary == true_labels)}/{len(true_labels)})")
    print(f"  Inverted:  AUC = {auc_inv:.4f}, Accuracy = {accuracy_inv:.4f} ({sum(inv_binary == true_labels)}/{len(true_labels)})")
    
    best_auc = max(auc, auc_inv)
    inverted_better = auc_inv > auc
    
    print(f"\nPer-patient predictions:")
    print(f"  {'Patient':<25} {'True':>6} {'Pred':>8} {'Correct':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8}")
    
    use_preds = inv_predictions if inverted_better else predictions
    for pid, true_label, pred in zip(patient_ids_ordered, true_labels, use_preds):
        pred_label = 1 if pred > 0.5 else 0
        correct = "✓" if pred_label == true_label else "✗"
        label_str = "R" if true_label else "NR"
        print(f"  {pid:<25} {label_str:>6} {pred:>8.4f} {correct:>8}")
    
    print(f"\n  Mean pred - Responders: {use_preds[true_labels == 1].mean():.4f}")
    print(f"  Mean pred - Non-responders: {use_preds[true_labels == 0].mean():.4f}")
    
    # Try different regularization strengths
    print("\n" + "-" * 70)
    print("  Regularization Sweep")
    print("-" * 70)
    
    for C in [0.01, 0.1, 1.0, 10.0]:
        preds = []
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            preds.append(model.predict_proba(X_test_scaled)[0, 1])
        
        preds = np.array(preds)
        auc_c = roc_auc_score(true_labels, preds)
        auc_c_inv = roc_auc_score(true_labels, 1 - preds)
        acc_c = accuracy_score(true_labels, (preds > 0.5).astype(int))
        print(f"  C={C:<5}: AUC={max(auc_c, auc_c_inv):.4f}, Acc={acc_c:.4f}")
    
    print("\n" + "=" * 70)
    print(f"  Baseline Complete! Best AUC = {best_auc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()


