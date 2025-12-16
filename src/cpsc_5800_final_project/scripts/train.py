#!/usr/bin/env python3
"""
MIL Training Script v2 - Production-Ready with Ray Data

Following best practices from:
- https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html
- https://docs.ray.io/en/latest/data/working-with-images.html
- https://albumentations.ai/docs/2-core-concepts/pipelines/

Features:
- Ray Data for data loading and preprocessing
- Ray Train with prepare_model()
- albumentations for augmentations
- LOO-CV evaluation

Note: MIL requires grouping FOVs into patient-level bags.
We use Ray Data to preprocess, then group in the training worker.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Training configuration."""
    seed: int = 0
    cache_dir: str = "/mnt/shared_storage/preprocessed_data"
    load_size: int = 512
    crop_size: int = 256
    n_channels: int = 21
    hidden_dim: int = 64
    fovs_per_patient: int = 20
    batch_size: int = 1
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-3
    num_workers: int = 1
    use_gpu: bool = True


def load_metadata(cache_dir: str) -> Dict:
    """Load dataset metadata."""
    with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
        return json.load(f)


# =============================================================================
# Training Function with Ray Data
# =============================================================================

def train_func(train_config: Dict):
    """
    Training function using Ray Data.
    
    Ray Data approach:
    - Access preprocessed data via ray.train.get_dataset_shard("train")
    - Use iter_torch_batches() for streaming data
    - Group by patient_id to form MIL bags
    """
    import os
    import random
    from collections import defaultdict
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import ray.train
    import ray.train.torch
    import albumentations as A
    
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    class LocalConfig:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    # -------------------------------------------------------------------------
    # Model Definitions
    # -------------------------------------------------------------------------
    class ChannelAttentionCNN(nn.Module):
        def __init__(self, in_channels=21, hidden_channels=32, embed_dim=64, dropout=0.5):
            super().__init__()
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(),
                nn.Linear(in_channels // 2, in_channels),
                nn.Sigmoid(),
            )
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, embed_dim),
            )
        
        def forward(self, x):
            attn = self.channel_attention(x)
            x = x * attn.unsqueeze(-1).unsqueeze(-1)
            x = self.conv(x)
            x = self.fc(x)
            return x

    class SimpleMILClassifier(nn.Module):
        def __init__(self, in_channels=21, hidden_channels=32, embed_dim=64, hidden_dim=32, dropout=0.5):
            super().__init__()
            self.backbone = ChannelAttentionCNN(in_channels, hidden_channels, embed_dim, dropout)
            self.attention = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.classifier = nn.Linear(embed_dim, 1)
        
        def forward(self, x):
            b, n, c, h, w = x.shape
            x_flat = x.view(b * n, c, h, w)
            embeddings = self.backbone(x_flat)
            embeddings = embeddings.view(b, n, -1)
            attn_scores = self.attention(embeddings)
            attn_weights = F.softmax(attn_scores, dim=1)
            bag_embedding = (embeddings * attn_weights).sum(dim=1)
            logits = self.classifier(bag_embedding)
            probs = torch.sigmoid(logits)
            return {"logits": logits, "probabilities": probs}

    # -------------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------------
    def get_augmentation(crop_size, training=True):
        if training:
            return A.Compose([
                A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            return A.Compose([
                A.CenterCrop(width=crop_size, height=crop_size, p=1.0),
            ])

    # -------------------------------------------------------------------------
    # Set seed
    # -------------------------------------------------------------------------
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------
    # Main training logic
    # -------------------------------------------------------------------------
    config = LocalConfig(train_config["config"])
    fold_idx = train_config.get("fold_idx", 0)
    val_record = train_config.get("val_record", None)
    
    set_seed(config.seed + fold_idx)
    
    # Get Ray Data shard
    train_data_shard = ray.train.get_dataset_shard("train")
    
    # Collect all data from Ray Data into patient groups
    # This is needed for MIL where we need to form bags per patient
    patient_data = defaultdict(lambda: {"images": [], "label": None})
    
    print(f"Loading data from Ray Data shard...")
    for batch in train_data_shard.iter_batches(batch_size=100):
        for i in range(len(batch["image"])):
            pid = batch["patient_id"][i]
            img = batch["image"][i]
            label = batch["label"][i]
            if img is not None:
                patient_data[pid]["images"].append(img)
                patient_data[pid]["label"] = label
    
    print(f"Loaded {len(patient_data)} patients")
    
    # Create model
    model = SimpleMILClassifier(
        in_channels=config.n_channels,
        hidden_dim=config.hidden_dim,
    )
    model = ray.train.torch.prepare_model(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    device = ray.train.torch.get_device()
    
    augment = get_augmentation(config.crop_size, training=True)
    patient_ids = list(patient_data.keys())
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        random.shuffle(patient_ids)
        
        for pid in patient_ids:
            pdata = patient_data[pid]
            images = pdata["images"]
            label = pdata["label"]
            
            if len(images) == 0:
                continue
            
            # Sample and augment FOVs
            n_sample = min(config.fovs_per_patient, len(images))
            indices = random.sample(range(len(images)), n_sample)
            
            processed = []
            for i in indices:
                img = images[i]
                augmented = augment(image=img)
                img_chw = np.transpose(augmented["image"], (2, 0, 1))
                processed.append(img_chw)
            
            # Pad if needed
            while len(processed) < config.fovs_per_patient:
                processed.append(processed[random.randint(0, len(processed) - 1)])
            
            # Create bag tensor: (1, N, C, H, W)
            bag = np.stack(processed[:config.fovs_per_patient])
            bag = torch.from_numpy(bag).unsqueeze(0).to(device)
            bag_label = torch.tensor([label], dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            output = model(bag)
            loss = criterion(output["logits"].squeeze(-1), bag_label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # Validation
        val_metrics = {}
        if val_record and epoch == config.epochs - 1:
            model.eval()
            val_augment = get_augmentation(config.crop_size, training=False)
            
            # Load validation FOVs
            from skimage.transform import resize as sk_resize
            val_images = []
            for fpath in val_record["fov_files"][:config.fovs_per_patient]:
                if os.path.exists(fpath):
                    try:
                        img = np.load(fpath)
                        img = sk_resize(img, (config.load_size, config.load_size),
                                       preserve_range=True, anti_aliasing=True).astype(np.float32)
                        val_images.append(img)
                    except:
                        continue
            
            if val_images:
                processed = []
                for img in val_images[:config.fovs_per_patient]:
                    augmented = val_augment(image=img)
                    img_chw = np.transpose(augmented["image"], (2, 0, 1))
                    processed.append(img_chw)
                
                while len(processed) < config.fovs_per_patient:
                    processed.append(processed[0])
                
                bag = np.stack(processed[:config.fovs_per_patient])
                bag = torch.from_numpy(bag).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(bag)
                    prob = output["probabilities"].cpu().item()
                    pred = 1 if prob > 0.5 else 0
                    val_metrics = {
                        "val_prob": prob,
                        "val_pred": pred,
                        "val_label": val_record["label"],
                        "val_correct": int(pred == val_record["label"])
                    }
        
        metrics = {"epoch": epoch, "train_loss": avg_train_loss, **val_metrics}
        ray.train.report(metrics)


# =============================================================================
# Create Ray Dataset
# =============================================================================

def create_ray_dataset(train_records: List[Dict], config: Config):
    """
    Create Ray Dataset from patient records.
    
    Preprocessing happens via Ray Data map operations,
    which can scale to large datasets.
    """
    import ray.data
    from skimage.transform import resize as sk_resize
    
    # Build FOV-level records
    fov_records = []
    for rec in train_records:
        for fpath in rec["fov_files"]:
            fov_records.append({
                "fov_path": fpath,
                "patient_id": rec["patient_id"],
                "label": rec["label"],
            })
    
    # Create dataset
    ds = ray.data.from_items(fov_records)
    
    # Define preprocessing function
    load_size = config.load_size
    
    def load_fov(row):
        try:
            img = np.load(row["fov_path"])
            img = sk_resize(img, (load_size, load_size),
                           preserve_range=True, anti_aliasing=True).astype(np.float32)
            return {
                "image": img,
                "patient_id": row["patient_id"],
                "label": row["label"],
            }
        except Exception:
            return {
                "image": None,
                "patient_id": row["patient_id"],
                "label": row["label"],
            }
    
    # Apply preprocessing with Ray Data
    ds = ds.map(load_fov)
    ds = ds.filter(lambda row: row["image"] is not None)
    
    return ds


# =============================================================================
# LOO-CV
# =============================================================================

def run_loocv(config: Config):
    """Run LOO-CV with Ray Train + Ray Data."""
    import ray
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    ray.init(ignore_reinit_error=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    metadata = load_metadata(config.cache_dir)
    patient_ids = list(metadata["patients"].keys())
    n_patients = len(patient_ids)
    
    print("=" * 70)
    print("  MIL Training - LOO-CV with Ray Train + Ray Data")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Load size: {config.load_size}")
    print(f"  Crop size: {config.crop_size}")
    print(f"  FOVs per patient: {config.fovs_per_patient}")
    print(f"  Epochs: {config.epochs}")
    print(f"\nPatients: {n_patients}")
    print("-" * 70)
    
    # Build records
    all_records = []
    for pid in patient_ids:
        pdata = metadata["patients"][pid]
        all_records.append({
            "patient_id": pid,
            "fov_files": [os.path.join(config.cache_dir, f["filename"]) for f in pdata["fovs"]],
            "label": pdata["label"],
        })
    
    # LOO-CV
    results = []
    for fold_idx, test_pid in enumerate(patient_ids):
        print(f"\n=== Fold {fold_idx + 1}/{n_patients}: Test={test_pid} ===")
        
        train_records = [r for r in all_records if r["patient_id"] != test_pid]
        val_record = [r for r in all_records if r["patient_id"] == test_pid][0]
        
        # Create Ray Dataset for training
        train_ds = create_ray_dataset(train_records, config)
        
        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "config": config.__dict__,
                "fold_idx": fold_idx,
                "val_record": val_record,
            },
            datasets={"train": train_ds},
            scaling_config=ScalingConfig(
                num_workers=config.num_workers,
                use_gpu=config.use_gpu,
            ),
            run_config=RunConfig(name=f"mil_fold_{fold_idx}"),
        )
        
        result = trainer.fit()
        
        final_metrics = result.metrics or {}
        prob = final_metrics.get("val_prob", 0.5)
        pred = final_metrics.get("val_pred", 0)
        label = final_metrics.get("val_label", val_record["label"])
        
        results.append({
            "patient_id": test_pid,
            "prob": prob,
            "pred": pred,
            "label": label,
        })
        
        print(f"  prob={prob:.3f}, pred={pred}, label={label}")
    
    # Metrics
    y_true = [r["label"] for r in results]
    y_prob = [r["prob"] for r in results]
    y_pred = [r["pred"] for r in results]
    
    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\nAUC: {auc:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"\nBaseline: AUC=0.77")
    
    if auc > 0.77:
        print("\n✅ Beat baseline!")
    else:
        print(f"\n❌ Below baseline by {0.77 - auc:.3f}")
    
    return {"auc": auc, "accuracy": acc}


def main():
    config = Config()
    run_loocv(config)


if __name__ == "__main__":
    main()
