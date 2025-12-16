#!/usr/bin/env python3
"""
Preprocess FAST Multiplex Images

Downloads raw images from GCS, applies preprocessing (illumination correction,
channel stacking, normalization), and saves to local storage for training.

Output: Preprocessed .npy files at native 1728×1728 resolution + metadata.json

Usage:
    python preprocess_data.py --output /path/to/cache --max-fovs 30 --num-workers 4
"""

import os
from pathlib import Path
import numpy as np
import gcsfs
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cpsc_5800_final_project.data.channel_stacker import MarkerPanel, enumerate_fovs
from cpsc_5800_final_project.data.enhanced_preprocessing import load_fov_enhanced
from cpsc_5800_final_project.data.labels import load_labels, get_sample_ids


def find_config_paths():
    """Find marker panel and labels CSV paths."""
    cwd = Path.cwd()
    candidates = [
        (
            cwd / "src" / "cpsc_5800_final_project" / "config" / "marker_panel.yaml",
            cwd / "bCPS_DataUpload_Ratna.csv",
        ),
        (
            Path(
                "/home/ray/default/src/cpsc_5800_final_project/config/marker_panel.yaml"
            ),
            Path("/home/ray/default/bCPS_DataUpload_Ratna.csv"),
        ),
    ]
    for marker_path, labels_path in candidates:
        if marker_path.exists() and labels_path.exists():
            return marker_path, labels_path
    raise FileNotFoundError(f"Could not find config files. CWD: {cwd}")


def process_single_fov(args):
    """
    Process a single field of view (FOV).

    Downloads from GCS, applies preprocessing, saves as .npy file.
    Supports resumption by skipping already-processed files.
    """
    sample_id, fov_id, marker_panel_path, bucket, prefix, output_dir = args

    filename = f"{sample_id}_{fov_id}.npy"
    filepath = os.path.join(output_dir, filename)

    # Check if already processed (for resumption)
    if os.path.exists(filepath):
        try:
            image = np.load(filepath)
            return {
                "success": True,
                "sample_id": sample_id,
                "fov_id": fov_id,
                "filename": filename,
                "shape": list(image.shape),
                "size_mb": image.nbytes / (1024 * 1024),
                "cached": True,
            }
        except:
            os.remove(filepath)  # Corrupted file - reprocess

    fs = gcsfs.GCSFileSystem()
    marker_panel = MarkerPanel.from_yaml(marker_panel_path)

    try:
        image, _ = load_fov_enhanced(
            fs,
            bucket,
            sample_id,
            fov_id,
            marker_panel,
            prefix,
            check_alignment=False,
            illumination_correction=True,
            normalize=True,
        )

        if image is None:
            return {"success": False, "sample_id": sample_id, "fov_id": fov_id}

        # Save at full resolution
        np.save(filepath, image.astype(np.float32))

        return {
            "success": True,
            "sample_id": sample_id,
            "fov_id": fov_id,
            "filename": filename,
            "shape": list(image.shape),
            "size_mb": image.nbytes / (1024 * 1024),
            "cached": False,
        }
    except Exception as e:
        return {
            "success": False,
            "sample_id": sample_id,
            "fov_id": fov_id,
            "error": str(e),
        }


def preprocess_data(
    output_dir: str = "./preprocessed_data/",
    max_fovs_per_patient: int = 30,
    num_workers: int = 4,
):
    """
    Preprocess all patient FOVs and save to output directory.

    Args:
        output_dir: Directory to save preprocessed .npy files
        max_fovs_per_patient: Maximum FOVs to process per patient
        num_workers: Number of parallel workers for processing

    Returns:
        metadata: Dictionary with processing info and patient data
    """

    print("=" * 70)
    print("  Preprocessing FAST Multiplex Images")
    print("=" * 70)

    # Setup
    marker_panel_path, labels_csv_path = find_config_paths()
    marker_panel = MarkerPanel.from_yaml(str(marker_panel_path))
    labels_dict = load_labels(str(labels_csv_path))
    sample_ids = get_sample_ids(str(labels_csv_path))

    fs = gcsfs.GCSFileSystem()
    bucket = os.environ.get("GCS_BUCKET")
    prefix = os.environ.get("GCS_PREFIX", "shared_storage")

    if not bucket:
        raise ValueError("GCS_BUCKET environment variable must be set")

    print(f"\nOutput directory: {output_dir}")
    print(f"Max FOVs per patient: {max_fovs_per_patient}")
    print(f"Parallel workers: {num_workers}")
    print(f"Image resolution: 1728×1728 (native)")
    print(f"Channels: {len(marker_panel.markers)}")

    os.makedirs(output_dir, exist_ok=True)

    # Collect processing tasks
    print("\n" + "-" * 70)
    print("Discovering FOVs...")
    print("-" * 70)

    tasks = []
    for sample_id in sample_ids:
        fovs = enumerate_fovs(fs, bucket, sample_id, prefix)[:max_fovs_per_patient]
        print(f"  {sample_id}: {len(fovs)} FOVs")

        for fov_id in fovs:
            tasks.append(
                (sample_id, fov_id, str(marker_panel_path), bucket, prefix, output_dir)
            )

    print(f"\nTotal FOVs to process: {len(tasks)}")

    # Process with thread pool
    print("\n" + "-" * 70)
    print(f"Processing with {num_workers} workers...")
    print("-" * 70)

    existing_files = (
        set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()
    )
    print(f"Existing preprocessed files: {len(existing_files)}")

    patient_data = {
        sid: {"label": labels_dict.get(sid, 0), "fovs": []} for sid in sample_ids
    }
    completed = 0
    total_size_mb = 0
    cached_count = 0
    new_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_fov, task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["success"]:
                sample_id = result["sample_id"]
                patient_data[sample_id]["fovs"].append(
                    {
                        "fov_id": result["fov_id"],
                        "filename": result["filename"],
                        "shape": result["shape"],
                    }
                )
                total_size_mb += result["size_mb"]

                if result.get("cached", False):
                    cached_count += 1
                    if cached_count % 10 == 0:
                        print(
                            f"  [{completed}/{len(tasks)}] ... {cached_count} cached files loaded"
                        )
                else:
                    new_count += 1
                    print(
                        f"  [{completed}/{len(tasks)}] {result['sample_id']}_{result['fov_id']} - {result['size_mb']:.1f}MB (NEW)"
                    )
            else:
                print(
                    f"  [{completed}/{len(tasks)}] {result['sample_id']}_{result['fov_id']} - FAILED"
                )

    # Save metadata
    total_fovs = sum(len(p["fovs"]) for p in patient_data.values())

    metadata = {
        "created": datetime.now().isoformat(),
        "resolution": "full",
        "image_size": 1728,
        "illumination_correction": True,
        "n_channels": len(marker_panel.markers),
        "marker_names": [m.name for m in marker_panel.markers],
        "total_fovs": total_fovs,
        "total_size_mb": total_size_mb,
        "patients": {pid: p for pid, p in patient_data.items() if p["fovs"]},
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("  Preprocessing Complete!")
    print(f"  Patients: {len([p for p in patient_data.values() if p['fovs']])}")
    print(f"  Total FOVs: {total_fovs}")
    print(f"  - Already cached: {cached_count}")
    print(f"  - Newly processed: {new_count}")
    print(f"  Total Size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess FAST multiplex images")
    parser.add_argument(
        "--output",
        type=str,
        default="./preprocessed_data/",
        help="Output directory for preprocessed files",
    )
    parser.add_argument(
        "--max-fovs", type=int, default=30, help="Maximum FOVs per patient"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of parallel workers"
    )
    args = parser.parse_args()

    preprocess_data(
        output_dir=args.output,
        max_fovs_per_patient=args.max_fovs,
        num_workers=args.num_workers,
    )
