"""
Alignment Quality Control for FAST cyclic immunofluorescence images.

Checks alignment between DAPI images across cycles using mutual information.
FOVs with poor alignment (MI < threshold) should be excluded from analysis.

Reference: Oh et al. - FOVs classified as misaligned if any DAPI MI < 0.06
"""

import numpy as np
from typing import Optional, Tuple
from gcsfs import GCSFileSystem
import tifffile
from sklearn.metrics import mutual_info_score

from .channel_stacker import get_base_sample_id


def compute_mutual_information(
    image1: np.ndarray,
    image2: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Compute mutual information between two images.
    
    Args:
        image1: First image (2D array)
        image2: Second image (2D array)
        bins: Number of bins for histogram
    
    Returns:
        Mutual information score
    """
    # Flatten images
    flat1 = image1.flatten()
    flat2 = image2.flatten()
    
    # Bin the values
    flat1_binned = np.digitize(flat1, np.linspace(flat1.min(), flat1.max(), bins))
    flat2_binned = np.digitize(flat2, np.linspace(flat2.min(), flat2.max(), bins))
    
    # Compute MI
    return mutual_info_score(flat1_binned, flat2_binned)


def load_dapi_image(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    cycle: int,
    prefix: str = "shared_storage",
) -> np.ndarray:
    """
    Load DAPI (w1) image for a specific cycle.
    
    DAPI is always channel w1 in all cycles.
    """
    base_id = get_base_sample_id(sample_id)
    filename = f"{base_id}_cycle{cycle}_w1_{fov_id}_t1.TIF"
    path = f"{bucket}/{prefix}/{sample_id}/cycle{cycle}/{filename}"
    
    with fs.open(path, "rb") as f:
        img = tifffile.imread(f)
    
    if img.ndim > 2:
        img = img.squeeze()
    
    return img.astype(np.float32)


def check_fov_alignment(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    prefix: str = "shared_storage",
    reference_cycle: int = 0,
    cycles_to_check: Optional[list[int]] = None,
    mi_threshold: float = 0.06,
    downsample: int = 4,
) -> Tuple[bool, dict]:
    """
    Check alignment quality for a single FOV by comparing DAPI across cycles.
    
    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID
        fov_id: FOV ID (e.g., "s100")
        prefix: GCS prefix
        reference_cycle: Reference cycle (default: 0)
        cycles_to_check: List of cycles to check (default: 1-7)
        mi_threshold: Minimum MI threshold (default: 0.06 per paper)
        downsample: Downsample factor for speed (default: 4)
    
    Returns:
        Tuple of (is_aligned: bool, details: dict)
        details contains per-cycle MI values
    """
    if cycles_to_check is None:
        cycles_to_check = list(range(1, 8))  # cycles 1-7
    
    # Load reference DAPI
    try:
        ref_dapi = load_dapi_image(fs, bucket, sample_id, fov_id, reference_cycle, prefix)
    except FileNotFoundError:
        return False, {"error": f"Reference DAPI (cycle {reference_cycle}) not found"}
    
    # Downsample for speed
    if downsample > 1:
        ref_dapi = ref_dapi[::downsample, ::downsample]
    
    # Check each cycle
    mi_values = {}
    is_aligned = True
    
    for cycle in cycles_to_check:
        try:
            cycle_dapi = load_dapi_image(fs, bucket, sample_id, fov_id, cycle, prefix)
            
            if downsample > 1:
                cycle_dapi = cycle_dapi[::downsample, ::downsample]
            
            # Compute MI
            mi = compute_mutual_information(ref_dapi, cycle_dapi)
            mi_values[f"cycle{cycle}"] = mi
            
            if mi < mi_threshold:
                is_aligned = False
                
        except FileNotFoundError:
            mi_values[f"cycle{cycle}"] = None
            is_aligned = False
    
    return is_aligned, {
        "mi_values": mi_values,
        "mi_threshold": mi_threshold,
        "min_mi": min(v for v in mi_values.values() if v is not None) if mi_values else None,
    }


def filter_aligned_fovs(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_ids: list[str],
    prefix: str = "shared_storage",
    mi_threshold: float = 0.06,
    verbose: bool = True,
) -> Tuple[list[str], list[str]]:
    """
    Filter FOVs to keep only well-aligned ones.
    
    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID
        fov_ids: List of FOV IDs to check
        prefix: GCS prefix
        mi_threshold: MI threshold for alignment
        verbose: Print progress
    
    Returns:
        Tuple of (aligned_fovs, misaligned_fovs)
    """
    aligned = []
    misaligned = []
    
    for fov_id in fov_ids:
        is_aligned, details = check_fov_alignment(
            fs, bucket, sample_id, fov_id, prefix,
            mi_threshold=mi_threshold,
        )
        
        if is_aligned:
            aligned.append(fov_id)
        else:
            misaligned.append(fov_id)
            if verbose:
                min_mi = details.get("min_mi", "N/A")
                print(f"    Misaligned: {fov_id} (min MI: {min_mi:.4f})" if min_mi else f"    Misaligned: {fov_id}")
    
    return aligned, misaligned


