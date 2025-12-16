"""
Enhanced preprocessing pipeline for FAST multiplex IF images.

Combines:
1. Quench subtraction (cycle - quench background)
2. Simple illumination correction (vignetting)
3. Percentile normalization
4. Optional alignment QC

This is a simplified version of the full pipeline from Oh et al.
Full pipeline would include BaSiC illumination correction, cell segmentation, etc.
"""

import numpy as np
from typing import Optional, Tuple
from gcsfs import GCSFileSystem

from .quench_subtraction import load_fov_with_quench_subtraction
from .illumination import correct_vignetting_simple
from .preprocessing import percentile_normalize
from .alignment_qc import check_fov_alignment
from .channel_stacker import MarkerPanel


def preprocess_fov_enhanced(
    image: np.ndarray,
    illumination_correction: bool = True,
    illumination_sigma: float = 100.0,
    normalize: bool = True,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """
    Apply enhanced preprocessing to a quench-subtracted FOV.

    Args:
        image: Quench-subtracted image (H, W, C)
        illumination_correction: Whether to apply vignetting correction
        illumination_sigma: Sigma for illumination estimation
        normalize: Whether to apply percentile normalization
        low_percentile: Low percentile for clipping
        high_percentile: High percentile for clipping

    Returns:
        Preprocessed image
    """
    result = image.copy()

    # Step 1: Illumination correction (per-channel vignetting)
    if illumination_correction:
        result = correct_vignetting_simple(result, sigma=illumination_sigma)

    # Step 2: Percentile normalization
    if normalize:
        result = percentile_normalize(result, low_percentile, high_percentile)

    return result


def load_fov_enhanced(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    marker_panel: MarkerPanel,
    prefix: str = "shared_storage",
    check_alignment: bool = False,
    alignment_threshold: float = 0.06,
    illumination_correction: bool = True,
    normalize: bool = True,
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Load FOV with enhanced preprocessing pipeline.

    Pipeline:
    1. (Optional) Check alignment across cycles
    2. Load with quench subtraction
    3. Apply illumination correction
    4. Apply percentile normalization

    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID
        fov_id: FOV ID
        marker_panel: MarkerPanel configuration
        prefix: GCS prefix
        check_alignment: Whether to check DAPI alignment
        alignment_threshold: MI threshold for alignment (0.06 per paper)
        illumination_correction: Whether to correct vignetting
        normalize: Whether to normalize intensities

    Returns:
        Tuple of (image, metadata_dict)
        image is None if FOV fails QC
    """
    metadata = {
        "sample_id": sample_id,
        "fov_id": fov_id,
        "alignment_checked": check_alignment,
        "alignment_passed": None,
        "preprocessing": [],
    }

    # Step 1: Alignment QC (optional)
    if check_alignment:
        is_aligned, align_details = check_fov_alignment(
            fs,
            bucket,
            sample_id,
            fov_id,
            prefix,
            mi_threshold=alignment_threshold,
        )
        metadata["alignment_passed"] = is_aligned
        metadata["alignment_details"] = align_details

        if not is_aligned:
            return None, metadata

    # Step 2: Load with quench subtraction
    try:
        image = load_fov_with_quench_subtraction(
            fs, bucket, sample_id, fov_id, marker_panel, prefix
        )
        metadata["preprocessing"].append("quench_subtraction")
    except Exception as e:
        metadata["error"] = str(e)
        return None, metadata

    # Step 3: Illumination correction
    if illumination_correction:
        image = correct_vignetting_simple(image, sigma=100.0)
        metadata["preprocessing"].append("illumination_correction")

    # Step 4: Percentile normalization
    if normalize:
        image = percentile_normalize(image)
        metadata["preprocessing"].append("percentile_normalization")

    return image, metadata


class EnhancedPreprocessor:
    """
    Stateful preprocessor that can compute statistics across FOVs.

    For more robust illumination correction, we can compute flat-fields
    from multiple FOVs rather than per-FOV estimation.
    """

    def __init__(
        self,
        fs: GCSFileSystem,
        bucket: str,
        marker_panel: MarkerPanel,
        prefix: str = "shared_storage",
    ):
        self.fs = fs
        self.bucket = bucket
        self.marker_panel = marker_panel
        self.prefix = prefix

        # Cached statistics
        self.flatfields = {}  # sample_id -> per-channel flatfields
        self.channel_stats = {}  # Global channel statistics

    def compute_flatfields_for_sample(
        self,
        sample_id: str,
        fov_ids: list[str],
        max_fovs: int = 30,
    ) -> dict:
        """
        Compute flat-fields from multiple FOVs for a sample.

        More robust than per-FOV estimation.
        """
        from .illumination import estimate_flatfield_from_stack
        from .quench_subtraction import load_fov_with_quench_subtraction

        # Sample FOVs
        sampled_fovs = fov_ids[:max_fovs]

        # Load images
        images = []
        for fov_id in sampled_fovs:
            try:
                img = load_fov_with_quench_subtraction(
                    self.fs,
                    self.bucket,
                    sample_id,
                    fov_id,
                    self.marker_panel,
                    self.prefix,
                )
                images.append(img)
            except:
                continue

        if not images:
            return {}

        # Compute per-channel flatfields
        n_channels = images[0].shape[-1]
        flatfields = {}

        for c in range(n_channels):
            channel_images = [img[:, :, c] for img in images]
            flatfields[c] = estimate_flatfield_from_stack(channel_images)

        self.flatfields[sample_id] = flatfields
        return flatfields

    def preprocess_fov(
        self,
        sample_id: str,
        fov_id: str,
        use_cached_flatfield: bool = True,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Preprocess a single FOV using cached flat-fields if available.
        """
        from .quench_subtraction import load_fov_with_quench_subtraction
        from .illumination import apply_flatfield_correction

        metadata = {"sample_id": sample_id, "fov_id": fov_id}

        # Load with quench subtraction
        try:
            image = load_fov_with_quench_subtraction(
                self.fs, self.bucket, sample_id, fov_id, self.marker_panel, self.prefix
            )
        except Exception as e:
            metadata["error"] = str(e)
            return None, metadata

        # Apply illumination correction
        if use_cached_flatfield and sample_id in self.flatfields:
            flatfields = self.flatfields[sample_id]
            corrected = np.zeros_like(image)
            for c in range(image.shape[-1]):
                if c in flatfields:
                    corrected[:, :, c] = apply_flatfield_correction(
                        image[:, :, c], flatfields[c]
                    )
                else:
                    corrected[:, :, c] = image[:, :, c]
            image = corrected
            metadata["illumination"] = "cached_flatfield"
        else:
            image = correct_vignetting_simple(image)
            metadata["illumination"] = "per_fov"

        # Normalize
        image = percentile_normalize(image)

        return image, metadata
