"""
Quench subtraction for FAST cyclic immunofluorescence images.

Implements background subtraction as described in the paper:
- cycle1 markers: subtract cycle0 (unstained reference)
- cycle2-7 markers: subtract corresponding quench image (quench1-6)

Reference: Oh et al., "Spatial analysis identifies DC niches..."
"""

import numpy as np
from typing import Optional
from gcsfs import GCSFileSystem
import tifffile

from .channel_stacker import MarkerPanel, Marker, get_base_sample_id


def get_quench_path(
    sample_id: str,
    fov_id: str,
    cycle: int,
    channel: int,
    prefix: str = "shared_storage",
) -> str:
    """
    Get GCS path for the quench/reference image to subtract.

    Args:
        sample_id: Sample ID (folder name)
        fov_id: FOV ID (e.g., "s100")
        cycle: Marker cycle (1-7)
        channel: Channel number (2-4 for w2-w4)
        prefix: GCS prefix

    Returns:
        GCS path to quench/reference image

    Mapping:
        cycle 1 → subtract cycle0 (unstained)
        cycle 2 → subtract quench1
        cycle 3 → subtract quench2
        ...
        cycle 7 → subtract quench6
    """
    base_id = get_base_sample_id(sample_id)

    if cycle == 1:
        # Subtract cycle0 (unstained reference)
        folder = "cycle0"
        filename = f"{base_id}_cycle0_w{channel}_{fov_id}_t1.TIF"
    else:
        # Subtract previous quench
        quench_num = cycle - 1
        folder = f"quench{quench_num}"
        filename = f"{base_id}_quench{quench_num}_w{channel}_{fov_id}_t1.TIF"

    return f"{prefix}/{sample_id}/{folder}/{filename}"


def load_fov_with_quench_subtraction(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    marker_panel: MarkerPanel,
    prefix: str = "shared_storage",
    clip_negative: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load FOV with proper quench subtraction.

    For each marker channel:
    1. Load the stained image (cycle N)
    2. Load the corresponding quench/reference image
    3. Subtract: stained - quench
    4. Clip negative values to 0 (if clip_negative=True)

    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID (folder name)
        fov_id: FOV ID (e.g., "s100")
        marker_panel: MarkerPanel configuration
        prefix: GCS prefix
        clip_negative: Whether to clip negative values to 0
        dtype: Output dtype

    Returns:
        Quench-subtracted image array of shape (H, W, 21)
    """
    base_id = get_base_sample_id(sample_id)
    channels = []
    height = None
    width = None

    for marker in marker_panel.markers:
        # Path to stained image
        stained_filename = (
            f"{base_id}_cycle{marker.cycle}_w{marker.channel}_{fov_id}_t1.TIF"
        )
        stained_path = (
            f"{bucket}/{prefix}/{sample_id}/cycle{marker.cycle}/{stained_filename}"
        )

        # Path to quench/reference image
        quench_rel_path = get_quench_path(
            sample_id, fov_id, marker.cycle, marker.channel, prefix
        )
        quench_path = f"{bucket}/{quench_rel_path}"

        try:
            # Load stained image
            with fs.open(stained_path, "rb") as f:
                stained = tifffile.imread(f).astype(dtype)
            if stained.ndim > 2:
                stained = stained.squeeze()

            # Load quench/reference image
            with fs.open(quench_path, "rb") as f:
                quench = tifffile.imread(f).astype(dtype)
            if quench.ndim > 2:
                quench = quench.squeeze()

            # Verify dimensions match
            if stained.shape != quench.shape:
                raise ValueError(
                    f"Shape mismatch: stained {stained.shape} vs quench {quench.shape} "
                    f"for marker {marker.name}"
                )

            # Quench subtraction
            subtracted = stained - quench

            # Clip negative values
            if clip_negative:
                subtracted = np.maximum(subtracted, 0)

            # Track dimensions
            if height is None:
                height, width = subtracted.shape
            elif subtracted.shape != (height, width):
                raise ValueError(
                    f"Inconsistent dimensions: expected ({height}, {width}), "
                    f"got {subtracted.shape} for {marker.name}"
                )

            channels.append(subtracted)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Missing file for {marker.name} (cycle {marker.cycle}, w{marker.channel}): {e}"
            )

    # Stack channels: (H, W, 21)
    return np.stack(channels, axis=-1)


def load_fov_raw_and_quench(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    marker_panel: MarkerPanel,
    prefix: str = "shared_storage",
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both raw stained and quench images for a FOV.

    Useful for debugging or custom preprocessing.

    Returns:
        Tuple of (stained_image, quench_image), each shape (H, W, 21)
    """
    base_id = get_base_sample_id(sample_id)
    stained_channels = []
    quench_channels = []

    for marker in marker_panel.markers:
        # Stained
        stained_filename = (
            f"{base_id}_cycle{marker.cycle}_w{marker.channel}_{fov_id}_t1.TIF"
        )
        stained_path = (
            f"{bucket}/{prefix}/{sample_id}/cycle{marker.cycle}/{stained_filename}"
        )

        with fs.open(stained_path, "rb") as f:
            stained = tifffile.imread(f).astype(dtype)
        if stained.ndim > 2:
            stained = stained.squeeze()
        stained_channels.append(stained)

        # Quench
        quench_rel_path = get_quench_path(
            sample_id, fov_id, marker.cycle, marker.channel, prefix
        )
        quench_path = f"{bucket}/{quench_rel_path}"

        with fs.open(quench_path, "rb") as f:
            quench = tifffile.imread(f).astype(dtype)
        if quench.ndim > 2:
            quench = quench.squeeze()
        quench_channels.append(quench)

    return np.stack(stained_channels, axis=-1), np.stack(quench_channels, axis=-1)
