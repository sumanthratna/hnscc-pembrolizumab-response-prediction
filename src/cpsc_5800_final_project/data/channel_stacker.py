"""Load and stack multi-channel FAST images from GCS into 21-channel tensors."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re
import yaml
import numpy as np
import tifffile
import gcsfs
from gcsfs import GCSFileSystem


def get_base_sample_id(sample_id: str) -> str:
    """
    Extract base sample ID from folder name.

    The folder name may include suffixes like _reimage, _section2, _1, _2, etc.
    but the actual files inside use the base ID (e.g., PIO9, PIO26, PIO51).

    Examples:
        PIO9_1_reimage -> PIO9
        PIO26_reimage -> PIO26
        PIO51_section2_reimage -> PIO51
        PIO1 -> PIO1
    """
    # Match pattern: PIO followed by digits, then optional suffixes
    match = re.match(r"^(PIO\d+)", sample_id)
    if match:
        return match.group(1)
    return sample_id


@dataclass
class Marker:
    """Represents a single marker in the panel."""

    name: str
    cycle: int
    channel: int  # 1-4 corresponding to w1-w4
    index: int  # Position in final 21-channel tensor (0-20)


@dataclass
class MarkerPanel:
    """Marker panel configuration loaded from YAML."""

    markers: List[Marker]

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MarkerPanel":
        """Load marker panel from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        markers = [
            Marker(
                name=m["name"],
                cycle=m["cycle"],
                channel=m["channel"],
                index=m["index"],
            )
            for m in config["markers"]
        ]

        # Sort by index to ensure correct ordering
        markers.sort(key=lambda m: m.index)

        return cls(markers=markers)

    def get_gcs_path(
        self,
        sample_id: str,
        fov_id: str,
        marker: Marker,
        bucket: str,
        prefix: str,
    ) -> str:
        """
        Construct GCS path for a specific marker channel.

        Args:
            sample_id: Sample ID (folder name, e.g., "PIO9_1_reimage")
            fov_id: FOV ID (e.g., "s100")
            marker: Marker object
            bucket: GCS bucket name
            prefix: GCS prefix (e.g., "shared_storage")

        Returns:
            GCS path string (e.g., "gs://bucket/shared_storage/PIO9_1_reimage/cycle0/PIO9_cycle0_w1_s100_t1.TIF")
        """
        # Files use base sample ID (PIO9), but folder uses full name (PIO9_1_reimage)
        base_id = get_base_sample_id(sample_id)
        # Format: {base_id}_cycle{cycle}_w{channel}_{fov}_t1.TIF
        filename = f"{base_id}_cycle{marker.cycle}_w{marker.channel}_{fov_id}_t1.TIF"
        return f"{prefix}/{sample_id}/cycle{marker.cycle}/{filename}"


def enumerate_fovs(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    prefix: str = "shared_storage",
    cycle: int = 1,  # Use cycle 1 (first marker cycle) instead of cycle 0
    channel: int = 2,  # Use w2 (first marker channel) instead of w1 (DAPI)
) -> List[str]:
    """
    Enumerate all FOV IDs for a sample by scanning GCS.

    Uses cycle 1, channel 2 (w2) as reference to find FOVs with marker data.
    (Cycle 0 is just DAPI and may have more FOVs than marker cycles)

    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID (folder name, e.g., "PIO9_1_reimage")
        prefix: GCS prefix
        cycle: Cycle number to use for enumeration (default: 0)
        channel: Channel number to use for enumeration (default: 1, w1)

    Returns:
        List of FOV IDs (e.g., ["s1", "s2", ..., "s200"])
    """
    # List files in the cycle directory (use full sample_id for folder)
    cycle_dir = f"{prefix}/{sample_id}/cycle{cycle}/"

    try:
        files = fs.ls(f"{bucket}/{cycle_dir}", detail=False)
    except FileNotFoundError:
        return []

    # Extract FOV IDs from filenames
    # Files use base sample ID (e.g., PIO9), not full folder name (PIO9_1_reimage)
    base_id = get_base_sample_id(sample_id)
    # Format: {base_id}_cycle{cycle}_w{channel}_{fov}_t1.TIF
    pattern = f"{base_id}_cycle{cycle}_w{channel}_"

    fov_ids = set()

    for file_path in files:
        filename = Path(file_path).name
        if filename.startswith(pattern) and filename.endswith("_t1.TIF"):
            # Extract FOV ID (e.g., "s100" from "PIO9_cycle0_w1_s100_t1.TIF")
            suffix = filename[len(pattern) :]
            fov_id = suffix.replace("_t1.TIF", "")
            fov_ids.add(fov_id)

    # Sort FOV IDs numerically (s1, s2, ..., s10, s11, ...)
    def fov_sort_key(fov: str) -> int:
        # Extract number from "s100" -> 100
        return int(fov[1:]) if fov.startswith("s") else 0

    return sorted(fov_ids, key=fov_sort_key)


def load_fov_channels(
    fs: GCSFileSystem,
    bucket: str,
    sample_id: str,
    fov_id: str,
    marker_panel: MarkerPanel,
    prefix: str = "shared_storage",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load all 21 channels for a single FOV and stack into (H, W, 21) array.

    Args:
        fs: GCSFileSystem instance
        bucket: GCS bucket name
        sample_id: Sample ID (e.g., "PIO1")
        fov_id: FOV ID (e.g., "s100")
        marker_panel: MarkerPanel configuration
        prefix: GCS prefix
        dtype: Output dtype (default: float32)

    Returns:
        Stacked image array of shape (H, W, 21) with channels in marker index order

    Raises:
        FileNotFoundError: If any required channel file is missing
        ValueError: If image dimensions are inconsistent across channels
    """
    channels = []
    height = None
    width = None

    for marker in marker_panel.markers:
        # Get GCS path for this marker
        gcs_path = marker_panel.get_gcs_path(sample_id, fov_id, marker, bucket, prefix)
        full_path = f"{bucket}/{gcs_path}"

        try:
            # Read TIF from GCS
            with fs.open(full_path, "rb") as f:
                img = tifffile.imread(f)

            # Ensure 2D (handle any extra dimensions)
            if img.ndim > 2:
                img = img.squeeze()
            if img.ndim != 2:
                raise ValueError(f"Expected 2D image, got {img.ndim}D for {gcs_path}")

            # Check dimensions are consistent
            if height is None:
                height, width = img.shape
            elif img.shape != (height, width):
                raise ValueError(
                    f"Inconsistent image dimensions: expected ({height}, {width}), "
                    f"got {img.shape} for {gcs_path}"
                )

            channels.append(img.astype(dtype))

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Missing channel file: {full_path} "
                f"(marker: {marker.name}, cycle: {marker.cycle}, channel: {marker.channel})"
            )

    # Stack channels along last axis: (H, W, 21)
    stacked = np.stack(channels, axis=-1)

    return stacked
