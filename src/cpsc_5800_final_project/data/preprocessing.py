"""Preprocessing for raw multiplex immunofluorescence images."""

import numpy as np
from typing import Optional, Tuple


def percentile_normalize(
    image: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Percentile-based normalization per channel.

    Args:
        image: (H, W, C) array
        low_percentile: Lower percentile for clipping (default: 1)
        high_percentile: Upper percentile for clipping (default: 99)

    Returns:
        Normalized image with values in [0, 1]
    """
    result = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[-1]):
        channel = image[:, :, c].astype(np.float32)

        p_low = np.percentile(channel, low_percentile)
        p_high = np.percentile(channel, high_percentile)

        # Clip and scale to [0, 1]
        channel = np.clip(channel, p_low, p_high)
        if p_high > p_low:
            channel = (channel - p_low) / (p_high - p_low)
        else:
            channel = np.zeros_like(channel)

        result[:, :, c] = channel

    return result


def background_subtract(
    image: np.ndarray,
    background_percentile: float = 5.0,
) -> np.ndarray:
    """
    Simple background subtraction per channel.

    Estimates background as a low percentile value and subtracts it.

    Args:
        image: (H, W, C) array
        background_percentile: Percentile to use as background estimate

    Returns:
        Background-subtracted image (non-negative)
    """
    result = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[-1]):
        channel = image[:, :, c].astype(np.float32)
        background = np.percentile(channel, background_percentile)
        channel = channel - background
        channel = np.maximum(channel, 0)  # No negative values
        result[:, :, c] = channel

    return result


def preprocess_fov(
    image: np.ndarray,
    background_percentile: float = 5.0,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single FOV.

    Steps:
    1. Background subtraction (per channel)
    2. Percentile normalization (per channel)

    Args:
        image: (H, W, C) raw image array
        background_percentile: Percentile for background estimation
        low_percentile: Lower percentile for clipping
        high_percentile: Upper percentile for clipping

    Returns:
        Preprocessed image with values in [0, 1]
    """
    # Step 1: Background subtraction
    image = background_subtract(image, background_percentile)

    # Step 2: Percentile normalization
    image = percentile_normalize(image, low_percentile, high_percentile)

    return image


def compute_channel_stats(
    images: list[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global channel statistics across multiple images.

    Useful for dataset-wide normalization.

    Args:
        images: List of (H, W, C) arrays

    Returns:
        means: (C,) array of per-channel means
        stds: (C,) array of per-channel standard deviations
    """
    if not images:
        raise ValueError("No images provided")

    n_channels = images[0].shape[-1]

    # Accumulate stats
    channel_values = [[] for _ in range(n_channels)]

    for img in images:
        for c in range(n_channels):
            # Sample to avoid memory issues
            flat = img[:, :, c].flatten()
            sampled = flat[::100]  # Take every 100th pixel
            channel_values[c].extend(sampled)

    means = np.array([np.mean(v) for v in channel_values])
    stds = np.array([np.std(v) for v in channel_values])

    return means, stds


def zscore_normalize(
    image: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    clip_range: float = 3.0,
) -> np.ndarray:
    """
    Z-score normalization using pre-computed statistics.

    Args:
        image: (H, W, C) array
        means: (C,) per-channel means
        stds: (C,) per-channel stds
        clip_range: Clip to +/- this many standard deviations

    Returns:
        Normalized image (approximately zero mean, unit variance per channel)
    """
    result = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[-1]):
        channel = image[:, :, c].astype(np.float32)

        if stds[c] > 0:
            channel = (channel - means[c]) / stds[c]
            channel = np.clip(channel, -clip_range, clip_range)
        else:
            channel = np.zeros_like(channel)

        result[:, :, c] = channel

    return result
