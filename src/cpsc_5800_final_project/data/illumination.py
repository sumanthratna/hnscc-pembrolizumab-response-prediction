"""
Simple illumination correction for multiplex IF images.

Implements a simplified flat-field correction without requiring BaSiC.
Uses percentile-based estimation of the illumination pattern.

For proper illumination correction, BaSiC (https://github.com/peng-lab/BaSiCPy)
should be used, but this provides a reasonable approximation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional


def estimate_flatfield_single(
    image: np.ndarray,
    sigma: float = 50.0,
    percentile: float = 50.0,
) -> np.ndarray:
    """
    Estimate flat-field from a single image using Gaussian smoothing.
    
    This is a simplified approach - proper BaSiC uses multiple images.
    
    Args:
        image: 2D image array
        sigma: Gaussian smoothing sigma (larger = more smoothing)
        percentile: Percentile for normalization
    
    Returns:
        Estimated flat-field (same shape as input)
    """
    # Heavy Gaussian smoothing to get low-frequency illumination pattern
    smoothed = gaussian_filter(image.astype(np.float64), sigma=sigma)
    
    # Normalize so mean is 1
    if smoothed.mean() > 0:
        smoothed = smoothed / smoothed.mean()
    else:
        smoothed = np.ones_like(smoothed)
    
    return smoothed.astype(np.float32)


def estimate_flatfield_from_stack(
    images: list[np.ndarray],
    method: str = "median",
    sigma: float = 50.0,
) -> np.ndarray:
    """
    Estimate flat-field from multiple images (more robust).
    
    Args:
        images: List of 2D images (same channel, different FOVs)
        method: "median" or "mean" for combining images
        sigma: Gaussian smoothing sigma
    
    Returns:
        Estimated flat-field
    """
    if not images:
        raise ValueError("No images provided")
    
    # Stack images
    stack = np.stack(images, axis=0)
    
    # Combine
    if method == "median":
        combined = np.median(stack, axis=0)
    else:
        combined = np.mean(stack, axis=0)
    
    # Smooth to get illumination pattern
    smoothed = gaussian_filter(combined.astype(np.float64), sigma=sigma)
    
    # Normalize
    if smoothed.mean() > 0:
        smoothed = smoothed / smoothed.mean()
    else:
        smoothed = np.ones_like(smoothed)
    
    return smoothed.astype(np.float32)


def apply_flatfield_correction(
    image: np.ndarray,
    flatfield: np.ndarray,
    clip_min: float = 0.0,
) -> np.ndarray:
    """
    Apply flat-field correction to an image.
    
    Corrected = Image / Flatfield
    
    Args:
        image: Input image (2D or 3D with channels last)
        flatfield: Flat-field estimate (2D, same H×W as image)
        clip_min: Minimum value after correction
    
    Returns:
        Corrected image
    """
    # Avoid division by zero
    flatfield_safe = np.maximum(flatfield, 0.01)
    
    if image.ndim == 2:
        corrected = image / flatfield_safe
    elif image.ndim == 3:
        # Apply same flatfield to all channels
        corrected = image / flatfield_safe[:, :, np.newaxis]
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    
    # Clip negative values
    if clip_min is not None:
        corrected = np.maximum(corrected, clip_min)
    
    return corrected.astype(np.float32)


def correct_vignetting_simple(
    image: np.ndarray,
    sigma: float = 100.0,
) -> np.ndarray:
    """
    Simple vignetting correction using per-image flat-field estimate.
    
    This is a quick approximation - for better results, use
    estimate_flatfield_from_stack with multiple images.
    
    Args:
        image: Input image (H, W) or (H, W, C)
        sigma: Smoothing sigma for flat-field estimation
    
    Returns:
        Vignetting-corrected image
    """
    if image.ndim == 2:
        flatfield = estimate_flatfield_single(image, sigma=sigma)
        return apply_flatfield_correction(image, flatfield)
    
    elif image.ndim == 3:
        # Correct each channel separately
        corrected_channels = []
        for c in range(image.shape[-1]):
            channel = image[:, :, c]
            flatfield = estimate_flatfield_single(channel, sigma=sigma)
            corrected = apply_flatfield_correction(channel, flatfield)
            corrected_channels.append(corrected)
        return np.stack(corrected_channels, axis=-1)
    
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")


def compute_radial_flatfield(
    shape: tuple[int, int],
    falloff: float = 0.3,
) -> np.ndarray:
    """
    Create a synthetic radial flat-field (vignetting pattern).
    
    Useful for correction when you know the vignetting is radial
    but don't have calibration images.
    
    Args:
        shape: (height, width) of output
        falloff: Vignetting falloff factor (0 = no vignetting, 1 = strong)
    
    Returns:
        Synthetic flat-field (1.0 at center, decreasing toward edges)
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    
    # Center coordinates
    cy, cx = h / 2, w / 2
    
    # Normalized distance from center (0 at center, 1 at corners)
    max_dist = np.sqrt(cy**2 + cx**2)
    dist = np.sqrt((y - cy)**2 + (x - cx)**2) / max_dist
    
    # Vignetting pattern (cos^4 approximation)
    flatfield = 1.0 - falloff * dist**2
    
    return flatfield.astype(np.float32)


