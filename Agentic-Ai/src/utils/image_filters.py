"""
image_filters.py  –  Phase 5 OpenCV-Based Image Filter Library

Provides filters that the edit agent can apply to scene images:
  - brightness, contrast, sepia, blur, sharpen, vignette, grayscale
  - apply_filter_chain() for composing multiple filters

Each function reads, transforms, and saves the image, returning the path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


def _ensure_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python-headless is required for image filters. "
            "Install with: pip install opencv-python-headless"
        )


def _read_image(image_path: str | Path) -> "np.ndarray":
    _ensure_cv2()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img


def _save_image(image: "np.ndarray", image_path: str | Path) -> str:
    _ensure_cv2()
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


# ── Individual Filters ───────────────────────────────────────────────────────

def apply_brightness(
    image_path: str | Path,
    factor: float = 1.3,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Adjust image brightness.

    Parameters
    ----------
    factor : float
        > 1.0 = brighter, < 1.0 = darker.  e.g. 0.7 = 30% darker, 1.3 = 30% brighter
    """
    img = _read_image(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _save_image(result, output_path or image_path)


def apply_contrast(
    image_path: str | Path,
    factor: float = 1.5,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Adjust image contrast.

    Parameters
    ----------
    factor : float
        > 1.0 = higher contrast, < 1.0 = lower contrast.
    """
    img = _read_image(image_path)
    mean = np.mean(img)
    result = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255)
    return _save_image(result.astype(np.uint8), output_path or image_path)


def apply_sepia(
    image_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """Apply warm sepia/vintage tone."""
    img = _read_image(image_path)
    # Sepia transformation matrix
    kernel = np.array([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189],
    ])
    result = cv2.transform(img, kernel)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return _save_image(result, output_path or image_path)


def apply_blur(
    image_path: str | Path,
    kernel_size: int = 15,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Apply Gaussian blur.

    Parameters
    ----------
    kernel_size : int
        Must be odd.  Larger = more blur.  Default 15.
    """
    img = _read_image(image_path)
    # Ensure kernel size is odd
    k = max(3, kernel_size)
    if k % 2 == 0:
        k += 1
    result = cv2.GaussianBlur(img, (k, k), 0)
    return _save_image(result, output_path or image_path)


def apply_sharpen(
    image_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """Apply unsharp mask sharpening."""
    img = _read_image(image_path)
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ], dtype=np.float32)
    result = cv2.filter2D(img, -1, kernel)
    return _save_image(result, output_path or image_path)


def apply_vignette(
    image_path: str | Path,
    strength: float = 0.7,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Apply vignette effect (darkened edges).

    Parameters
    ----------
    strength : float
        0.0 = no vignette, 1.0 = maximum darkness at edges.
    """
    img = _read_image(image_path)
    rows, cols = img.shape[:2]

    # Create Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, cols * 0.4)
    kernel_y = cv2.getGaussianKernel(rows, rows * 0.4)
    kernel = kernel_y * kernel_x.T

    # Normalise
    mask = kernel / kernel.max()
    mask = mask * strength + (1 - strength)

    # Apply
    result = img.copy().astype(np.float32)
    for i in range(3):
        result[:, :, i] = result[:, :, i] * mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    return _save_image(result, output_path or image_path)


def apply_grayscale(
    image_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """Convert to grayscale (black & white)."""
    img = _read_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert back to 3-channel for consistency
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _save_image(result, output_path or image_path)


# ── Filter Chain ─────────────────────────────────────────────────────────────

FILTER_REGISTRY: Dict[str, Any] = {
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "sepia": apply_sepia,
    "blur": apply_blur,
    "sharpen": apply_sharpen,
    "vignette": apply_vignette,
    "grayscale": apply_grayscale,
}


def apply_filter_chain(
    image_path: str | Path,
    filters: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Apply a sequence of filters to an image.

    Parameters
    ----------
    filters : list of dict
        Each dict has {"name": "sepia", ...optional kwargs}.
        Example: [{"name": "sepia"}, {"name": "brightness", "factor": 0.8}]
    output_path : optional
        If provided, write the final result here instead of overwriting.

    Returns
    -------
    str
        Path to the output image.
    """
    current_path = str(image_path)

    for i, filt in enumerate(filters):
        name = filt.get("name", "").lower()
        func = FILTER_REGISTRY.get(name)
        if func is None:
            raise ValueError(f"Unknown filter: {name!r}. Available: {list(FILTER_REGISTRY)}")

        kwargs = {k: v for k, v in filt.items() if k != "name"}

        # Only set output_path on the last filter if a custom path was given
        if output_path and i == len(filters) - 1:
            kwargs["output_path"] = output_path

        current_path = func(current_path, **kwargs)

    return current_path


def get_available_filters() -> List[str]:
    """Return list of available filter names."""
    return list(FILTER_REGISTRY.keys())
