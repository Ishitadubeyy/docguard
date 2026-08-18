"""OCR-oriented image preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from src.preprocessing.config import PreprocessingConfig


def preprocess_for_ocr(
    image: np.ndarray,
    config: PreprocessingConfig | None = None,
) -> np.ndarray:
    """Apply configured preprocessing steps suitable for OCR."""
    cfg = config or PreprocessingConfig()
    cv2 = _require_opencv()

    if image.ndim == 2:
        working = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        working = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        working = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported image shape for preprocessing: {image.shape}")

    if cfg.deskew:
        working = _deskew(working, cv2, cfg.deskew_min_angle)

    if cfg.grayscale:
        working = _to_grayscale(working, cv2)

    if cfg.resize:
        working = _resize_if_needed(working, cv2, cfg.max_dimension)

    if cfg.contrast_enhancement:
        working = _enhance_contrast(working, cv2)

    if cfg.denoise:
        working = _denoise(working, cv2)

    if cfg.threshold:
        working = _apply_threshold(working, cv2, cfg.threshold_method)

    return working


def _require_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for preprocessing. Install with: pip install opencv-python"
        ) from exc
    return cv2


def _to_grayscale(image: np.ndarray, cv2) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_if_needed(image: np.ndarray, cv2, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image

    scale = max_dimension / float(longest)
    new_size = (max(int(width * scale), 1), max(int(height * scale), 1))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _enhance_contrast(image: np.ndarray, cv2) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, alpha, beta = cv2.split(lab)
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, alpha, beta))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _denoise(image: np.ndarray, cv2) -> np.ndarray:
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, h=10)
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def _apply_threshold(image: np.ndarray, cv2, method: str) -> np.ndarray:
    gray = _to_grayscale(image, cv2)
    if method == "otsu":
        _, thresholded = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return thresholded

    if method != "adaptive":
        raise ValueError(f"Unsupported threshold method: {method}")

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def _deskew(image: np.ndarray, cv2, min_angle: float) -> np.ndarray:
    gray = _to_grayscale(image, cv2)
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = -angle

    if abs(angle) < min_angle:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
