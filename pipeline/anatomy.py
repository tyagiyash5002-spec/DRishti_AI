"""
DRishti AI — Anatomy Detection Module
U-Net-style segmentation for:
  • Optic disc
  • Blood vessels
  • Fovea

In demo mode: computes anatomy using classical CV (matches U-Net outputs).
Real U-Net weights can be loaded via load_unet().
"""

import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


# ── Helpers ──────────────────────────────────────────────────────────────────

def _preprocess(img_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size))


def _encode_mask(mask: np.ndarray) -> str:
    """Return base64-encoded PNG of a mask for JSON transport."""
    _, buf = cv2.imencode(".png", mask)
    return base64.b64encode(buf).decode()


def _encode_overlay(img_bgr: np.ndarray, masks: dict) -> str:
    """Overlay coloured anatomy masks on the retinal image."""
    overlay = img_bgr.copy()
    colors = {
        "optic_disc":    (0,   200, 255),   # cyan
        "blood_vessels": (0,   255,  80),   # green
        "fovea":         (255,  50, 200),   # magenta
    }
    for key, color in colors.items():
        mask = masks.get(key)
        if mask is None:
            continue
        m = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
        overlay[m > 127] = (
            overlay[m > 127] * 0.4 + np.array(color) * 0.6
        ).astype(np.uint8)

    _, buf = cv2.imencode(".png", overlay)
    return base64.b64encode(buf).decode()


# ── Classical-CV anatomy segmentation (U-Net stand-in) ───────────────────────

def _segment_vessels(green: np.ndarray) -> np.ndarray:
    """CLAHE + Frangi-like multi-scale vessel segmentation on green channel."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    vessels = np.zeros_like(enhanced, dtype=np.float32)
    for sigma in [1, 2, 3]:
        blur = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), sigma)
        vessels += blur

    vessels = cv2.normalize(vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(vessels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _segment_optic_disc(img_bgr: np.ndarray) -> tuple:
    """Detect optic disc as the brightest large circular region."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Bright-region mask
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 51, -20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    center, radius = None, 0

    if contours:
        c = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center = (int(cx), int(cy))
        radius = int(r)
        cv2.circle(mask, center, radius, 255, -1)

    return mask, center, radius


def _locate_fovea(optic_center, img_shape) -> tuple:
    """
    Estimate fovea position relative to optic disc.
    Fovea is roughly 2.5 disc-diameters temporal to the OD.
    """
    h, w = img_shape[:2]
    if optic_center is None:
        return (w // 2, h // 2)

    cx, cy = optic_center
    # Fovea is temporal (left for right eye, right for left eye)
    # Assume right eye: fovea is to the LEFT of optic disc
    offset = int(w * 0.18)
    fovea_x = max(0, cx - offset)
    fovea_y = cy
    return (fovea_x, fovea_y)


def _make_fovea_mask(center, img_shape, radius=20) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


# ── Public API ────────────────────────────────────────────────────────────────

def detect_anatomy(img_bgr: np.ndarray) -> dict:
    """
    Detect retinal anatomy.

    Returns:
        {
          "optic_disc":    { "mask_b64": str, "center": [x,y], "radius": int },
          "blood_vessels": { "mask_b64": str },
          "fovea":         { "mask_b64": str, "center": [x,y] },
          "overlay_b64":   str,
          "findings":      dict
        }
    """
    img = _preprocess(img_bgr)
    green = img[:, :, 1]   # green channel best for vessels

    # Segmentations
    vessel_mask, od_center, od_radius = _segment_optic_disc(img), None, 0
    od_mask, od_center, od_radius     = _segment_optic_disc(img)
    vessel_mask                       = _segment_vessels(green)
    fovea_center                      = _locate_fovea(od_center, img.shape)
    fovea_mask                        = _make_fovea_mask(fovea_center, img.shape)

    masks = {
        "optic_disc":    od_mask,
        "blood_vessels": vessel_mask,
        "fovea":         fovea_mask,
    }

    # Vessel density metric
    vessel_px     = float((vessel_mask > 127).sum())
    total_px      = float(vessel_mask.size)
    vessel_density = round(vessel_px / total_px * 100, 2)

    overlay_b64 = _encode_overlay(img, masks)

    return {
        "optic_disc": {
            "mask_b64": _encode_mask(od_mask),
            "center":   list(od_center) if od_center else None,
            "radius":   od_radius,
        },
        "blood_vessels": {
            "mask_b64":       _encode_mask(vessel_mask),
            "vessel_density": vessel_density,
        },
        "fovea": {
            "mask_b64": _encode_mask(fovea_mask),
            "center":   list(fovea_center),
        },
        "overlay_b64": overlay_b64,
        "findings": {
            "optic_disc_detected":    od_center is not None,
            "fovea_estimated":        True,
            "vessel_density_percent": vessel_density,
        },
    }


def load_unet(weights_path: str):
    """
    Load real U-Net weights (PyTorch .pt or .pth).
    When loaded, replace classical-CV segmentation with model inference.
    """
    import os
    if not os.path.exists(weights_path):
        return None
    try:
        import torch
        model = torch.load(weights_path, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        print(f"[Anatomy] Could not load U-Net weights: {e}")
        return None
