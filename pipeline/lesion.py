"""
DRishti AI — Lesion Detection Module
FPN-style segmentation for:
  • Microaneurysms   (MA)
  • Hemorrhages      (HE)
  • Hard exudates    (EX)
  • Soft exudates    (SE / cotton-wool spots)

In demo mode: classical CV detection tuned to match FPN outputs.
Real FPN weights can be loaded via load_fpn().
"""

import numpy as np
import cv2
import base64


# ── Colour palette for overlay ────────────────────────────────────────────────
LESION_COLORS = {
    "microaneurysms": (0,   0,   255),   # red
    "hemorrhages":    (0,   50,  200),   # dark red
    "hard_exudates":  (0,   220, 255),   # yellow
    "soft_exudates":  (200, 200, 255),   # light yellow-white
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_img(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def _preprocess(img_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    return cv2.resize(img_bgr, (size, size))


# ── Classical-CV lesion detectors (FPN stand-ins) ─────────────────────────────

def _detect_microaneurysms(green: np.ndarray) -> np.ndarray:
    """
    MAs: small dark dots on the green channel.
    CLAHE + top-hat morphology + blob detection.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(green)

    # Top-hat for small dark spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel)

    _, binary = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)

    # Keep only small blobs (MAs are tiny)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 2 <= area <= 80:
            mask[labels == i] = 255
    return mask


def _detect_hemorrhages(green: np.ndarray) -> np.ndarray:
    """
    Hemorrhages: larger dark red blobs.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq    = clahe.apply(green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    tophat = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, kernel)

    _, binary = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 80 < area <= 1500:
            mask[labels == i] = 255
    return mask


def _detect_hard_exudates(img_bgr: np.ndarray) -> np.ndarray:
    """
    Hard exudates: bright yellowish deposits.
    Use LAB colour space (high L, shifted a/b).
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l_ch)

    _, binary = cv2.threshold(l_eq, 210, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 <= area <= 2000:
            mask[labels == i] = 255
    return mask


def _detect_soft_exudates(img_bgr: np.ndarray) -> np.ndarray:
    """
    Soft exudates (cotton-wool spots): fluffy bright white regions.
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (9, 9), 0)

    _, binary = cv2.threshold(blur, 220, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 100 <= area <= 5000:
            mask[labels == i] = 255
    return mask


def _count_quadrant_hemorrhages(mask: np.ndarray) -> list:
    """
    Split mask into 4 quadrants and count connected hemorrhage blobs per quadrant.
    Used by ECDE for the 4-2-1 clinical evidence rule.
    """
    h, w   = mask.shape
    quads  = [
        mask[:h//2, :w//2],   # top-left
        mask[:h//2, w//2:],   # top-right
        mask[h//2:, :w//2],   # bottom-left
        mask[h//2:, w//2:],   # bottom-right
    ]
    counts = []
    for q in quads:
        n, _, stats, _ = cv2.connectedComponentsWithStats(q)
        # Exclude background (label 0)
        counts.append(max(0, n - 1))
    return counts


def _build_overlay(img_bgr: np.ndarray, masks: dict) -> np.ndarray:
    overlay = img_bgr.copy()
    for name, color in LESION_COLORS.items():
        m = masks.get(name)
        if m is None:
            continue
        m_rs = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]))
        overlay[m_rs > 127] = (
            overlay[m_rs > 127] * 0.35 + np.array(color) * 0.65
        ).astype(np.uint8)
    return overlay


def _build_single_overlay(img_bgr: np.ndarray, mask: np.ndarray, color: tuple, label: str) -> np.ndarray:
    overlay = img_bgr.copy()
    if mask is not None and mask.size > 0:
        m_rs = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]))
        hit = m_rs > 127
        overlay[hit] = (overlay[hit] * 0.25 + np.array(color) * 0.75).astype(np.uint8)
        
        # Crisp outlines & target markers
        contours, _ = cv2.findContours(m_rs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 2:
                cv2.drawContours(overlay, [cnt], -1, color, 2, cv2.LINE_AA)
                (x, y), r = cv2.minEnclosingCircle(cnt)
                cv2.circle(overlay, (int(x), int(y)), int(max(r + 3, 5)), color, 1, cv2.LINE_AA)
    return overlay


def _extract_coordinates(masks: dict) -> dict:
    coords = {}
    limits = {
        "microaneurysms": 60,
        "hemorrhages": 45,
        "hard_exudates": 35,
        "soft_exudates": 25,
    }
    for name, mask in masks.items():
        items = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            cx, cy = centroids[i]
            items.append({
                "x": round(float(cx), 1),
                "y": round(float(cy), 1),
                "w": int(bw),
                "h": int(bh),
                "r": max(4, min(38, int(np.sqrt(area / np.pi) * 1.4))),
                "area": int(area)
            })
        # Prioritize most prominent lesions
        items.sort(key=lambda it: it["area"], reverse=True)
        max_cnt = limits.get(name, 50)
        coords[name] = items[:max_cnt]
    return coords


# ── Public API ────────────────────────────────────────────────────────────────

def detect_lesions(img_bgr: np.ndarray) -> dict:
    """
    Detect DR lesions in the retinal image.

    Returns:
        {
          "masks":          { lesion_type: mask_b64 },
          "counts":         { lesion_type: int },
          "overlay_b64":    str,
          "quadrant_hemorrhages": [int×4],
          "coordinates":    { lesion_type: list[dict] },
          "findings":       dict
        }
    """
    img   = _preprocess(img_bgr)
    green = img[:, :, 1]

    ma_mask = _detect_microaneurysms(green)
    he_mask = _detect_hemorrhages(green)
    ex_mask = _detect_hard_exudates(img)
    se_mask = _detect_soft_exudates(img)

    masks = {
        "microaneurysms": ma_mask,
        "hemorrhages":    he_mask,
        "hard_exudates":  ex_mask,
        "soft_exudates":  se_mask,
    }

    def _count(m): return int(cv2.connectedComponentsWithStats(m)[0]) - 1

    counts = {k: max(0, _count(v)) for k, v in masks.items()}

    quad_he = _count_quadrant_hemorrhages(he_mask)
    coordinates = _extract_coordinates(masks)

    overlay = _build_overlay(img, masks)

    # Build individual overlays for dedicated visualization tabs
    single_overlays = {
        "microaneurysms": _encode_img(_build_single_overlay(img, ma_mask, (0, 0, 255), "MA")),
        "hemorrhages":    _encode_img(_build_single_overlay(img, he_mask, (20, 20, 200), "HE")),
        "hard_exudates":  _encode_img(_build_single_overlay(img, ex_mask, (0, 230, 255), "EX")),
        "soft_exudates":  _encode_img(_build_single_overlay(img, se_mask, (240, 240, 240), "SE")),
    }

    return {
        "masks": {k: _encode_img(v) for k, v in masks.items()},
        "counts":  counts,
        "overlay_b64": _encode_img(overlay),
        "overlays": single_overlays,
        "quadrant_hemorrhages": quad_he,
        "coordinates": coordinates,
        "findings": {
            "microaneurysms_detected": counts["microaneurysms"] > 0,
            "hemorrhages_detected":    counts["hemorrhages"]    > 0,
            "hard_exudates_detected":  counts["hard_exudates"]  > 0,
            "soft_exudates_detected":  counts["soft_exudates"]  > 0,
            "total_lesion_count":      sum(counts.values()),
        },
    }


def load_fpn(weights_path: str):
    """Load real FPN weights (PyTorch). Returns None if unavailable."""
    import os
    if not os.path.exists(weights_path):
        return None
    try:
        import torch
        model = torch.load(weights_path, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        print(f"[Lesion] Could not load FPN weights: {e}")
        return None
