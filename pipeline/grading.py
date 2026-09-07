"""
DRishti AI — Disease Grading Module
Integrates:
  • ResNet-50  (PyTorch torchvision)
  • EfficientNet-B4 (PyTorch timm or torchvision)

Both models output a 5-class probability distribution:
  0: No DR
  1: Mild NPDR
  2: Moderate NPDR
  3: Severe NPDR
  4: Proliferative DR

MC Dropout: 25 inference passes with dropout active for uncertainty.
"""

import numpy as np
import cv2
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io

# ── Constants ─────────────────────────────────────────────────────────────────
DR_CLASSES   = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
NUM_CLASSES  = 5
MC_RUNS      = 25
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Image preprocessing ───────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    pil   = _bgr_to_pil(img_bgr)
    tensor = _transform(pil).unsqueeze(0).to(DEVICE)
    return tensor


# ── Model builders ────────────────────────────────────────────────────────────

def _build_resnet50(pretrained: bool = True) -> nn.Module:
    """ResNet-50 with 5-class head."""
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(m.fc.in_features, NUM_CLASSES),
    )
    return m.to(DEVICE)


def _build_efficientnet_b4(pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 with 5-class head."""
    try:
        import timm
        m = timm.create_model("efficientnet_b4", pretrained=pretrained,
                              num_classes=NUM_CLASSES, drop_rate=0.4)
        return m.to(DEVICE)
    except ImportError:
        # Fallback: use torchvision EfficientNet-B4
        m = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        in_feat = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_feat, NUM_CLASSES),
        )
        return m.to(DEVICE)


# ── Global model singletons (lazy-loaded) ─────────────────────────────────────
_resnet50       = None
_efficientnet   = None


def _get_resnet50():
    global _resnet50
    if _resnet50 is None:
        _resnet50 = _build_resnet50(pretrained=True)
        _resnet50.eval()
    return _resnet50


def _get_efficientnet():
    global _efficientnet
    if _efficientnet is None:
        _efficientnet = _build_efficientnet_b4(pretrained=True)
        _efficientnet.eval()
    return _efficientnet


def load_resnet50_weights(path: str):
    """Load fine-tuned ResNet-50 weights."""
    import os
    if not os.path.exists(path):
        return
    global _resnet50
    m = _build_resnet50(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    _resnet50 = m
    print(f"[Grading] Loaded ResNet-50 from {path}")


def load_efficientnet_weights(path: str):
    """Load fine-tuned EfficientNet-B4 weights."""
    import os
    if not os.path.exists(path):
        return
    global _efficientnet
    m = _build_efficientnet_b4(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    _efficientnet = m
    print(f"[Grading] Loaded EfficientNet-B4 from {path}")


# ── Inference helpers ─────────────────────────────────────────────────────────

def _enable_dropout(model: nn.Module):
    """Switch all Dropout layers to training mode (for MC Dropout)."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _predict_once(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def _mc_dropout_predict(model: nn.Module, tensor: torch.Tensor,
                        n_runs: int = MC_RUNS) -> dict:
    """
    25 MC Dropout inference passes → mean probability + std uncertainty.
    """
    model.eval()
    _enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_runs):
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    model.eval()          # restore fully-eval mode

    all_probs  = np.array(all_probs)   # (n_runs, 5)
    mean_probs = all_probs.mean(axis=0)
    std_probs  = all_probs.std(axis=0)

    pred_class = int(mean_probs.argmax())
    confidence = float(mean_probs[pred_class])
    uncertainty = float(std_probs[pred_class])

    return {
        "probs":        mean_probs.tolist(),
        "std":          std_probs.tolist(),
        "pred_class":   pred_class,
        "pred_label":   DR_CLASSES[pred_class],
        "confidence":   round(confidence, 4),
        "uncertainty":  round(uncertainty, 4),
        "mc_runs":      n_runs,
        "all_probs":    all_probs.tolist(),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def grade_image(img_bgr: np.ndarray) -> dict:
    """
    Run both ResNet-50 and EfficientNet-B4 with MC Dropout.

    Returns:
        {
          "resnet50":      { pred_label, confidence, uncertainty, probs, ... },
          "efficientnet":  { pred_label, confidence, uncertainty, probs, ... },
          "ensemble_probs": list[float],
          "ensemble_label": str,
          "ensemble_confidence": float,
        }
    """
    tensor = _preprocess(img_bgr)

    resnet_result  = _mc_dropout_predict(_get_resnet50(), tensor)
    effnet_result  = _mc_dropout_predict(_get_efficientnet(), tensor)

    # Simple ensemble: average probability distributions
    ensemble_probs = (
        np.array(resnet_result["probs"]) + np.array(effnet_result["probs"])
    ) / 2.0

    ens_class = int(ensemble_probs.argmax())

    return {
        "resnet50":           resnet_result,
        "efficientnet":       effnet_result,
        "ensemble_probs":     ensemble_probs.tolist(),
        "ensemble_label":     DR_CLASSES[ens_class],
        "ensemble_confidence": round(float(ensemble_probs[ens_class]), 4),
    }
