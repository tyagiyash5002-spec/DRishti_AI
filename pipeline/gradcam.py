"""
DRishti AI — Grad-CAM Visualization Module
Generates gradient-weighted class activation maps from the grading model.
Uses ResNet-50's layer4 as the target convolutional layer.
"""

import numpy as np
import cv2
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


IMG_SIZE = 224

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM hook for any CNN with a named target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.gradients    = None
        self.activations  = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Return a (H, W) float32 CAM in [0, 1]."""
        self.model.eval()
        output = self.model(tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights  = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam      = (weights * self.activations).sum(dim=1).squeeze()  # (H, W)
        cam      = F.relu(cam)
        cam      = cam.cpu().numpy()

        # Normalise
        cam -= cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam /= cam_max

        return cam.astype(np.float32)


def _encode(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def generate_gradcam(img_bgr: np.ndarray, model=None, class_idx: int = None) -> dict:
    """
    Generate Grad-CAM heatmap overlaid on the retinal image.

    Args:
        img_bgr:    Input retinal image (BGR numpy array).
        model:      Grading model (ResNet-50). If None, loads the singleton.
        class_idx:  Target class for Grad-CAM. None = predicted class.

    Returns:
        {
          "heatmap_b64":   str,   # raw Grad-CAM heatmap
          "overlay_b64":   str,   # heatmap overlaid on original image
          "class_idx":     int,
          "class_label":   str,
        }
    """
    from pipeline.grading import _get_resnet50, DR_CLASSES, DEVICE, _transform
    from pipeline.grading import _enable_dropout

    if model is None:
        model = _get_resnet50()

    # Target layer: last conv block of ResNet-50
    target_layer = model.layer4[-1]

    grad_cam = GradCAM(model, target_layer)

    # Preprocess image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(img_rgb)
    tensor  = _transform(pil).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # Generate CAM
    cam = grad_cam.generate(tensor, class_idx=class_idx)

    # Predicted class
    with torch.no_grad():
        model.eval()
        logits    = model(tensor)
        pred_idx  = int(logits.argmax(dim=1).item())

    used_class = class_idx if class_idx is not None else pred_idx

    # Resize CAM to original image size
    h, w      = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Colour heatmap
    heatmap  = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)

    return {
        "heatmap_b64": _encode(heatmap_color),
        "overlay_b64": _encode(overlay),
        "class_idx":   used_class,
        "class_label": DR_CLASSES[used_class],
    }
