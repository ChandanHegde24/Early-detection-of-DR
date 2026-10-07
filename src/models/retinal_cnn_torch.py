"""
retinal_cnn_torch.py - PyTorch EfficientNetB0 loader + Grad-CAM for DR grading.
Place at: src/models/retinal_cnn_torch.py
"""
import io
import base64
import numpy as np
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2

IMG_SIZE     = 224
NUM_CLASSES  = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

DEFAULT_CNN_MODEL_PATHS = (
    "saved_models/best_checkpoint_v2.pth",
    "saved_models/cnn_weights.pth",
    "saved_models/cnn_classes_2_3_4_weights.pth",
    "saved_models/cnn_stage0_weights.pth",
)


def resolve_cnn_model_path(path: str | None = None) -> str:
    """Return the active 5-class DR checkpoint, preferring the newest compatible model."""
    if path:
        if Path(path).exists():
            return path
        # keep searching valid fallback paths if the explicit file is absent

    valid_candidates: list[str] = []
    for candidate in DEFAULT_CNN_MODEL_PATHS:
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue
        try:
            ckpt = torch.load(candidate_path, map_location=DEVICE, weights_only=False)
            inferred = ckpt.get("num_classes")
            if inferred is None and isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
                last_key = next(reversed(state))
                inferred = state[last_key].shape[0]
            if inferred in (None, NUM_CLASSES):
                valid_candidates.append(str(candidate_path))
        except Exception:
            continue

    if valid_candidates:
        return valid_candidates[0]

    return str(Path(DEFAULT_CNN_MODEL_PATHS[0]))


def _build_model(num_classes: int = NUM_CLASSES):
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f, 256), nn.ReLU(),
        nn.BatchNorm1d(256), nn.Dropout(0.3), nn.Linear(256, num_classes),
    )
    return m


def load_cnn_model(path: str | None = None, num_classes: int | None = None):
    resolved_path = resolve_cnn_model_path(path)
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"CNN model not found at '{resolved_path}'. Run train_stage2.py first.")
    ckpt = torch.load(resolved_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    inferred_num_classes = num_classes or ckpt.get("num_classes", NUM_CLASSES)
    model = _build_model(num_classes=inferred_num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"[retinal_cnn_torch] Loaded from {resolved_path} on {DEVICE} ({inferred_num_classes} classes)")
    return model


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: nn.Module):
        self.model     = model
        self.gradients = None
        self.activations = None
        # Hook into the last conv block of EfficientNetB0
        target_layer = model.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(tensor)
        score  = output[0, class_idx]
        score.backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalize 0–1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam


def _cam_to_base64(cam: np.ndarray, orig_img: np.ndarray) -> tuple:
    """Convert CAM array → base64 heatmap and overlay."""
    # Heatmap (colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay on original image (resize orig to 224×224)
    orig_resized = cv2.resize(orig_img, (IMG_SIZE, IMG_SIZE))
    overlay      = (0.5 * orig_resized + 0.5 * heatmap).astype(np.uint8)

    def to_b64(arr):
        _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode("utf-8")

    return to_b64(heatmap), to_b64(overlay)


# ── Main inference ────────────────────────────────────────────────────────────
def predict_fundus(model: nn.Module, image_bytes: bytes) -> Dict:
    # Load original image for overlay
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_np  = np.array(pil_img)

    tensor   = VAL_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    # ── Grad-CAM (needs gradients) ─────────────────────────────────────────
    model.train(False)
    gradcam = GradCAM(model)

    with torch.enable_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        grade  = int(np.argmax(probs))
        cam    = gradcam.generate(tensor, grade)

    heatmap_b64, overlay_b64 = _cam_to_base64(cam, orig_np)

    # ── Scores ────────────────────────────────────────────────────────────────
    risk_score = float(probs @ [0.0, 0.25, 0.5, 0.75, 1.0])
    tier = "Urgent" if risk_score >= 0.72 else "Moderate" if risk_score >= 0.42 else "Low Risk"
    recs = [
        "No DR detected. Continue annual screening.",
        "Mild NPDR. Monitor every 12 months.",
        "Moderate NPDR. Referral within 3 months.",
        "Severe NPDR. Urgent referral within 1 month.",
        "Proliferative DR. Immediate referral required.",
    ]

    return {
        "predicted_grade":           grade,
        "predicted_label":           GRADE_LABELS[grade],
        "risk_score":                risk_score,
        "screening_tier":            tier,
        "grade_probabilities":       [
            {"grade": i, "label": GRADE_LABELS[i], "probability": float(probs[i])}
            for i in range(NUM_CLASSES)
        ],
        "model_used":                "EfficientNetB0-PyTorch",
        "grad_cam_available":        True,
        "grad_cam_heatmap":          heatmap_b64,
        "grad_cam_overlay":          overlay_b64,
        "baseline_clinical_score":   None,
        "baseline_recommendation":   recs[grade],
        "baseline_factor_breakdown": None,
    }
