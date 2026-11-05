"""
MiniFASNet Liveness Detection Service for Biometric Authentication.

Provides lightweight face anti-spoofing verification using MiniFASNetV1SE.
Integrates into Registration and Login pipelines to enforce liveness-first verification:
Camera Frame -> MiniFASNet Liveness -> REAL Confirmation -> Capture Verified Frame
-> MediaPipe BlazeFace -> Face Crop -> FaceNet Embedding -> FAISS / PostgreSQL.
"""

import os
import sys
import threading
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from uniface import RetinaFace

logger = logging.getLogger(__name__)

# ==============================================================================
# Model Architecture (MiniFASNetV1SE)
# ==============================================================================

CHANNEL_CONFIGS = {
    "v1se": {
        "stem": 32,
        "transition1": {"expand": 103, "out": 64},
        "stage2": [(13, 64), (26, 64), (13, 64), (52, 64)],
        "transition2": {"expand": 231, "out": 128},
        "stage3": [(154, 128), (52, 128), (26, 128), (52, 128), (26, 128), (26, 128)],
        "transition3": {"expand": 308, "out": 128},
        "stage4": [(26, 128), (26, 128)],
        "final": 512,
    }
}


class ConvBNPReLU(nn.Module):
    """Conv2d + BatchNorm2d + PReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.bn(self.conv(x)))


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.relu(self.bn1(self.fc1(scale)))
        scale = self.sigmoid(self.bn2(self.fc2(scale)))
        return x * scale


class InvertedResidual(nn.Module):
    """Inverted residual block: expand -> depthwise -> project (+ optional SE & residual)."""

    def __init__(
        self,
        in_channels: int,
        expand_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        self.conv = ConvBNPReLU(in_channels, expand_channels, kernel_size=1)
        self.conv_dw = ConvBNPReLU(expand_channels, expand_channels, 3, stride, 1, groups=expand_channels)
        self.project = ConvBNPReLU(expand_channels, out_channels, kernel_size=1, activation=False)
        self.se = SqueezeExcite(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        out = self.se(out)

        if self.use_residual:
            out = out + x
        return out


class ResidualStack(nn.Module):
    """Stack of inverted residual blocks (SE applied to last block only)."""

    def __init__(
        self,
        in_channels: int,
        block_configs: list[tuple[int, int]],
        use_se: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        current_ch = in_channels

        for i, (expand_ch, out_ch) in enumerate(block_configs):
            layers.append(
                InvertedResidual(
                    in_channels=current_ch,
                    expand_channels=expand_ch,
                    out_channels=out_ch,
                    stride=1,
                    use_se=use_se and (i == len(block_configs) - 1),
                )
            )
            current_ch = out_ch

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MiniFASNet(nn.Module):
    """Lightweight Face Anti-Spoofing Network (expects 80x80 input)."""

    def __init__(
        self,
        config: dict,
        dropout: float = 0.0,
        num_classes: int = 3,
        use_se: bool = False,
    ) -> None:
        super().__init__()

        stem_ch = config["stem"]
        t1 = config["transition1"]
        t2 = config["transition2"]
        t3 = config["transition3"]
        final_ch = config["final"]

        self.stem = ConvBNPReLU(3, stem_ch, kernel_size=3, stride=2, padding=1)
        self.stem_dw = ConvBNPReLU(stem_ch, stem_ch, kernel_size=3, stride=1, padding=1, groups=stem_ch)

        self.transition1 = InvertedResidual(stem_ch, t1["expand"], t1["out"], stride=2)
        self.stage2 = ResidualStack(t1["out"], config["stage2"], use_se=use_se)

        self.transition2 = InvertedResidual(config["stage2"][-1][1], t2["expand"], t2["out"], stride=2)
        self.stage3 = ResidualStack(t2["out"], config["stage3"], use_se=use_se)

        self.transition3 = InvertedResidual(config["stage3"][-1][1], t3["expand"], t3["out"], stride=2)
        self.stage4 = ResidualStack(t3["out"], config["stage4"], use_se=use_se)

        stage4_out = config["stage4"][-1][1]
        self.final_expand = ConvBNPReLU(stage4_out, final_ch, kernel_size=1)
        self.final_dw = ConvBNPReLU(final_ch, final_ch, kernel_size=5, groups=final_ch, activation=False)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(final_ch, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_dw(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        x = self.transition3(x)
        x = self.stage4(x)
        x = self.final_expand(x)
        x = self.final_dw(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.classifier(x)
        return x


class MiniFASNetV1SE(MiniFASNet):
    """MiniFASNet V1 with Squeeze-and-Excitation modules (~0.43M params)."""

    def __init__(self) -> None:
        super().__init__(config=CHANNEL_CONFIGS["v1se"], dropout=0.75, num_classes=3, use_se=True)


# ==============================================================================
# Preprocessing Utilities
# ==============================================================================

def xyxy2xywh(bbox: np.ndarray | list) -> np.ndarray:
    """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format."""
    if isinstance(bbox, list):
        bbox = np.array(bbox)
    result = np.copy(bbox)
    result[..., 2] = bbox[..., 2] - bbox[..., 0]
    result[..., 3] = bbox[..., 3] - bbox[..., 1]
    return result


def crop_face(
    image: np.ndarray,
    bbox: list[int],
    scale: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop and resize face region with expansion scale."""
    src_h, src_w = image.shape[:2]
    x, y, box_w, box_h = bbox

    if box_h <= 0 or box_w <= 0:
        return cv2.resize(image, (out_w, out_h))

    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w = box_w * scale
    new_h = box_h * scale

    center_x = x + box_w / 2
    center_y = y + box_h / 2

    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(src_w - 1, int(center_x + new_w / 2))
    y2 = min(src_h - 1, int(center_y + new_h / 2))

    cropped = image[y1 : y2 + 1, x1 : x2 + 1]
    if cropped.size == 0:
        return cv2.resize(image, (out_w, out_h))
    return cv2.resize(cropped, (out_w, out_h))


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert numpy image (HWC) to PyTorch float tensor (CHW)."""
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


# ==============================================================================
# Liveness Service Implementation
# ==============================================================================

DEFAULT_LIVENESS_THRESHOLD = 0.85


def get_liveness_threshold() -> float:
    """
    Retrieves the configurable REAL confidence threshold from environment.
    Defaults to 0.85 per pipeline requirements.
    """
    raw_val = os.getenv("LIVENESS_REAL_THRESHOLD", str(DEFAULT_LIVENESS_THRESHOLD))
    try:
        val = float(raw_val)
        if not (0.0 <= val <= 1.0):
            raise ValueError("Liveness threshold must be between 0.0 and 1.0.")
        return val
    except Exception as e:
        logger.warning(
            f"[CONFIG WARNING] Invalid LIVENESS_REAL_THRESHOLD '{raw_val}': {e}. "
            f"Falling back to {DEFAULT_LIVENESS_THRESHOLD}."
        )
        return DEFAULT_LIVENESS_THRESHOLD


class LivenessService:
    """
    Singleton service for MiniFASNetV1SE Face Anti-Spoofing.
    Loads once on startup and provides thread-safe CPU inference.
    Fails closed if model fails to load.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self.device = torch.device("cpu")
        self._lock = threading.Lock()
        self.model: MiniFASNetV1SE | None = None
        self.detector: RetinaFace | None = None
        self._is_loaded = False
        self._load_error: str | None = None

        self._initialize(weights_path)

    def _resolve_weights_path(self, custom_path: str | None) -> str:
        """Find the MiniFASNetV1SE.pth weights file across known project paths."""
        if custom_path:
            if os.path.isfile(custom_path):
                return os.path.abspath(custom_path)
            raise FileNotFoundError(f"Specified weights path not found: {custom_path}")

        candidates = []
        env_path = os.getenv("LIVENESS_WEIGHTS_PATH")
        if env_path:
            if os.path.isfile(env_path):
                return os.path.abspath(env_path)
            raise FileNotFoundError(f"LIVENESS_WEIGHTS_PATH specified in env but not found: {env_path}")

        # Standard project layout relative locations
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)

        candidates.extend([
            os.path.join(project_root, "liveliness", "weights", "MiniFASNetV1SE.pth"),
            os.path.join(current_dir, "weights", "MiniFASNetV1SE.pth"),
            os.path.join(project_root, "face-anti-spoofing", "weights", "MiniFASNetV1SE.pth"),
        ])

        for path in candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)

        raise FileNotFoundError(
            f"MiniFASNetV1SE.pth weights file not found. Checked: {candidates}"
        )

    def _initialize(self, custom_path: str | None) -> None:
        """Load MiniFASNet model and RetinaFace detector."""
        try:
            weights_file = self._resolve_weights_path(custom_path)
            print(f"[LIVENESS] Loading MiniFASNetV1SE weights from: {weights_file}")

            model = MiniFASNetV1SE()
            state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()

            self.model = model
            self.detector = RetinaFace()
            self._is_loaded = True
            self._load_error = None
            print("[LIVENESS] MiniFASNetV1SE and RetinaFace detector initialized successfully!")
        except Exception as e:
            self._is_loaded = False
            self._load_error = str(e)
            print(f"[LIVENESS CRITICAL] Failed to initialize MiniFASNet liveness model: {e}")
            logger.error(f"Liveness initialization failed: {e}", exc_info=True)

    def is_available(self) -> bool:
        """Return True if model and detector are successfully loaded."""
        return self._is_loaded and self.model is not None and self.detector is not None

    def verify_liveness(self, image: np.ndarray, threshold: float | None = None) -> dict:
        """
        Step 2: Liveness Detection (MiniFASNetV1SE).
        Classifies the incoming camera frame as REAL or FAKE.

        Fails closed: if model failed to load, rejects with model_unavailable.
        If no face is detected, returns no_face.
        Only returns is_real=True if class is Real (index 1) AND confidence >= threshold.

        Returns:
            dict containing:
                is_real (bool): True strictly when real face confirmed above threshold.
                score (float): Confidence score for REAL class.
                threshold (float): Active threshold used.
                label (str): "Real" or "Fake".
                status (str): "real", "fake_or_uncertain", "no_face", or "model_unavailable".
                message (str): User-friendly explanation.
                bbox (list | None): Detected face bbox in [x, y, w, h] format.
        """
        active_threshold = threshold if threshold is not None else get_liveness_threshold()

        # Fail closed check
        if not self.is_available():
            return {
                "is_real": False,
                "score": 0.0,
                "threshold": active_threshold,
                "label": "Error",
                "status": "model_unavailable",
                "message": f"Liveness detection service unavailable: {self._load_error}. Operations failed closed.",
                "bbox": None
            }

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "is_real": False,
                "score": 0.0,
                "threshold": active_threshold,
                "label": "Invalid",
                "status": "invalid_image",
                "message": "Invalid camera frame: input image is empty or corrupted.",
                "bbox": None
            }

        # Thread-safe face localization with RetinaFace
        with self._lock:
            try:
                faces = self.detector.detect(image)
            except Exception as det_err:
                logger.error(f"Face detection error during liveness check: {det_err}")
                faces = []

        # Filter faces with detection confidence >= 0.5
        valid_faces = [f for f in faces if getattr(f, "confidence", 1.0) >= 0.5]

        if not valid_faces:
            return {
                "is_real": False,
                "score": 0.0,
                "threshold": active_threshold,
                "label": "No Face",
                "status": "no_face",
                "message": "No face detected in camera frame. Please align your face with the camera.",
                "bbox": None
            }

        # Select primary face (largest bounding box area)
        primary_face = max(
            valid_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        bbox_xywh = xyxy2xywh(primary_face.bbox).astype(int).tolist()

        # MiniFASNetV1SE expects scale 4.0 and 80x80 input size
        face_crop = crop_face(image, bbox_xywh, scale=4.0, out_w=80, out_h=80)
        tensor = to_tensor(face_crop).unsqueeze(0).to(self.device)

        # Thread-safe MiniFASNet inference
        with self._lock:
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        label_idx = int(np.argmax(probs))
        # Class mapping in MiniFASNet: index 1 is Real; indices 0 and 2 are spoof/fake
        real_score = float(probs[1])

        # Require BOTH argmax == 1 AND real_score >= threshold
        is_real = (label_idx == 1 and real_score >= active_threshold)
        label = "Real" if label_idx == 1 else "Fake"

        if is_real:
            status = "real"
            message = f"Real face verified successfully (confidence: {real_score:.2f})."
        elif label_idx != 1:
            status = "fake"
            message = "Liveness check failed: Spoof or fake face detected. Please face the camera and try again."
        else:
            status = "uncertain"
            message = (
                f"Liveness check failed: Face confidence ({real_score:.2f}) "
                f"below required threshold ({active_threshold:.2f}). Please face the camera directly."
            )

        return {
            "is_real": is_real,
            "score": real_score,
            "threshold": active_threshold,
            "label": label,
            "status": status,
            "message": message,
            "bbox": bbox_xywh
        }


# ==============================================================================
# Global Singleton Accessor
# ==============================================================================

_global_liveness_service: LivenessService | None = None
_init_lock = threading.Lock()


def get_liveness_service(weights_path: str | None = None) -> LivenessService:
    """
    Returns the singleton LivenessService instance.
    Initializes on first call (typically backend startup).
    """
    global _global_liveness_service
    if _global_liveness_service is None:
        with _init_lock:
            if _global_liveness_service is None:
                _global_liveness_service = LivenessService(weights_path=weights_path)
    return _global_liveness_service
