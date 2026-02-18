"""CV detection service using YOLO for object and face detection.

This module provides object and face detection capabilities using YOLOv8.
All models are lazy-loaded on first use to avoid import-time overhead.
"""

import logging
from pathlib import Path
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)


class CVDetectionService:
    """YOLO object and face detection service with lazy model loading."""

    def __init__(self, device: str = "cuda:0"):
        """Initialize service with device specification.

        Args:
            device: Device for inference ("cuda:0" or "cpu")
        """
        self.device = device
        self._model = None

    def _load_models(self):
        """Lazy-load YOLO model on first use.

        Raises:
            RuntimeError: If model loading fails (network error, corrupt weights, etc.)
        """
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO model (yolov8m.pt)...")
            self._model = YOLO("yolov8m.pt")
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(
                f"Failed to load YOLO model: {e}. On first run, models (~50MB) are "
                "auto-downloaded. Ensure internet connectivity and write access to the "
                "current directory."
            )

    def detect_objects_and_faces(
        self, image_path: str, confidence_threshold: float = 0.5
    ) -> Dict[str, List[dict]]:
        """Run detection sweep on single image.

        Args:
            image_path: Path to image file
            confidence_threshold: Minimum confidence for detections (0.0-1.0)

        Returns:
            {
                "objects": [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...],
                "faces": [{"class": "person", "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
            }
        """
        self._load_models()

        results = {"objects": [], "faces": []}

        # Run YOLO detection
        yolo_results = self._model.predict(
            image_path, conf=confidence_threshold, device=self.device, verbose=False
        )[0]

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

            class_name = yolo_results.names[cls_id]

            # Add to objects list
            results["objects"].append(
                {"class": class_name, "confidence": conf, "bbox": bbox}
            )

            # Extract faces from "person" detections (upper 40% of person bbox)
            if class_name == "person":
                x1, y1, x2, y2 = bbox
                face_height = (y2 - y1) * 0.4
                face_bbox = [x1, y1, x2, y1 + face_height]

                results["faces"].append(
                    {"class": "person", "confidence": conf, "bbox": face_bbox}
                )

        return results

    def detect_faces_from_bytes(
        self, image_bytes: bytes, confidence_threshold: float = 0.5
    ) -> List[dict]:
        """Detect faces from raw image bytes.

        Converts bytes to PIL/numpy, runs YOLO, extracts face bboxes
        from person detections using upper-40% logic.

        Args:
            image_bytes: Raw image data (PNG, JPEG, etc.)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)

        Returns:
            List of face dicts: [{"confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
        """
        self._load_models()

        import io
        import numpy as np

        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img_pil)

        yolo_results = self._model.predict(
            img_np, conf=confidence_threshold, device=self.device, verbose=False
        )[0]

        faces = []
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()

            class_name = yolo_results.names[cls_id]

            if class_name == "person":
                x1, y1, x2, y2 = bbox
                face_height = (y2 - y1) * 0.4
                face_bbox = [x1, y1, x2, y1 + face_height]
                faces.append({"confidence": conf, "bbox": face_bbox})

        return faces

    def save_crop(
        self,
        image_path: str,
        bbox: List[float],
        output_path: str,
        padding: float = 0.1,
    ) -> str:
        """Crop region from image with padding and save to disk.

        Args:
            image_path: Path to source image
            bbox: Bounding box as [x1, y1, x2, y2]
            output_path: Where to save the crop
            padding: Percentage padding to add (0.1 = 10%)

        Returns:
            output_path
        """
        img = Image.open(image_path)
        w, h = img.size

        # Apply padding
        x1, y1, x2, y2 = bbox
        pad_w = (x2 - x1) * padding
        pad_h = (y2 - y1) * padding

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        # Crop and save
        crop = img.crop((x1, y1, x2, y2))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)

        return output_path
