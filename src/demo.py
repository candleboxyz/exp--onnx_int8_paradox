"""
Live Inference Demo
==================
Real-time object detection using optimized models.
"""

import logging
import os
import time
from collections import deque
from typing import Any, Sequence

import cv2
import numpy as np

from .onnx_utils.ort_session_constrain import build_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


class EdgeYOLODemo:
    """Live demo for edge-optimized YOLO model."""

    def __init__(
        self,
        model_path: str = "yolov5nu_640_int8.onnx",
        conf_threshold: float = 0.25,
    ):
        """Initialize the demo with optimized model."""

        # for deployment, not benchmarking
        self.session = build_session(
            model_path,
            use_cuda=False,
            use_coreml=True,  # auto-detect CoreML
            for_benchmarking=False,  # enable memory optimizations
            intra_op_num_threads=None,  # let ORT decide (auto)
            inter_op_num_threads=None,  # let ORT decide (auto)
        )

        self.input_name = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.fps_history = deque(maxlen=30)

        # COCO class names (first 20 for brevity)
        self.classes = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
        ]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLO inference."""
        self.original_shape = image.shape[:2]

        # Resize to model input size
        input_img = cv2.resize(image, (640, 640))

        # Convert BGR to RGB and normalize
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0

        # CHW format for ONNX
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0)

        return input_img

    def postprocess(self, outputs: Sequence[Any]) -> tuple[list, list, list]:
        """
        Convert YOLO outputs to bounding boxes.
        Handles the [1, 84, 2100] output format.
        """
        predictions = outputs[0][0]  # Shape: [84, 2100]
        predictions = predictions.T  # Transpose to [2100, 84]

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            # First 4 values are bbox coordinates
            x_center, y_center, width, height = pred[:4]

            # Next 80 values are class scores
            class_scores = pred[4:84] if len(pred) >= 84 else pred[4:]

            # Get best class
            if len(class_scores) > 0:
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence > self.conf_threshold:
                    # Scale coordinates back to original image size
                    h_orig, w_orig = self.original_shape

                    x1 = int((x_center - width / 2) * w_orig / 640)
                    y1 = int((y_center - height / 2) * h_orig / 640)
                    x2 = int((x_center + width / 2) * w_orig / 640)
                    y2 = int((y_center + height / 2) * h_orig / 640)

                    # Clip to image boundaries
                    x1 = max(0, min(x1, w_orig))
                    y1 = max(0, min(y1, h_orig))
                    x2 = max(0, min(x2, w_orig))
                    y2 = max(0, min(y2, h_orig))

                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))

        # Apply NMS
        if len(boxes):
            indices = self.nms(boxes, scores, 0.45)
            boxes = [boxes[i] for i in indices]
            scores = [scores[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]

        return boxes, scores, class_ids

    def nms(
        self, boxes: list | np.ndarray, scores: list | np.ndarray, iou_threshold: float
    ) -> list[int]:
        """Non-Maximum Suppression to remove overlapping boxes."""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Sort by score
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]

            # Calculate IoU
            x1 = np.maximum(current_box[0], other_boxes[:, 0])
            y1 = np.maximum(current_box[1], other_boxes[:, 1])
            x2 = np.minimum(current_box[2], other_boxes[:, 2])
            y2 = np.minimum(current_box[3], other_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            box_area = (current_box[2] - current_box[0]) * (
                current_box[3] - current_box[1]
            )
            boxes_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
                other_boxes[:, 3] - other_boxes[:, 1]
            )

            union = box_area + boxes_area - intersection
            iou = intersection / (union + 1e-6)

            # Remove boxes with high IoU
            indices = indices[1:][iou < iou_threshold]

        return keep

    def draw_results(
        self,
        image: np.ndarray,
        boxes: list,
        scores: list,
        class_ids: list,
        inference_time: float,
    ) -> np.ndarray:
        """Draw bounding boxes and stats on image."""
        # Draw bounding boxes
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # Use class_id for color consistency
            np.random.seed(class_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if class_id < len(self.classes):
                label = f"{self.classes[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Draw stats overlay
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0

        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        # Add stats text
        cv2.putText(
            image,
            "Edge AI YOLOv5 Demo",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            "Model: YOLOv5n INT8 (2.8MB)",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"Inference: {inference_time:.1f}ms",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"FPS: {avg_fps:.1f}",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            image,
            f"Objects: {len(boxes)}",
            (15, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        return image

    def run_inference(self, image: np.ndarray) -> tuple:
        """Run single inference on image."""
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        start = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = (time.time() - start) * 1000

        # Postprocess
        boxes, scores, class_ids = self.postprocess(outputs)

        # Update FPS history
        self.fps_history.append(1000 / inference_time)

        return boxes, scores, class_ids, inference_time

    def run_webcam(self):
        """Run live webcam demo."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return

        print("Press 'q' to quit, 's' to save screenshot")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            boxes, scores, class_ids, time_ms = self.run_inference(frame)

            # Draw results
            frame = self.draw_results(frame, boxes, scores, class_ids, time_ms)

            # Display
            cv2.imshow("Edge AI YOLOv5 Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")

        cap.release()
        cv2.destroyAllWindows()
