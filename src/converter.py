"""
Model Conversion Pipeline
=========================
Handles PyTorch to ONNX conversion with optimization options.
"""

import logging
import os

import torch
import torchvision.models as models
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConverter:
    """Handles model conversion between frameworks."""

    @staticmethod
    def pytorch_to_onnx(
        model_path: str,
        output_path: str,
        input_shape: tuple[int, ...] = (1, 3, 640, 640),
        simplify: bool = True,
    ) -> dict:
        """
        Convert PyTorch model to ONNX format.

        Returns:
            Dictionary with conversion statistics
        """
        stats = {}

        # Handle YOLOv5 models
        if "yolo" in model_path.lower():
            model = YOLO(model_path)
            model.export(format="onnx", imgsz=input_shape[-1], simplify=simplify)
            default_name = model_path.replace(".pt", ".onnx")
            os.rename(default_name, output_path)
        else:
            # Standard PyTorch models (ResNet, etc.)
            if model_path == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = torch.load(model_path)

            model.eval()
            dummy_input = torch.randn(*input_shape)

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

        # Collect statistics
        if os.path.exists(output_path):
            stats["output_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)
            stats["success"] = True
        else:
            stats["success"] = False

        return stats
