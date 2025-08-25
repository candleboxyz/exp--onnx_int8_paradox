"""
Model Quantization Module
========================
INT8 quantization for edge deployment.
"""

import logging
import os

from onnxruntime.quantization import QuantType, quantize_dynamic

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handles various quantization strategies."""

    @staticmethod
    def quantize_onnx_dynamic(
        input_model: str, output_model: str, weight_type: QuantType = QuantType.QUInt8
    ) -> dict:
        """
        Apply dynamic quantization to ONNX model.

        Returns:
            Dictionary with quantization statistics.
            Keys:
                - original_size_mb (float)
                - quantized_size_mb (float, optional)
                - compression_ratio (float, optional)
                - success (bool)
                - error (str, optional)
        """

        original_size = os.path.getsize(input_model) / (1024 * 1024)
        stats = {
            "original_size_mb": original_size,
            "quantized_size_mb": None,
            "compression_ratio": None,
            "success": False,
            "error": None,
        }

        try:
            # Perform quantization
            quantize_dynamic(
                model_input=input_model,
                model_output=output_model,
                weight_type=weight_type,
            )

            # Get quantized size
            quantized_size = os.path.getsize(output_model) / (1024 * 1024)
            stats["quantized_size_mb"] = quantized_size
            stats["compression_ratio"] = original_size / quantized_size
            stats["success"] = True

            logger.info(
                f"Quantization successful: {original_size:.2f}MB -> {quantized_size:.2f}MB"
            )

        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            stats["success"] = False
            stats["error"] = str(e)

        return stats
