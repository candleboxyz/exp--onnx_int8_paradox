"""
Model Analysis Module
=====================
Analyzes model structure and optimization opportunities.
"""

from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort


class ModelAnalyzer:
    """Analyzes ONNX models for optimization opportunities."""

    @staticmethod
    def analyze_onnx_model(model_path: str) -> dict:
        """
        Analyze ONNX model structure.

        Returns:
            Dictionary with model statistics
        """
        model = onnx.load(model_path)

        # Count operations
        op_counts = defaultdict(int)
        for node in model.graph.node:
            op_counts[node.op_type] += 1

        # Count parameters
        total_params = 0
        param_memory = 0
        for tensor in model.graph.initializer:
            shape = [dim for dim in tensor.dims]
            params = np.prod(shape)
            total_params += params
            # Assume FP32 unless specified
            param_memory += params * 4

        # Get I/O shapes
        input_shape = [
            dim.dim_value
            for dim in model.graph.input[0].type.tensor_type.shape.dim
        ]
        output_shape = [
            dim.dim_value
            for dim in model.graph.output[0].type.tensor_type.shape.dim
        ]

        # Find fusion opportunities
        fusion_opportunities = 0
        for i, node in enumerate(model.graph.node):
            if node.op_type == "Conv":
                # Check if followed by BatchNorm and ReLU
                if i + 1 < len(model.graph.node):
                    if model.graph.node[i + 1].op_type in [
                        "BatchNormalization",
                        "Relu",
                    ]:
                        fusion_opportunities += 1

        return {
            "total_nodes": len(model.graph.node),
            "total_parameters": total_params,
            "parameter_memory_mb": param_memory / (1024 * 1024),
            "input_shape": input_shape,
            "output_shape": output_shape,
            "operation_counts": dict(op_counts),
            "top_operations": dict(
                sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ),
            "fusion_opportunities": fusion_opportunities,
        }

    @staticmethod
    def analyze_yolo_output(session_path: str) -> dict:
        """
        Analyze YOLO model output format.
        """
        session = ort.InferenceSession(session_path)

        # Get model metadata
        inputs = []
        for inp in session.get_inputs():
            inputs.append(
                {"name": inp.name, "shape": inp.shape, "type": inp.type}
            )

        outputs = []
        for out in session.get_outputs():
            outputs.append(
                {"name": out.name, "shape": out.shape, "type": out.type}
            )

        return {"inputs": inputs, "outputs": outputs}
