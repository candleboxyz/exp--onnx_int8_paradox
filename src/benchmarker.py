"""
Performance Benchmarking Module
===============================
Measures and compares model inference performance under multiple scenarios.
"""

import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import torch

from .onnx_utils.ort_session_constrain import build_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Benchmarker:
    """Handles performance benchmarking of models under controlled conditions."""

    def __init__(self, num_runs: int = 100, warmup: int = 10):
        self.num_runs = num_runs
        self.warmup = warmup

    def benchmark_onnx_comprehensive(self, model_path: str) -> Dict[str, Any]:
        """
        Comprehensive benchmark under multiple scenarios for scientific comparison.

        This method tests the same model under different thread configurations
        to understand performance characteristics. All models are tested with
        identical scenarios to maintain controlled variables.

        Args:
            model_path: Path to ONNX model file

        Returns:
            Dictionary with results for each scenario, including:
            - constrained: Single-thread for controlled comparison
            - practical: Auto-threading for real-world performance
            - apple_optimized: CoreML acceleration (if on Apple Silicon)
        """
        results = {}

        # Define benchmark scenarios with clear scientific purpose
        scenarios = {
            "constrained": {
                "description": "Single-thread execution for controlled comparison",
                "settings": {
                    "use_cuda": False,
                    "use_coreml": False,  # Disable for consistency across platforms
                    "for_benchmarking": True,
                    "intra_op_num_threads": 1,  # Single thread within operators
                    "inter_op_num_threads": 1,  # Single thread across operators
                },
            },
            "practical": {
                "description": "Auto-threading as in real deployment",
                "settings": {
                    "use_cuda": False,
                    "use_coreml": False,  # keep disabled for fair comparison
                    "for_benchmarking": False,  # enable memory patterns
                    "intra_op_num_threads": None,  # let ORT decide
                    "inter_op_num_threads": None,  # let ORT decide
                },
            },
        }

        # Add platform-specific scenario if on Apple Silicon
        import platform

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            scenarios["apple_optimized"] = {
                "description": "Apple Silicon with CoreML (when applicable)",
                "settings": {
                    "use_cuda": False,
                    "use_coreml": True,  # enable CoreML provider
                    "for_benchmarking": False,
                    "intra_op_num_threads": None,
                    "inter_op_num_threads": None,
                },
            }

        # Run benchmarks for each scenario
        for scenario_name, scenario_config in scenarios.items():
            logger.info(
                f"Running scenario: {scenario_name} - {scenario_config['description']}"
            )

            try:
                # Create session with scenario-specific settings
                session = build_session(model_path, **scenario_config["settings"])

                # Run benchmark
                timing_results = self._benchmark_session(session, model_path)

                # Store results with metadata for reproducibility
                results[scenario_name] = {
                    "description": scenario_config["description"],
                    "settings": scenario_config["settings"],
                    "timings": timing_results,
                    "providers": session.get_providers(),
                }

                logger.info(
                    f"  {scenario_name}: {timing_results['mean_ms']:.2f}ms "
                    f"({timing_results['fps']:.1f} FPS) CV={timing_results['cv']:.3f}"
                )

            except Exception as e:
                logger.warning(f"  {scenario_name} failed: {str(e)}")
                results[scenario_name] = {
                    "description": scenario_config["description"],
                    "error": str(e),
                }

        return results

    def benchmark_onnx(self, model_path: str, input_shape: tuple | None = None) -> dict:
        """
        Standard benchmark with single configuration (backward compatibility).

        Uses 'constrained' scenario (single-thread) for reproducible results.
        This maintains backward compatibility with existing code while ensuring
        controlled experimental conditions.

        Args:
            model_path: Path to ONNX model
            input_shape: Override model's input shape if needed

        Returns:
            Dictionary with timing statistics
        """
        
        # Use constrained scenario as default for scientific reproducibility
        session = build_session(
            model_path,
            use_cuda=False,
            use_coreml=False,  # disable for consistency
            for_benchmarking=True,
            intra_op_num_threads=1,  # single thread for controlled comparison
            inter_op_num_threads=1,  # sequential execution
        )

        return self._benchmark_session(session, model_path, input_shape)

    def _benchmark_session(
        self,
        session: ort.InferenceSession,
        model_path: str,
        input_shape: tuple | None = None,
    ) -> dict:
        """
        Internal method to benchmark an ONNX Runtime session.

        This method handles the actual benchmarking logic, including:
        - Input shape resolution and normalization
        - Warmup runs to stabilize performance
        - Multiple timed runs for statistical significance
        - Memory cleanup to prevent buildup

        Args:
            session: Configured ONNX Runtime session
            model_path: Path to model (for logging purposes)
            input_shape: Override input shape, otherwise use model's shape

        Returns:
            Dictionary with comprehensive timing statistics
        """
        input_name = session.get_inputs()[0].name

        # Resolve input shape: prefer explicit argument over model metadata
        raw_shape = (
            input_shape if input_shape is not None else session.get_inputs()[0].shape
        )

        # Guard clause: some models do not store shape metadata (rank unknown)
        if raw_shape is None:
            raise ValueError(
                "Model input shape is unknown. Pass `input_shape` explicitly or "
                "export the model with static shapes."
            )

        # Ensure iterating over the shape (accept list or tuple)
        if not isinstance(raw_shape, (list, tuple)):
            raise TypeError(
                f"Unexpected shape type: {type(raw_shape).__name__} "
                f"(expected list/tuple)."
            )

        # Normalize dynamic dimensions:
        # - Any symbolic string (e.g., 'N', 'batch_size') -> 1
        # - -1 (dynamic) -> 1
        # - None (unknown) -> 1
        # - Positive ints remain unchanged
        input_shape = tuple(
            [
                1 if (isinstance(dim, str) or dim in (-1, None)) else dim
                for dim in raw_shape
            ]
        )

        # Create dummy input with random data
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup runs to stabilize cache and GPU state
        for _ in range(self.warmup):
            _ = session.run(None, {input_name: dummy_input})

        # Actual benchmark runs
        times_ms = []
        for _ in range(self.num_runs):
            t0 = time.perf_counter()
            outputs = session.run(None, {input_name: dummy_input})
            dt_ms = (time.perf_counter() - t0) * 1000.0  # convert to ms
            times_ms.append(dt_ms)

            # Clear outputs to prevent memory buildup
            del outputs

        return self.calculate_statistics(times_ms)

    def calculate_statistics(self, times_ms: List[float]) -> dict:
        """
        Calculate comprehensive benchmark statistics.

        Computes various statistical measures to understand performance
        characteristics beyond simple mean/std.

        Args:
            times_ms: List of inference times in milliseconds

        Returns:
            Dictionary containing:
            - Basic stats: mean, std, min, max
            - Percentiles: p50, p95, p99
            - Derived metrics: fps, cv, iqr
            - Quality indicators: outlier count
        """
        t_np_arr = np.array(times_ms)
        mean = t_np_arr.mean()
        std = t_np_arr.std()

        return {
            # Basic statistics
            "mean_ms": float(mean),
            "std_ms": float(std),
            "min_ms": float(t_np_arr.min()),
            "max_ms": float(t_np_arr.max()),
            # Percentiles for distribution understanding
            "p50_ms": float(np.percentile(t_np_arr, 50)),  # Median
            "p95_ms": float(np.percentile(t_np_arr, 95)),  # 95% of runs are faster
            "p99_ms": float(np.percentile(t_np_arr, 99)),  # 99% of runs are faster
            # Derived metrics
            "fps": float(1000.0 / mean),
            # Statistical quality indicators
            "cv": float(std / mean),  # Coefficient of Variation (lower is more stable)
            "iqr": float(np.percentile(t_np_arr, 75) - np.percentile(t_np_arr, 25)),
            # ↳ Interquartile Range (robust measure of spread)
            "outliers": int(np.sum(np.abs(t_np_arr - mean) > 2 * std)),
            # ↳ Count of measurements >2 std from mean
        }

    def compare_models_comprehensive(self, model_paths: List[str]) -> Dict:
        """
        Compare multiple models under all scenarios.

        This is the main entry point for comparing FP32 vs INT8 models
        under different execution conditions.

        Args:
            model_paths: List of paths to ONNX models to compare

        Returns:
            Nested dictionary: {model_name: {scenario: results}}
        """
        comparison_results = {}

        for path in model_paths:
            model_name = os.path.basename(path)
            logger.info(f"\n{'='*50}")
            logger.info(f"Benchmarking: {model_name}")
            logger.info(f"{'='*50}")

            comparison_results[model_name] = self.benchmark_onnx_comprehensive(path)

        # Generate and print summary
        self._print_comparison_summary(comparison_results)

        return comparison_results

    def _print_comparison_summary(self, results: Dict):
        """
        Print a formatted summary of comparison results.

        Creates a readable table showing performance across all scenarios
        and models for quick understanding of results.
        """
        logger.info(f"\n{'='*50}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*50}")

        # Extract scenarios (assuming all models tested same scenarios)
        first_model = next(iter(results.values()))
        scenarios = list(first_model.keys())

        for scenario in scenarios:
            logger.info(f"\n{scenario.upper()} Scenario:")
            logger.info("-" * 30)

            for model_name, model_results in results.items():
                if scenario in model_results and "timings" in model_results[scenario]:
                    timings = model_results[scenario]["timings"]
                    logger.info(
                        f"  {model_name:<20} {timings['mean_ms']:>8.2f}ms "
                        f"({timings['fps']:>6.1f} FPS) "
                        f"CV={timings['cv']:.3f}"
                    )
                elif scenario in model_results and "error" in model_results[scenario]:
                    logger.info(
                        f"  {model_name:<20} ERROR: {model_results[scenario]['error']}"
                    )

    def benchmark_pytorch(self, model, input_shape: tuple = (1, 3, 640, 640)) -> dict:
        """
        Benchmark PyTorch model inference speed.

        Maintains same methodology as ONNX benchmarking for fair comparison.

        Args:
            model: PyTorch model in eval mode
            input_shape: Input tensor shape

        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        dummy_input = torch.randn(*input_shape)

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy_input)

        # Benchmark
        times_ms = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                t0 = time.perf_counter()
                outputs = model(dummy_input)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                times_ms.append(dt_ms)

                # Clear outputs to prevent memory buildup
                del outputs

        return self.calculate_statistics(times_ms)

    def compare_models(self, model_paths: List[str]) -> Dict:
        """
        Simple comparison using default settings (backward compatibility).

        Uses constrained scenario for all models.

        Args:
            model_paths: List of ONNX model paths

        Returns:
            Dictionary with benchmark results for each model
        """
        results = {}
        for path in model_paths:
            name = os.path.basename(path)
            results[name] = self.benchmark_onnx(path)
            logger.info(
                f"{name}: {results[name]['mean_ms']:.2f}ms "
                f"({results[name]['fps']:.1f} FPS)"
            )

        return results
