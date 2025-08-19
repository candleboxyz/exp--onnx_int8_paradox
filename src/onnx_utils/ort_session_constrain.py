"""
ONNX Session Constraint
=======================
Centralize ORT session creation so options are consistent across modules.
// Keep this minimal; extend only when truly needed.
"""

import logging
import platform

import onnxruntime as ort

logger = logging.getLogger(__name__)


def build_session(
    model_path: str,
    use_cuda: bool = False,
    use_coreml: bool = False,
    for_benchmarking: bool = True,
    intra_op_num_threads: int | None = None,
    # ↳ None -> do not set, keep ORT default (0 = auto); or set an explicit integer
    inter_op_num_threads: int | None = 1,
    # ↳ keep 1 unless parallel branches benefit
) -> ort.InferenceSession:
    """Create a lean ONNX Runtime session with stable settings for benchmarking/inference.

    - Enables full graph optimizations for realistic performance numbers.
    - Uses sequential execution to avoid scheduling overhead unless you have parallel branches.
    - Sets a moderate log level (WARNING+) to keep console noise low.
    - Chooses providers based on availability and request (CoreML > CUDA > CPU).

    Args:
        model_path: Path to ONNX model file
        use_cuda: Request CUDA execution if available
        use_coreml: Request CoreML execution on Apple Silicon
        for_benchmarking: If True, optimize for consistent measurements; if False, for deployment
        intra_op_num_threads: Parallelism within single operator (None=auto)
        inter_op_num_threads: Parallelism across operators (1=sequential)

    Returns:
        Configured ORT InferenceSession
    """
    so = ort.SessionOptions()

    # Enable aggressive graph optimizations (constant folding, fusion, etc.)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Prefer sequential execution unless your graph has real parallel branches.
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Keep logs quieter (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL).
    so.log_severity_level = 2

    # Memory pattern optimization: disable for benchmarking (consistency), enable for deployment (efficiency)
    so.enable_mem_pattern = not for_benchmarking

    # Optional: fix thread counts to reduce run-to-run variance
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = int(intra_op_num_threads)
        # ↳ intra: parallelism within a single operator (usually the main knob)
    if inter_op_num_threads is not None:
        so.inter_op_num_threads = int(inter_op_num_threads)
        # ↳ inter: parallelism across independent operators/nodes

    # Make execution deterministic when supported (slight perf impact possible)
    # // useful for benchmarking and reproducibility
    if for_benchmarking:
        try:
            so.add_session_config_entry("session.use_deterministic_compute", "1")
        except Exception:
            pass  # option may not exist in older ORT versions

    # Provider selection logic: prioritize by platform and request
    available = set(ort.get_available_providers())
    providers = []

    # Auto-detect Apple Silicon and suggest CoreML
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    if use_coreml or (is_apple_silicon and not use_cuda):
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
            logger.info("CoreML provider enabled for Apple Silicon acceleration")
        elif use_coreml:
            logger.warning("CoreML requested but not available")

    if use_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        logger.debug("CUDA provider enabled")
    elif use_cuda:
        logger.warning(
            "CUDA requested but not available; available_providers=%s",
            sorted(available),
        )

    # Always include CPU as fallback
    providers.append("CPUExecutionProvider")

    # Log provider selection
    logger.debug(
        "Providers selected: %s (from available: %s)", providers, sorted(available)
    )

    # DEBUG: compact summary of ORT session settings for reproducibility/troubleshooting.
    logger.debug(
        "ORT options -> graph_opt=%s, exec_mode=%s, intra=%s, inter=%s, ort_log=%s, mem_pattern=%s, benchmarking=%s",
        getattr(so.graph_optimization_level, "name", so.graph_optimization_level),
        getattr(so.execution_mode, "name", so.execution_mode),
        intra_op_num_threads if intra_op_num_threads is not None else "auto",
        inter_op_num_threads if inter_op_num_threads is not None else "auto",
        so.log_severity_level,
        not for_benchmarking,  # mem_pattern enabled = not benchmarking
        for_benchmarking,
    )

    return ort.InferenceSession(model_path, sess_options=so, providers=providers)
