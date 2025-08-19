import onnxruntime as ort


def _build_session(
    model_path: str, use_cuda: bool = False
) -> ort.InferenceSession:
    # Create SessionOptions to control optimization, threading, reproducibility, and diagnostics
    so = ort.SessionOptions()

    # Enable aggressive graph optimizations (constant folding, operator fusion, layout, etc.)
    # Good for measuring realistic end-to-end performance rather than debug-mode numbers.
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Fix thread counts to reduce run-to-run variance. Tune to your machine/workload.
    # intra_op: parallelism within a single operator (usually the main knob)
    # inter_op: parallelism across independent operators/nodes
    so.intra_op_num_threads = (
        0  # 0 lets ORT choose wisely; or set an explicit integer
    )
    so.inter_op_num_threads = (
        1  # keep 1 unless you see parallel branches benefiting
    )

    # Prefer sequential execution to avoid scheduling overhead unless the graph has parallel branches.
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Make execution deterministic when supported (slight perf impact possible).
    # Useful for benchmarking and reproducibility.
    try:
        so.add_session_config_entry("session.use_deterministic_compute", "1")
    except Exception:
        pass  # option may not exist in older ORT versions

    # If your input shapes vary a lot within the same session, consider disabling memory pattern
    # to avoid re-planning overhead per new shape. Keep enabled for fixed shapes.
    # so.enable_mem_pattern = False

    # Dump the optimized graph to inspect applied fusions/rewrites (optional, for debugging).
    # so.optimized_model_filepath = "model.optimized.onnx"

    # Enable built-in profiler to collect operator-wise timing (has overhead; enable when needed).
    # so.enable_profiling = True

    # Keep logs quieter unless diagnosing issues (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL).
    so.log_severity_level = 2

    # Choose execution providers in desired order (GPU first then CPU fallback).
    providers = ["CPUExecutionProvider"]
    if use_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ort.InferenceSession(
        model_path, sess_options=so, providers=providers
    )
