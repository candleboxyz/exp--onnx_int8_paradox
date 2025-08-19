# 문제 진단용 스크립트
import time

import numpy as np
import onnxruntime as ort


def diagnose_model(model_path):
    """Diagnose why INT8 is so slow"""
    print(f"\n=== Diagnosing {model_path} ===")

    # Check providers
    session = ort.InferenceSession(model_path)
    print(f"Active providers: {session.get_providers()}")

    # Check model metadata
    for provider in session.get_providers():
        print(f"\nProvider: {provider}")
        options = None
        try:
            options = session.get_provider_options()[provider]
            print(f"  Options: {options}")
        except Exception:
            if options is None:
                print("  No options available")
            raise

    # Check input/output details
    for inp in session.get_inputs():
        print(f"Input: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # Simple benchmark without our constraints
    dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup
    for _ in range(5):
        session.run(None, {"images": dummy})

    # Time 10 runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        session.run(None, {"images": dummy})
        times.append((time.perf_counter() - start) * 1000)

    print(f"Raw timing: {np.mean(times):.2f}ms (std: {np.std(times):.2f}ms)")

    # Check if quantized ops are actually being used
    import onnx

    model = onnx.load(model_path)
    quant_ops = [node.op_type for node in model.graph.node if "Quant" in node.op_type]
    print(f"Quantized operations found: {len(quant_ops)}")
    if quant_ops:
        print(f"  Types: {set(quant_ops)}")

    return np.mean(times)


# Run diagnosis
fp32_time = diagnose_model("output/model.onnx")
int8_time = diagnose_model("output/model_int8.onnx")

print("\n=== Summary ===")
print(f"FP32: {fp32_time:.2f}ms")
print(f"INT8: {int8_time:.2f}ms")
print(f"Slowdown: {int8_time/fp32_time:.1f}x")
