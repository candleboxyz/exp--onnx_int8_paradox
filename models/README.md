# Models Directory

## Structure
- Original models (.pt, .pth)
- ONNX exports (.onnx)
- Quantized models (*_int8.onnx)
- Optimized models (*_optimized.onnx)

## Naming Convention
- `{model}_{resolution}.onnx` - Base ONNX model
- `{model}_{resolution}_int8.onnx` - INT8 quantized
- `{model}_{resolution}_pruned.onnx` - Pruned model

