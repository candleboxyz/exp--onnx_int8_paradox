# Edge AI Optimization Report

## Model Analysis
- Total Operations: 262
- Parameters: 2,691,221.0
- Parameter Memory: 10.27 MB
- Fusion Opportunities: 0

## Size Optimization
- Original: 10.33 MB
- Quantized: 2.87 MB
- Compression: 3.60x

## Performance Analysis

### Controlled Benchmark (Single-thread)
**Purpose**: Scientific comparison with identical conditions
**Configuration**: `intra_op_threads=1`, `inter_op_threads=1`

| Model | Mean Latency | FPS | CV | Note |
|-------|-------------|-----|-----|------|
| FP32  | 4.03ms | 248.3 | 0.114 | Baseline |
| INT8  | 148.29ms | 6.7 | 0.019 | Thread bottleneck |

**Speedup**: 0.027x

### Practical Benchmark (Auto-threading)
**Purpose**: Real-world deployment performance
**Configuration**: threads=auto

| Model | Mean Latency | FPS | CV | Note |
|-------|-------------|-----|-----|------|
| FP32  | 5.83ms | 171.6 | 0.127 | Multi-thread overhead |
| INT8  | 63.92ms | 15.6 | 0.048 | Parallelized quantization |

**Speedup**: 0.091x

## Key Findings

1. **Thread Constraints Impact**: INT8 models with DynamicQuantizeLinear operations 
   require parallel execution for acceptable performance.
   
2. **Platform Dependencies**: Performance characteristics vary significantly 
   based on hardware capabilities and runtime configuration.

3. **Size vs Speed Trade-off**: While 3.6x size reduction is achieved consistently,
   speed improvements are highly dependent on deployment conditions.
