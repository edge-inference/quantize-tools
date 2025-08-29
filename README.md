# LLM Compression Toolkit

A collection of helper scripts for quantizing and evaluating large language models using various compression techniques.

## Current Support

### AWQ (Activation-aware Weight Quantization)

- **Framework**: llm-compressor (vLLM ecosystem)
- **Quantization**: W4A16 (4-bit weights, 16-bit activations)
- **Features**:
  - Automated oneshot calibration
  - Configurable group size (default: 128)
  - Parallel processing (CPU workers)
  - Memory-optimized settings
  - vLLM-compatible outputs

## Quick Start

### Prerequisites

```bash
pip install llmcompressor vllm lm-eval
```

### Quantize with AWQ

1. Run quantization:
```bash
# Default: 2048 cal samples, 3072 seq len, WikiText-103
./run_llmcompressor.sh

# Custom settings
MAX_CALIB_SAMPLES=1024 MAX_SEQ_LEN=2048 ./run_llmcompressor.sh
```

2. Evaluate perplexity:
```bash
# Compare baseline vs quantized
./benchmark.sh
```

3. Test inference:
```bash
python eval.py  
```

### Outputs

- Quantized model: `./models/deepseek-r1-llama-8b-llmc/`
- Benchmarks: `./bench_out/<timestamp>/`
- Logs: `./sparse_logs/`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Model to quantize |
| `OUT_DIR` | `./models/deepseek-r1-llama-8b-llmc` | Output directory |
| `DATASET_NAME` | `wikitext` | Calibration dataset (registry name) |
| `DATASET_CONFIG_NAME` | `wikitext-103-raw-v1` | Dataset config |
| `MAX_CALIB_SAMPLES` | `2048` | Calibration samples |
| `MAX_SEQ_LEN` | `3072` | Max sequence length |
| `SCHEME` | `W4A16` | Quantization scheme |
| `PREPROC_WORKERS` | `$(nproc)` | CPU workers for preprocessing |

### Advanced Options

- `PAD_TO_MAX_LEN=false`: Avoid padding to save VRAM
- `SHUFFLE_CALIB=true`: Shuffle calibration samples
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Reduce CUDA fragmentation

## Future Support

Planned additions:
- **GPTQ** (Gradient-based PTQ)
- **SmoothQuant** (Activation quantization)
- **FP8** (Mixed precision)
- **SparseGPT** (Weight sparsity)
- Custom dataset support (JSON/CSV)

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `MAX_SEQ_LEN` (e.g., 2048)
   - Set `PAD_TO_MAX_LEN=false`
   - Lower `MAX_CALIB_SAMPLES`

2. **Dataset Not Found**:
   - Use registered datasets: `wikitext`, `c4`, `open-platypus`
   - For custom: set `DATASET_NAME=custom` and `DATASET_PATH=/path/to/data.json`

3. **Low Perplexity Improvement**:
   - Increase `MAX_CALIB_SAMPLES`
   - Use domain-matched calibration data
   - Try different schemes (W4A16_ASYM)

## Performance Notes

- AWQ W4A16 typically achieves ~1.5x speedup vs FP16
- Memory reduction: ~60–70% (5GB vs 15GB for 8B models)
- Perplexity increase: Usually 5–10% on general tasks

## Contributing

Add new compression methods by:
1. Creating a new script (e.g., `run_gptq.sh`)
2. Updating this README
3. Adding benchmark support

## License

MIT License
