# Setup Guide

This guide walks you through setting up the GRPO training environment for LLMs using Axolotl and vLLM.

## Prerequisites

- Linux system with at least 2 GPUs (24GB+ VRAM each recommended)
- Python 3.10+
- CUDA 11.8 or higher
- Git

## Installation

### 1. Install UV Package Manager (Recommended)

UV is a fast Python package installer that handles dependencies efficiently:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Create Virtual Environment

```bash
uv venv grpo_env
source grpo_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install axolotl with vLLM support
uv pip install "axolotl[vllm]"

# Install TRL from main branch for latest GRPO features
uv pip install --no-deps "trl @ git+https://github.com/huggingface/trl.git@main"

# Optional: Install flash attention for better performance
uv pip install flash-attn --no-build-isolation
```

### 4. Additional Dependencies for YARA Training

If you plan to work with YARA rule generation:

```bash
uv pip install yara-python
```

## Hardware Requirements

### Minimum Requirements
- 2 GPUs with 16GB VRAM each
- 32GB system RAM
- 100GB free disk space

### Recommended Requirements
- 2 GPUs with 24GB+ VRAM each (e.g., RTX 4090, L40S, A100)
- 64GB+ system RAM
- 500GB+ fast SSD storage

## GPU Configuration

GRPO with vLLM requires a specific GPU setup:

1. **GPU 0**: Used for training
2. **GPU 1**: Used for vLLM inference server

**Important**: Due to TRL's implementation, vLLM must use the last N GPUs. In a 2-GPU system, this means GPU 1.

## Environment Variables

Set these environment variables for optimal performance:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your system
```

## Verifying Installation

Test your installation:

```bash
# Check axolotl installation
axolotl --help

# Check GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# Test vLLM import
python -c "import vllm; print('vLLM installed successfully')"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `micro_batch_size` in configuration
- Lower `gpu_memory_utilization` for vLLM
- Use smaller models or enable gradient checkpointing

### NCCL Communication Errors
- Ensure vLLM is using the last GPU (GPU 1 in 2-GPU setup)
- Check that both GPUs are visible and functioning

### Flash Attention Issues
- If installation fails, disable with `flash_attention: false` in config
- Ensure CUDA toolkit matches PyTorch version

## Next Steps

Once setup is complete, proceed to the [GRPO Documentation](2_grpo.md) to understand the training methodology.