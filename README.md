# GRPO Training for YARA Rule Generation

This repository implements Group Relative Policy Optimization (GRPO) training for teaching language models to generate high-quality YARA rules for malware detection.

## Quick Start

### 1. Setup Environment

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment
uv venv grpo_env
source grpo_env/bin/activate

# Install dependencies
uv pip install "axolotl[vllm]==0.9.2"
uv pip install --no-deps "trl @ git+https://github.com/huggingface/trl.git@main"
MAX_JOBS=64 uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install yara-python
```

### 2. Run Training

**Terminal 1 - Start vLLM Server (GPU 1):**
```bash
source grpo_env/bin/activate
CUDA_VISIBLE_DEVICES=1 axolotl vllm-serve orion_yara.yaml
```

**Terminal 2 - Start Training (GPU 0):**
```bash
source grpo_env/bin/activate
CUDA_VISIBLE_DEVICES=0 axolotl train orion_yara.yaml --num-processes 1
```

## Project Structure

- `orion_yara.yaml` - Main training configuration
- `orion_yara_grpo.py` - Custom reward functions for YARA rule quality
- `docs/` - Comprehensive documentation
  - `1_setup.md` - Installation guide
  - `2_grpo.md` - GRPO methodology
  - `3_yara_rewards.md` - YARA-specific rewards
  - `4_putting_it_all_together.md` - Complete workflow
  - `examples/gsm8k/` - Example implementation for math problems

## Key Features

- **YARA Syntax Validation**: Rewards syntactically correct rules using yara-python
- **Structure Rewards**: Encourages complete rule sections (meta, strings, condition)
- **Complexity Rewards**: Promotes sophisticated detection patterns
- **vLLM Acceleration**: Fast trajectory generation during GRPO training

## Hardware Requirements

- 2 GPUs with 24GB+ VRAM each (e.g., RTX 4090, L40S)
- 64GB+ system RAM
- CUDA 11.8+

## Configuration

The training uses the ctitools/orion_10k dataset, filtering for YARA generation tasks. Key parameters:

- Base model: Qwen3-1.7B (configurable)
- 16 response generations per prompt
- Custom reward functions for YARA quality
- Optimized for 2-GPU setup with vLLM

## Monitoring

Training progress is logged to Weights & Biases. Key metrics:
- Validity reward: Percentage of compilable rules
- Structure reward: Rule completeness scores
- Complexity reward: Rule sophistication metrics

## License

MIT