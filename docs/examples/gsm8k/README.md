# GSM8K GRPO Example

This example demonstrates training a language model on the GSM8K mathematics dataset using Group Relative Policy Optimization (GRPO). The model learns to generate structured XML responses with step-by-step reasoning.

## Overview

GSM8K (Grade School Math 8K) is a dataset of 8,500 high-quality grade school math problems that require 2-8 steps to solve. This example trains models to solve these problems while outputting structured responses in the following format:

```xml
<reasoning>
Step-by-step solution process
</reasoning>
<answer>
Final numerical answer
</answer>
```

## Files

- `gsm8k.yaml` - Training configuration for GRPO with vLLM acceleration
- `gsm8k_grpo.py` - Custom reward functions and data transformation for GSM8K

## Key Features

### Reward Functions

1. **Correctness Reward** (`correctness_reward_func`): 2.0 points for correct answers
2. **Format Validation Rewards**:
   - `strict_format_reward_func`: Validates exact XML format
   - `soft_format_reward_func`: More lenient format checking
   - `xmlcount_reward_func`: Rewards proper XML tag usage
3. **Integer Reward** (`int_reward_func`): Rewards integer answers when appropriate

### Configuration Highlights

- Uses Qwen3-4B as the base model
- vLLM acceleration on GPU 1 for trajectory generation
- 16 response generations per prompt for GRPO
- Optimized for 2-GPU setup with 24GB+ VRAM per GPU

## Usage

See the main documentation for setup and training instructions.