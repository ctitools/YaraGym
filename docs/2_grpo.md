# GRPO (Group Relative Policy Optimization)

## Overview

Group Relative Policy Optimization (GRPO) is a reinforcement learning method for fine-tuning language models based on reward signals. It's particularly effective for tasks where you want to optimize specific behaviors or outputs while maintaining the model's general capabilities.

## How GRPO Works

### 1. Generation Phase
- The model generates multiple responses (typically 16) for each prompt
- These responses are sampled with some randomness to ensure diversity
- vLLM accelerates this phase by handling batch generation efficiently

### 2. Reward Calculation
- Each generated response is evaluated by custom reward functions
- Multiple reward functions can be combined (e.g., correctness + format)
- Rewards guide the model toward desired behaviors

### 3. Optimization
- Responses are ranked by their reward scores
- The model is updated to increase the likelihood of high-reward responses
- KL divergence penalty prevents the model from deviating too far from its initial behavior

## Key Configuration Parameters

### Training Parameters

```yaml
rl: grpo  # Specify GRPO as the RL method
trl:
  beta: 0.001  # KL divergence coefficient
  max_completion_length: 512  # Maximum tokens to generate
  num_generations: 16  # Responses per prompt
  use_vllm: true  # Enable vLLM acceleration
```

### vLLM Configuration

```yaml
vllm:
  host: 0.0.0.0
  port: 8000
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.7  # Adjust based on GPU memory
  max_model_len: 800  # Should match sequence_len
```

### Optimization Settings

```yaml
learning_rate: 1.0e-6  # Lower than supervised fine-tuning
gradient_accumulation_steps: 1
micro_batch_size: 16  # Should match num_generations for single GPU
```

## Reward Functions

Reward functions are Python functions that evaluate generated responses:

```python
def my_reward_func(
    prompts: List[str],
    completions: List[str], 
    labels: List[str],
    context: Dict = None,
) -> List[float]:
    """
    Returns a reward score for each completion.
    Higher scores indicate better responses.
    """
    rewards = []
    for completion in completions:
        # Evaluate completion
        score = evaluate_response(completion)
        rewards.append(score)
    return rewards
```

### Best Practices for Reward Functions

1. **Scale Appropriately**: Keep rewards in a reasonable range (e.g., 0-2)
2. **Be Differentiable**: Provide gradients of quality, not just binary pass/fail
3. **Combine Multiple Signals**: Use multiple reward functions for different aspects
4. **Test Thoroughly**: Ensure rewards align with your actual goals

## Data Format

GRPO requires a specific data format with prompts and optional reference answers:

```python
def transform_fn(example, tokenizer=None):
    return {
        "prompt": [
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["reference_answer"]  # Optional
    }
```

## Memory Optimization

GRPO is memory-intensive due to multiple generation passes:

1. **Reduce Batch Size**: Lower `micro_batch_size` if OOM
2. **Shorter Sequences**: Reduce `max_completion_length` and `sequence_len`
3. **Gradient Checkpointing**: Enable with `gradient_checkpointing: true`
4. **Lower vLLM Memory**: Adjust `gpu_memory_utilization` (e.g., 0.5)

## Monitoring Training

Key metrics to watch:

- **Reward Mean/Std**: Should increase over time
- **KL Divergence**: Should stay reasonable (< 10)
- **Clip Ratio**: High values indicate too large policy updates
- **Generation Length**: Monitor if model outputs are being truncated

## Common Issues

### 1. Reward Hacking
- Model finds unintended ways to maximize reward
- Solution: More comprehensive reward functions

### 2. Mode Collapse  
- Model generates very similar responses
- Solution: Increase temperature, add diversity rewards

### 3. Forgetting
- Model loses general capabilities
- Solution: Lower learning rate, increase KL penalty

## Next Steps

Learn about implementing custom reward functions in the [YARA Rewards Documentation](3_yara_rewards.md).