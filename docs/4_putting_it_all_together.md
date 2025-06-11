# Putting It All Together: Complete Training Workflow

This guide walks through a complete GRPO training workflow, from setup to monitoring results.

## Step 1: Environment Setup

```bash
# 1. Install UV and create environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Create and activate virtual environment
uv venv grpo_env
source grpo_env/bin/activate

# 3. Install all dependencies
uv pip install "axolotl[vllm]"
uv pip install --no-deps "trl @ git+https://github.com/huggingface/trl.git@main"
uv pip install yara-python  # For YARA training
```

## Step 2: Create Configuration File

Create `orion_yara.yaml`:

```yaml
base_model: Qwen/Qwen3-1.7B
load_in_8bit: false
load_in_4bit: false
strict: false

torch_compile: true

# vLLM configuration
vllm:
  host: 0.0.0.0
  port: 8000
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.65
  dtype: auto
  max_model_len: 800

# GRPO configuration
rl: grpo
trl:
  beta: 0.001
  max_completion_length: 600
  use_vllm: true
  vllm_server_host: 0.0.0.0
  vllm_server_port: 8000
  vllm_server_timeout: 300
  reward_funcs:
    - orion_yara_grpo.yara_validity_reward_func
    - orion_yara_grpo.yara_structure_reward_func
    - orion_yara_grpo.yara_complexity_reward_func
  num_generations: 16

chat_template: qwen3
datasets:
  - path: ctitools/orion_10k
    type: orion_yara_grpo.axo_orion_yara_transform

dataset_prepared_path: /workspace/data/orion_yara_prepared
val_set_size: 0.0
output_dir: /workspace/data/axolotl-artifacts/orion-yara-outputs

# Performance settings
dataloader_prefetch_factor: 32
dataloader_num_workers: 2
dataloader_pin_memory: true
gc_steps: 1

sequence_len: 800
sample_packing: false
pad_to_sequence_len: false

# Weights & Biases logging
wandb_project: orion-yara-grpo
wandb_entity:  # Your W&B username/org
wandb_name:    # Run name (optional)

# Training hyperparameters
gradient_accumulation_steps: 1
micro_batch_size: 16
num_epochs: 1
max_steps: 100  # Adjust for full training

optimizer: adamw_torch_fused
lr_scheduler: constant_with_warmup
learning_rate: 1.0e-6
max_grad_norm: 1.0
weight_decay: 0.1

bf16: true
tf32: true

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
flash_attention: true

logging_steps: 1
warmup_steps: 10
evals_per_epoch: 0
saves_per_epoch: 1
save_strategy: "steps"
save_steps: 50
```

## Step 3: Create Reward Module

Create `orion_yara_grpo.py` with reward functions (see [YARA Rewards Documentation](3_yara_rewards.md) for full code).

## Step 4: Launch Training

### Terminal 1: Start vLLM Server

```bash
source grpo_env/bin/activate
CUDA_VISIBLE_DEVICES=1 axolotl vllm-serve orion_yara.yaml
```

Wait for "Application startup complete" message (usually 1-2 minutes).

### Terminal 2: Start Training

```bash
source grpo_env/bin/activate
CUDA_VISIBLE_DEVICES=0 axolotl train orion_yara.yaml --num-processes 1
```

## Step 5: Monitor Training

### Weights & Biases Dashboard

Monitor key metrics:
- **rewards/*/mean**: Should increase over time
- **kl**: Should stay reasonable (< 10)
- **completions/mean_length**: Track generation lengths
- **loss**: Should decrease

### Log Files

```bash
# Watch training progress
tail -f wandb/latest-run/files/output.log

# Check vLLM server status
tail -f vllm_yara_server.log
```

## Step 6: Evaluate Results

### Manual Testing

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/data/axolotl-artifacts/orion-yara-outputs/checkpoint-100"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

prompt = "Generate a YARA rule to detect a Windows executable"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=600)
print(tokenizer.decode(outputs[0]))
```

### Batch Evaluation

```python
import yara
from orion_yara_grpo import extract_yara_rule, validate_yara_rule

# Load test prompts
test_prompts = [
    "Generate a YARA rule for ransomware detection",
    "Create a YARA rule to identify PDF malware",
    # ... more prompts
]

# Generate and evaluate
valid_count = 0
for prompt in test_prompts:
    # Generate response
    response = generate_response(model, tokenizer, prompt)
    
    # Extract and validate
    rule = extract_yara_rule(response)
    if rule:
        is_valid, error = validate_yara_rule(rule)
        if is_valid:
            valid_count += 1

print(f"Valid rules: {valid_count}/{len(test_prompts)}")
```

## Common Issues and Solutions

### 1. Memory Issues

```yaml
# Reduce these values in config:
micro_batch_size: 8  # Half the batch size
vllm:
  gpu_memory_utilization: 0.5  # Reduce vLLM memory
max_completion_length: 400  # Shorter generations
```

### 2. Slow Training

```yaml
# Enable optimizations:
torch_compile: true
tf32: true
dataloader_num_workers: 4  # Increase workers
```

### 3. Poor Quality Rules

- Adjust reward weights
- Add more specific reward functions
- Increase training steps
- Use larger base model

### 4. Dataset Filtering Issues

Check filtered dataset size:

```python
from datasets import load_dataset

ds = load_dataset("ctitools/orion_10k")
yara_samples = [s for s in ds['train'] 
                if "generate a yara rule" in s['input'].lower()
                and s['output'].strip().startswith("rule ")]
print(f"YARA samples: {len(yara_samples)}")
```

## Production Tips

1. **Save Checkpoints**: Enable periodic saving for recovery
2. **Use Larger Models**: Qwen3-4B or larger for better quality
3. **Extended Training**: Run for multiple epochs on full dataset
4. **Hyperparameter Tuning**: Experiment with learning rates and KL penalty
5. **Ensemble Rewards**: Combine multiple reward signals carefully

## Next Steps

- Experiment with different base models
- Create domain-specific reward functions
- Fine-tune on your own datasets
- Deploy trained models for inference

## Additional Resources

- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [YARA Documentation](https://yara.readthedocs.io/)