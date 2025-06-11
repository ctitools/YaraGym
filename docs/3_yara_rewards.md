# YARA Rule Generation with Custom Rewards

This document explains how to implement GRPO training for YARA rule generation using custom reward functions.

## Overview

YARA is a pattern matching engine designed for malware identification and classification. Training an LLM to generate valid, high-quality YARA rules requires specialized reward functions that evaluate:

1. **Syntactic Validity**: Rules compile without errors
2. **Structural Completeness**: Rules contain required sections
3. **Complexity**: Rules are sophisticated enough to be useful

## Implementation

### 1. YARA Rule Extraction

First, we need to extract YARA rules from model outputs:

```python
def extract_yara_rule(text: str) -> Optional[str]:
    """Extract YARA rule from model output."""
    if not text:
        return None
    
    # Find complete rule blocks
    rule_pattern = r'rule\s+\w+[^{]*\{[^}]*\}'
    matches = re.findall(rule_pattern, text, re.DOTALL | re.MULTILINE)
    
    if matches:
        return matches[0].strip()
    
    return None
```

### 2. Validation with yara-python

Validate rules using the official YARA Python bindings:

```python
def validate_yara_rule(rule_text: str) -> Tuple[bool, str]:
    """Validate a YARA rule using yara-python."""
    try:
        yara.compile(source=rule_text)
        return True, ""
    except yara.SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"
```

### 3. Reward Functions

#### Validity Reward
Rewards syntactically correct rules:

```python
def yara_validity_reward_func(
    prompts: List[str],
    completions: List[str],
    labels: List[str],
    context: Dict = None,
) -> List[float]:
    """
    - 2.0 points for valid, compilable YARA rule
    - 0.5 points for partial rule structure
    - 0.0 points for invalid output
    """
    rewards = []
    for completion in completions:
        rule_text = extract_yara_rule(completion)
        if not rule_text:
            rewards.append(0.0)
            continue
            
        is_valid, _ = validate_yara_rule(rule_text)
        if is_valid:
            rewards.append(2.0)
        elif rule_text.startswith('rule ') and '{' in rule_text:
            rewards.append(0.5)  # Partial credit
        else:
            rewards.append(0.0)
    
    return rewards
```

#### Structure Reward
Evaluates rule completeness:

```python
def yara_structure_reward_func(...) -> List[float]:
    """Rewards presence of key YARA components."""
    rewards = []
    for completion in completions:
        rule_text = extract_yara_rule(completion)
        if not rule_text:
            rewards.append(0.0)
            continue
            
        score = 0.0
        # Check for rule declaration (0.25 points)
        if re.match(r'rule\s+\w+', rule_text):
            score += 0.25
        # Check for meta section (0.25 points)  
        if 'meta:' in rule_text:
            score += 0.25
        # Check for strings section (0.25 points)
        if 'strings:' in rule_text:
            score += 0.25
        # Check for condition section (0.25 points)
        if 'condition:' in rule_text:
            score += 0.25
            
        rewards.append(score)
    
    return rewards
```

#### Complexity Reward
Encourages sophisticated rules:

```python
def yara_complexity_reward_func(...) -> List[float]:
    """Rewards rule sophistication."""
    rewards = []
    for completion in completions:
        rule_text = extract_yara_rule(completion)
        if not rule_text:
            rewards.append(0.0)
            continue
            
        score = 0.0
        
        # Count string definitions (up to 0.5 points)
        string_count = len(re.findall(r'\$\w+\s*=', rule_text))
        score += min(string_count * 0.1, 0.5)
        
        # Check for string modifiers (0.25 points)
        modifiers = ['ascii', 'wide', 'nocase', 'fullword']
        if any(mod in rule_text for mod in modifiers):
            score += 0.25
            
        # Check for complex conditions (0.25 points)
        condition_match = re.search(r'condition:\s*(.+?)(?:}|$)', 
                                   rule_text, re.DOTALL)
        if condition_match:
            condition = condition_match.group(1)
            operators = [' and ', ' or ', ' of ', 'any of', 'all of']
            if any(op in condition for op in operators):
                score += 0.25
                
        rewards.append(score)
    
    return rewards
```

### 4. Dataset Transformation

Transform the orion_10k dataset for GRPO training:

```python
def axo_orion_yara_transform(cfg, *args, **kwargs):
    """Transform orion_10k samples for YARA training."""
    
    def transform_fn(sample, tokenizer=None):
        input_text = sample.get("input", "").lower()
        output_text = sample.get("output", "")
        
        # Only process YARA generation requests
        if "generate a yara rule" not in input_text:
            return {"prompt": [], "answer": ""}
            
        # Skip if output doesn't contain a rule
        if not output_text or not output_text.strip().startswith("rule "):
            return {"prompt": [], "answer": ""}
            
        return {
            "prompt": [
                {"role": "user", "content": sample["input"]},
            ],
            "answer": output_text
        }
    
    return transform_fn, {"remove_columns": ["input", "output", "messages"]}
```

## Configuration

Example configuration for YARA GRPO training:

```yaml
base_model: Qwen/Qwen3-1.7B
rl: grpo

trl:
  reward_funcs:
    - orion_yara_grpo.yara_validity_reward_func
    - orion_yara_grpo.yara_structure_reward_func  
    - orion_yara_grpo.yara_complexity_reward_func
  num_generations: 16
  max_completion_length: 600

datasets:
  - path: ctitools/orion_10k
    type: orion_yara_grpo.axo_orion_yara_transform
```

## Best Practices

1. **Balance Rewards**: Ensure no single reward dominates
2. **Test Thoroughly**: Validate rewards produce desired behavior
3. **Monitor Quality**: Check generated rules during training
4. **Iterate**: Refine rewards based on output quality

## Common Patterns in YARA Rules

When designing rewards, consider these common patterns:

```yara
rule ExampleRule {
    meta:
        author = "Security Researcher"
        description = "Detects specific malware"
        
    strings:
        $str1 = "malicious_string" ascii
        $hex1 = { 4D 5A 90 00 }
        $regex1 = /pattern[0-9]{2}/ nocase
        
    condition:
        uint16(0) == 0x5A4D and
        any of ($str*) and
        filesize < 1MB
}
```

## Debugging Tips

1. **Log Rewards**: Print reward distributions to understand scoring
2. **Inspect Failures**: Analyze rules that fail validation
3. **Test Edge Cases**: Ensure rewards handle partial/malformed rules
4. **Visualize Progress**: Plot reward trends over training

## Next Steps

See [Putting It All Together](4_putting_it_all_together.md) for a complete training workflow.