"""
GRPO training module for YARA rule generation from the ctitools/orion_10k dataset.
"""

import re
import yara
from typing import Dict, List, Optional, Tuple
# Define dataset constants for GRPO
class DatasetConstants:
    TEXT = "text"
    PROMPT = "prompt"
    CHOSEN = "chosen"
    REJECTED = "rejected"
from datasets import Dataset
import warnings

# Suppress YARA warnings during compilation
warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_yara_rule(text: str) -> Optional[str]:
    """
    Extract YARA rule from model output.
    Looks for rule blocks starting with 'rule' and ending with closing brace.
    """
    if not text:
        return None
    
    # Find all potential YARA rules in the text
    rule_pattern = r'rule\s+\w+[^{]*\{[^}]*\}'
    
    # First try to find a complete rule
    matches = re.findall(rule_pattern, text, re.DOTALL | re.MULTILINE)
    if matches:
        # Return the first complete rule found
        return matches[0].strip()
    
    # If no complete rule, try to extract partial rule if it starts with 'rule'
    if text.strip().startswith('rule '):
        # Find the last closing brace
        last_brace = text.rfind('}')
        if last_brace != -1:
            return text[:last_brace + 1].strip()
    
    return None


def validate_yara_rule(rule_text: str) -> Tuple[bool, str]:
    """
    Validate a YARA rule using yara-python.
    Returns (is_valid, error_message)
    """
    if not rule_text:
        return False, "No rule text provided"
    
    try:
        # Try to compile the rule
        yara.compile(source=rule_text)
        return True, ""
    except yara.SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except yara.Error as e:
        return False, f"YARA error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def yara_validity_reward_func(
    prompts: List[str],
    completions: List[str],
    labels: List[str] = None,
    context: Dict = None,
    **kwargs  # Accept any additional arguments
) -> List[float]:
    """
    Reward function based on YARA rule validity.
    - 2.0 points for a valid, compilable YARA rule
    - 0.5 points for a rule that starts correctly but has syntax errors
    - 0.0 points for no rule or completely invalid output
    """
    rewards = []
    
    for completion in completions:
        # Extract text from completion (may be a dict or list)
        if isinstance(completion, list) and len(completion) > 0:
            completion_text = completion[0].get('content', '') if isinstance(completion[0], dict) else str(completion[0])
        elif isinstance(completion, dict):
            completion_text = completion.get('content', '') or completion.get('text', '') or str(completion)
        else:
            completion_text = str(completion)
        
        # Extract YARA rule from completion
        rule_text = extract_yara_rule(completion_text)
        
        if not rule_text:
            rewards.append(0.0)
            continue
        
        # Validate the rule
        is_valid, error_msg = validate_yara_rule(rule_text)
        
        if is_valid:
            rewards.append(2.0)
        elif rule_text.startswith('rule ') and '{' in rule_text:
            # Partial credit for attempting a rule structure
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    
    return rewards


def yara_structure_reward_func(
    prompts: List[str],
    completions: List[str],
    labels: List[str] = None,
    context: Dict = None,
    **kwargs  # Accept any additional arguments
) -> List[float]:
    """
    Reward function for YARA rule structural completeness.
    Checks for presence of key components.
    """
    rewards = []
    
    for completion in completions:
        # Extract text from completion (may be a dict or list)
        if isinstance(completion, list) and len(completion) > 0:
            completion_text = completion[0].get('content', '') if isinstance(completion[0], dict) else str(completion[0])
        elif isinstance(completion, dict):
            completion_text = completion.get('content', '') or completion.get('text', '') or str(completion)
        else:
            completion_text = str(completion)
        
        rule_text = extract_yara_rule(completion_text)
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


def yara_complexity_reward_func(
    prompts: List[str],
    completions: List[str],
    labels: List[str] = None,
    context: Dict = None,
    **kwargs  # Accept any additional arguments
) -> List[float]:
    """
    Reward function for YARA rule complexity and quality.
    Rewards more sophisticated rules.
    """
    rewards = []
    
    for completion in completions:
        # Extract text from completion (may be a dict or list)
        if isinstance(completion, list) and len(completion) > 0:
            completion_text = completion[0].get('content', '') if isinstance(completion[0], dict) else str(completion[0])
        elif isinstance(completion, dict):
            completion_text = completion.get('content', '') or completion.get('text', '') or str(completion)
        else:
            completion_text = str(completion)
        
        rule_text = extract_yara_rule(completion_text)
        if not rule_text:
            rewards.append(0.0)
            continue
        
        score = 0.0
        
        # Count number of strings defined (up to 0.5 points)
        string_count = len(re.findall(r'\$\w+\s*=', rule_text))
        score += min(string_count * 0.1, 0.5)
        
        # Check for string modifiers (0.25 points)
        if any(mod in rule_text for mod in ['ascii', 'wide', 'nocase', 'fullword']):
            score += 0.25
        
        # Check for complex conditions (0.25 points)
        condition_match = re.search(r'condition:\s*(.+?)(?:}|$)', rule_text, re.DOTALL)
        if condition_match:
            condition = condition_match.group(1)
            # Complex conditions use 'and', 'or', 'of', etc.
            if any(op in condition for op in [' and ', ' or ', ' of ', 'any of', 'all of']):
                score += 0.25
        
        rewards.append(score)
    
    return rewards


def axo_orion_yara_transform(cfg, *args, **kwargs):
    """
    Transform orion_10k dataset samples for YARA rule generation training.
    Uses a dummy prompt for non-YARA samples to avoid filtering issues.
    """
    
    def transform_fn(sample, tokenizer=None):
        # Check if this is a YARA generation task
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
             
        return {
            "prompt": [
                {"role": "user", "content": input_text},
            ],
            "answer": output_text
        }
        
    
    return transform_fn, {"remove_columns": ["input", "output", "messages"]}


# For backward compatibility with existing code that might import specific functions
__all__ = [
    'extract_yara_rule',
    'validate_yara_rule', 
    'yara_validity_reward_func',
    'yara_structure_reward_func',
    'yara_complexity_reward_func',
    'axo_orion_yara_transform'
]