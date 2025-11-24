"""Custom prompt formatter for grid_request task."""
import json
from typing import Dict, Any, Optional

def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for grid_request task.
    
    Args:
        prompt_template: The base prompt template from default_prompt.txt
        sample: Input sample data
        ground_truth: Ground truth sample (not used for formatting)
        schema: JSON schema (default used if not specified)
        
    Returns:
        Formatted prompt string
    """
    # Format prompt with sample data
    sample_str = json.dumps(sample, indent=2)
    return f"{prompt_template}\n\nUser request:\n{sample_str}"
