"""Custom prompt formatter for retrieve_ACLs task."""
import json
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for retrieve_ACLs task.

    Args:
        prompt_template: The base prompt template from default_prompt.txt
        sample: Input sample containing 'requirementId'
        ground_truth: Ground truth sample (not used during inference)
        schema: Optional JSON schema (not used for this task)

    Returns:
        Formatted prompt string
    """
    requirement_id = sample.get('requirementId', '')

    # Build the prompt with requirement ID
    formatted_prompt = f"{prompt_template}\n\nAccess Requirement ID to analyze: {requirement_id}"

    return formatted_prompt
