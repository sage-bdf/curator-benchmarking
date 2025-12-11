"""Custom prompt formatter for retrieve_access_restrictions task."""
import json
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for retrieve_access_restrictions task.

    Args:
        prompt_template: The base prompt template from default_prompt.txt
        sample: Input sample containing 'entityId'
        ground_truth: Ground truth sample (not used during inference)
        schema: Optional JSON schema (not used for this task)

    Returns:
        Formatted prompt string
    """
    entity_id = sample.get('entityId', '')

    # Build the prompt with entity ID
    formatted_prompt = f"{prompt_template}\n\nEntity ID to analyze: {entity_id}"

    return formatted_prompt
