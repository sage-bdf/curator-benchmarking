"""Custom prompt formatter for search_query_generation task."""
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for search_query_generation task.

    Args:
        prompt_template: The base prompt template from default_prompt.txt
        sample: Input sample containing 'queryPhrase' and 'platform' fields
        ground_truth: Ground truth sample (not used for formatting)
        schema: Optional JSON schema (not used for this task)

    Returns:
        Formatted prompt string
    """
    query_phrase = sample.get('queryPhrase', '')
    platform = sample.get('platform', '')

    # Build the prompt
    formatted_prompt = f"{prompt_template}\n\n"
    formatted_prompt += f"Query phrase: \"{query_phrase}\"\n"
    formatted_prompt += f"Target platform: {platform}\n"

    return formatted_prompt

