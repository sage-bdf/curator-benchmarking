"""Custom prompt formatter for regex_generation task."""
import json
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for regex_generation task.
    
    Args:
        prompt_template: The base prompt template from default_prompt.txt
        sample: Input sample containing 'filenames' as a JSON array string
        ground_truth: Ground truth sample containing 'matches' as a JSON array string
        schema: Optional JSON schema (not used for this task)
        
    Returns:
        Formatted prompt string
    """
    # Parse filenames from input
    filenames_str = sample.get('filenames', '[]')
    try:
        if isinstance(filenames_str, str):
            filenames = json.loads(filenames_str)
        else:
            filenames = filenames_str
    except (json.JSONDecodeError, TypeError):
        filenames = []
    
    # Build the prompt
    formatted_prompt = f"{prompt_template}\n\nAll filenames:\n"
    for filename in filenames:
        formatted_prompt += f"- {filename}\n"
    
    # Add example matches if ground truth is available
    if ground_truth:
        gt_matches_str = ground_truth.get('matches', '[]')
        try:
            if isinstance(gt_matches_str, str):
                gt_matches = json.loads(gt_matches_str)
            else:
                gt_matches = gt_matches_str
        except (json.JSONDecodeError, TypeError):
            gt_matches = []
        
        # Provide 3 example matches
        num_examples = min(3, len(gt_matches))
        example_matches = gt_matches[:num_examples] if gt_matches else []
        
        if example_matches:
            formatted_prompt += f"\nExample matches (showing what should be extracted):\n"
            for i, match in enumerate(example_matches):
                # Find the corresponding filename
                if i < len(filenames):
                    formatted_prompt += f"- From '{filenames[i]}': {json.dumps(match)}\n"
    
    return formatted_prompt

