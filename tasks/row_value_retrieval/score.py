"""Custom scorer for row_value_retrieval task."""
import json
import re
from typing import Dict, Any, Optional


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()
    
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    
    return text


def _structured_score(
    prediction: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> float:
    """
    Score structured data (dictionaries) using strict exact matching.
    
    Adherence to the prompt (exact values, correct format) is part of the test.
    """
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return 0.0
    
    # Calculate field-level accuracy with strict matching
    all_keys = set(prediction.keys()) | set(ground_truth.keys())
    if not all_keys:
        return 1.0  # Both empty, perfect match
    
    matches = 0
    for key in all_keys:
        pred_val = prediction.get(key)
        truth_val = ground_truth.get(key)
        
        # Strict exact match only - no fuzzy matching
        if pred_val == truth_val:
            matches += 1
    
    return matches / len(all_keys)


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """Score row value retrieval by comparing JSON objects with column-value pairs."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        expected_values_str = ground_truth.get('expected_values', '{}')
        try:
            if isinstance(expected_values_str, str):
                expected_values = json.loads(expected_values_str)
            else:
                expected_values = expected_values_str
        except (json.JSONDecodeError, TypeError):
            expected_values = {}
        
        if not isinstance(pred_dict, dict) or not isinstance(expected_values, dict):
            return 0.0
        
        # Use structured scoring to compare the dictionaries
        return _structured_score(pred_dict, expected_values)
    except Exception as e:
        print(f"Error scoring row value retrieval: {e}")
        return None

