"""Custom scorer for column_enumeration task."""
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


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """Score column enumeration task by comparing column lists."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        pred_columns = pred_dict.get('columns', [])
        if not isinstance(pred_columns, list):
            return 0.0
        
        expected_columns_str = ground_truth.get('expected_columns', '[]')
        try:
            if isinstance(expected_columns_str, str):
                expected_columns = json.loads(expected_columns_str)
            else:
                expected_columns = expected_columns_str
        except (json.JSONDecodeError, TypeError):
            expected_columns = []
        
        # Normalize to lists of strings
        pred_columns = [str(c) for c in pred_columns]
        expected_columns = [str(c) for c in expected_columns]
        
        # Exact match (order matters)
        if pred_columns == expected_columns:
            return 1.0
        # Same columns, different order
        if set(pred_columns) == set(expected_columns) and len(pred_columns) == len(expected_columns):
            return 0.5
        # Partial match
        if len(expected_columns) > 0:
            matched = len(set(pred_columns) & set(expected_columns))
            return matched / len(expected_columns)
        return 0.0
    except Exception as e:
        print(f"Error scoring column enumeration: {e}")
        return None

