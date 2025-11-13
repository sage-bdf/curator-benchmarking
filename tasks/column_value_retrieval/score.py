"""Custom scorer for column_value_retrieval task."""
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
    """Score column value retrieval by comparing values."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        pred_value = pred_dict.get('value')
        expected_value = ground_truth.get('expected_value')
        
        # Normalize for comparison (handle null, numbers, strings)
        if pred_value is None and expected_value is None:
            return 1.0
        if pred_value is None or expected_value is None:
            return 0.0
        
        # Convert to strings for comparison, but preserve type if possible
        if isinstance(pred_value, (int, float)) and isinstance(expected_value, (int, float)):
            if abs(float(pred_value) - float(expected_value)) < 0.0001:
                return 1.0
            return 0.0
        
        # String comparison (case-insensitive for flexibility)
        if str(pred_value).strip().lower() == str(expected_value).strip().lower():
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error scoring column value retrieval: {e}")
        return None

