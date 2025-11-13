"""Custom scorer for validation_error_counting task."""
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
    """Score validation error counting task by comparing counts."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        pred_count = pred_dict.get('count')
        if pred_count is None:
            # Try to extract number from text response
            numbers = re.findall(r'\d+', prediction)
            if numbers:
                pred_count = int(numbers[0])
            else:
                return 0.0
        
        expected_count = ground_truth.get('expected_count')
        if expected_count is None:
            return 0.0
        
        # Convert to int for comparison
        try:
            pred_count = int(pred_count)
            expected_count = int(expected_count)
        except (ValueError, TypeError):
            return 0.0
        
        if pred_count == expected_count:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error scoring validation error counting: {e}")
        return None

