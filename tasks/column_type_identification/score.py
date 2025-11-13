"""Custom scorer for column_type_identification task."""
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
    """Score column type identification by comparing type strings."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        pred_type = pred_dict.get('type', '').lower().strip()
        expected_type = str(ground_truth.get('expected_type', '')).lower().strip()
        
        if not pred_type or not expected_type:
            return 0.0
        
        # Exact match
        if pred_type == expected_type:
            return 1.0
        
        # Handle common variations
        type_variations = {
            'int': 'integer',
            'integer': 'integer',
            'float': 'number',
            'double': 'number',
            'number': 'number',
            'num': 'number',
            'str': 'string',
            'string': 'string',
            'text': 'string',
            'date': 'date',
            'datetime': 'datetime',
            'timestamp': 'datetime',
            'bool': 'boolean',
            'boolean': 'boolean'
        }
        pred_normalized = type_variations.get(pred_type, pred_type)
        expected_normalized = type_variations.get(expected_type, expected_type)
        
        if pred_normalized == expected_normalized:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error scoring column type identification: {e}")
        return None

