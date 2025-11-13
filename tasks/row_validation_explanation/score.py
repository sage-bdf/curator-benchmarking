"""Custom scorer for row_validation_explanation task."""
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
    """Score row validation explanation by checking for expected keywords."""
    try:
        json_str = _extract_json(prediction)
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, check the raw text
            explanation_text = prediction.lower()
        else:
            explanation_text = pred_dict.get('explanation', '').lower()
        
        expected_keywords_str = ground_truth.get('expected_explanation_keywords', '[]')
        try:
            if isinstance(expected_keywords_str, str):
                expected_keywords = json.loads(expected_keywords_str)
            else:
                expected_keywords = expected_keywords_str
        except (json.JSONDecodeError, TypeError):
            expected_keywords = []
        
        if not expected_keywords:
            return 0.0
        
        # Check how many keywords are present
        found_keywords = [kw for kw in expected_keywords if kw.lower() in explanation_text]
        
        return len(found_keywords) / len(expected_keywords)
    except Exception as e:
        print(f"Error scoring row validation explanation: {e}")
        return None

