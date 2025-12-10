"""Custom scorer for grid_request task."""
import json
import re
from typing import Dict, Any, Optional


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """Score grid_request task by comparing JSON objects. Enforces raw JSON output."""
    try:
        # Parse prediction
        try:
            pred_obj = json.loads(prediction.strip())
            # If prediction was an escaped JSON string, pred_obj will be a string.
            # We need to parse it again to get the dict.
            if isinstance(pred_obj, str):
                pred_dict = json.loads(pred_obj)
            else:
                pred_dict = pred_obj
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse prediction as JSON: {prediction[:100]}...")
            return 0.0
        
        # Parse ground truth
        # ground_truth['request'] is now the inner JSON object (as a string)
        gt_request_str = ground_truth.get('request', '{}')
        try:
            if isinstance(gt_request_str, str):
                gt_dict = json.loads(gt_request_str)
            else:
                gt_dict = gt_request_str
        except (json.JSONDecodeError, TypeError):
            print(f"Failed to parse ground truth as JSON: {gt_request_str[:100]}...")
            return 0.0
        
        # Canonicalize and compare
        # Check if limit is required (default to False)
        # Handle string "false" or boolean False
        limit_required = ground_truth.get('limit_required', False)
        if isinstance(limit_required, str):
            limit_required = limit_required.lower() == 'true'
            
        # Normalize both prediction and ground truth
        def _normalize_recursively(obj):
            """Remove limit (if not required) and empty filters arrays."""
            if isinstance(obj, dict):
                # Remove limit if not required
                if not limit_required:
                    obj.pop('limit', None)
                # Remove empty filters arrays
                if 'filters' in obj and obj['filters'] == []:
                    obj.pop('filters', None)
                # Recurse into nested structures
                for key, value in list(obj.items()):
                    _normalize_recursively(value)
            elif isinstance(obj, list):
                for item in obj:
                    _normalize_recursively(item)

        if isinstance(pred_dict, dict):
            _normalize_recursively(pred_dict)
        if isinstance(gt_dict, dict):
            _normalize_recursively(gt_dict)

        if pred_dict == gt_dict:
            return 1.0
            
        return 0.0
        
    except Exception as e:
        print(f"Error scoring grid_request: {e}")
        return None
