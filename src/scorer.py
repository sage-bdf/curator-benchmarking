"""Scoring and evaluation for experiment results."""
from typing import Dict, Any, Optional
import json


class Scorer:
    """Handles scoring of predictions against ground truth."""
    
    def score(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> Optional[float]:
        """
        Score a prediction against ground truth.
        
        Args:
            prediction: The model's prediction (may be JSON string or text)
            ground_truth: The expected result as a dictionary
            
        Returns:
            Score between 0.0 and 1.0, or None if scoring is not possible
        """
        try:
            # Try to parse prediction as JSON
            try:
                pred_dict = json.loads(prediction)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, treat as text and do simple comparison
                return self._text_similarity_score(prediction, str(ground_truth))
            
            # If both are dictionaries, do structured comparison
            return self._structured_score(pred_dict, ground_truth)
        
        except Exception as e:
            print(f"Error scoring prediction: {e}")
            return None
    
    def _structured_score(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score structured data (dictionaries)."""
        if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
            return 0.0
        
        # Calculate field-level accuracy
        all_keys = set(prediction.keys()) | set(ground_truth.keys())
        if not all_keys:
            return 1.0  # Both empty, perfect match
        
        matches = 0
        for key in all_keys:
            pred_val = prediction.get(key)
            truth_val = ground_truth.get(key)
            
            if pred_val == truth_val:
                matches += 1
            elif isinstance(pred_val, str) and isinstance(truth_val, str):
                # Fuzzy match for strings (case-insensitive, whitespace-normalized)
                if pred_val.strip().lower() == truth_val.strip().lower():
                    matches += 1
        
        return matches / len(all_keys)
    
    def _text_similarity_score(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """Score text similarity using simple string matching."""
        pred_norm = prediction.strip().lower()
        truth_norm = ground_truth.strip().lower()
        
        if pred_norm == truth_norm:
            return 1.0
        
        # Simple word overlap
        pred_words = set(pred_norm.split())
        truth_words = set(truth_norm.split())
        
        if not truth_words:
            return 1.0 if not pred_words else 0.0
        
        intersection = pred_words & truth_words
        union = pred_words | truth_words
        
        return len(intersection) / len(union) if union else 0.0

