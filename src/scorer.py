"""Scoring and evaluation for experiment results."""
from typing import Dict, Any, Optional
import json
import re


class Scorer:
    """Handles scoring of predictions against ground truth."""
    
    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from text, handling markdown code blocks and other formatting.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Extracted JSON string, or None if no JSON found
        """
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        text = text.strip()
        
        # Try to find JSON object boundaries
        # Look for { ... } pattern
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        
        return text
    
    def score(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> Optional[float]:
        """
        Score a prediction against ground truth using strict matching.
        
        Args:
            prediction: The model's prediction (must be valid JSON, may be wrapped in markdown)
            ground_truth: The expected result as a dictionary
            
        Returns:
            Score between 0.0 and 1.0, or None if scoring is not possible
        """
        try:
            # Extract JSON from prediction (handles markdown code blocks)
            json_str = self._extract_json(prediction)
            
            # Try to parse prediction as JSON
            try:
                pred_dict = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                # If not JSON, return 0.0 - adherence to prompt format is part of the test
                return 0.0
            
            # If both are dictionaries, do strict structured comparison
            return self._structured_score(pred_dict, ground_truth)
        
        except Exception as e:
            print(f"Error scoring prediction: {e}")
            return None
    
    def _structured_score(
        self,
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
    

