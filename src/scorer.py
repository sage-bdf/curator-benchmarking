"""Scoring and evaluation for experiment results."""
from typing import Dict, Any, Optional, List
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
        ground_truth: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Score a prediction against ground truth using strict matching.
        
        Args:
            prediction: The model's prediction (must be valid JSON, may be wrapped in markdown)
            ground_truth: The expected result as a dictionary
            input_data: Optional input data (needed for regex generation tasks)
            
        Returns:
            Score between 0.0 and 1.0, or None if scoring is not possible
        """
        try:
            # Check if this is a regex generation task (ground truth has "matches" key)
            if 'matches' in ground_truth and input_data and 'filename' in input_data:
                return self._score_regex_generation(prediction, ground_truth, input_data)
            
            # Check if this is a column enumeration task (ground truth has "expected_columns" key)
            if 'expected_columns' in ground_truth:
                return self._score_column_enumeration(prediction, ground_truth)
            
            # Check if this is a column type identification task (ground truth has "expected_type" key)
            if 'expected_type' in ground_truth:
                return self._score_column_type_identification(prediction, ground_truth)
            
            # Check if this is a validation error counting task (ground truth has "expected_count" key)
            if 'expected_count' in ground_truth:
                return self._score_validation_error_counting(prediction, ground_truth)
            
            # Check if this is a row validation explanation task (ground truth has "expected_explanation_keywords" key)
            if 'expected_explanation_keywords' in ground_truth:
                return self._score_row_validation_explanation(prediction, ground_truth)
            
            # Check if this is a column value retrieval task (ground truth has "expected_value" key)
            if 'expected_value' in ground_truth:
                return self._score_column_value_retrieval(prediction, ground_truth)
            
            # Check if this is a row value retrieval task (ground truth has "expected_values" key)
            if 'expected_values' in ground_truth:
                return self._score_row_value_retrieval(prediction, ground_truth)
            
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
    
    def _score_regex_generation(
        self,
        prediction: str,
        ground_truth: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> float:
        """
        Score regex generation task by applying the generated regex to the input
        and comparing matches to ground truth.
        
        Args:
            prediction: The model's prediction containing a regex pattern
            ground_truth: Dictionary with "matches" key containing expected matches as JSON array
            input_data: Dictionary with "filename" key containing the input filename
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Extract JSON from prediction
            json_str = self._extract_json(prediction)
            
            # Parse prediction to get regex pattern
            try:
                pred_dict = json.loads(json_str)
            except (json.JSONDecodeError, TypeError):
                return 0.0
            
            # Extract regex pattern
            regex_pattern = pred_dict.get('regex', '')
            if not regex_pattern:
                return 0.0
            
            # Get input filename
            filename = input_data.get('filename', '')
            if not filename:
                return 0.0
            
            # Apply regex to filename
            try:
                matches = re.findall(regex_pattern, filename)
                # Convert to list of strings (re.findall returns tuples for groups)
                if matches:
                    # If regex has groups, matches will be tuples; flatten to strings
                    if isinstance(matches[0], tuple):
                        matches = [''.join(m) for m in matches]
                    else:
                        matches = [str(m) for m in matches]
                else:
                    matches = []
            except re.error as e:
                # Invalid regex pattern
                print(f"    Invalid regex pattern: {e}")
                return 0.0
            
            # Get expected matches from ground truth
            expected_matches_str = ground_truth.get('matches', '[]')
            try:
                if isinstance(expected_matches_str, str):
                    expected_matches = json.loads(expected_matches_str)
                else:
                    expected_matches = expected_matches_str
            except (json.JSONDecodeError, TypeError):
                expected_matches = []
            
            # Normalize to lists of strings for comparison
            matches = [str(m) for m in matches]
            expected_matches = [str(m) for m in expected_matches]
            
            # Compare matches (order matters for regex extraction)
            if matches == expected_matches:
                return 1.0
            else:
                # Partial credit: check if all expected matches are present
                if set(matches) == set(expected_matches) and len(matches) == len(expected_matches):
                    # Same matches but different order - give partial credit
                    return 0.5
                # Check how many expected matches are found
                found_count = len(set(matches) & set(expected_matches))
                if len(expected_matches) > 0:
                    return found_count / len(expected_matches)
                return 0.0
        
        except Exception as e:
            print(f"Error scoring regex generation: {e}")
            return 0.0
    
    def _score_column_enumeration(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score column enumeration task by comparing column lists."""
        try:
            json_str = self._extract_json(prediction)
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
            return 0.0
    
    def _score_column_type_identification(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score column type identification by comparing type strings."""
        try:
            json_str = self._extract_json(prediction)
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
            return 0.0
    
    def _score_validation_error_counting(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score validation error counting task by comparing counts."""
        try:
            json_str = self._extract_json(prediction)
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
            return 0.0
    
    def _score_row_validation_explanation(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score row validation explanation by checking for expected keywords."""
        try:
            json_str = self._extract_json(prediction)
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
            return 0.0
    
    def _score_column_value_retrieval(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score column value retrieval by comparing values."""
        try:
            json_str = self._extract_json(prediction)
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
            return 0.0
    
    def _score_row_value_retrieval(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Score row value retrieval by comparing JSON objects with column-value pairs."""
        try:
            json_str = self._extract_json(prediction)
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
            return self._structured_score(pred_dict, expected_values)
        except Exception as e:
            print(f"Error scoring row value retrieval: {e}")
            return 0.0
    

