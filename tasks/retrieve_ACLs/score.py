"""Custom scorer for regex_generation task."""
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
    """
    Score regex generation task by applying the generated regex to all input filenames
    and comparing matches to ground truth.
    
    Args:
        prediction: The model's prediction containing a regex pattern
        ground_truth: Dictionary with "matches" key containing expected matches as JSON array
        input_data: Dictionary with "filenames" key containing all input filenames as JSON array
        
    Returns:
        Score between 0.0 and 1.0 (fraction of correct matches), or None on error
    """
    try:
        # Extract JSON from prediction
        json_str = _extract_json(prediction)
        
        # Parse prediction to get regex pattern
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        # Extract regex pattern
        regex_pattern = pred_dict.get('regex', '')
        if not regex_pattern:
            return 0.0
        
        # Strip Python raw string prefix if present (r"...")
        regex_pattern = regex_pattern.strip()
        if regex_pattern.startswith('r"') and regex_pattern.endswith('"'):
            regex_pattern = regex_pattern[2:-1]
        elif regex_pattern.startswith("r'") and regex_pattern.endswith("'"):
            regex_pattern = regex_pattern[2:-1]
        elif regex_pattern.startswith('"') and regex_pattern.endswith('"'):
            regex_pattern = regex_pattern[1:-1]
        elif regex_pattern.startswith("'") and regex_pattern.endswith("'"):
            regex_pattern = regex_pattern[1:-1]
        
        # Get input filenames (now a list, not a single filename)
        if not input_data:
            return 0.0
        
        filenames_str = input_data.get('filenames', '[]')
        try:
            if isinstance(filenames_str, str):
                filenames = json.loads(filenames_str)
            else:
                filenames = filenames_str
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        if not filenames:
            return 0.0
        
        # Get expected matches from ground truth
        expected_matches_str = ground_truth.get('matches', '[]')
        try:
            if isinstance(expected_matches_str, str):
                expected_matches = json.loads(expected_matches_str)
            else:
                expected_matches = expected_matches_str
        except (json.JSONDecodeError, TypeError):
            return 0.0
        
        if len(filenames) != len(expected_matches):
            return 0.0
        
        # Apply regex to each filename and collect all matches
        all_matches = []
        try:
            for i, filename in enumerate(filenames):
                match_obj = re.search(regex_pattern, filename)
                expected = expected_matches[i] if i < len(expected_matches) else None
                
                if match_obj:
                    full_match = match_obj.group(0)
                    
                    # If the regex has capture groups, try to use them
                    if match_obj.groups():
                        groups = match_obj.groups()
                        
                        # Strategy 1: Check if expected match is a substring of full match
                        if expected and expected in full_match:
                            match = expected
                        # Strategy 2: Try group1-Tgroup2 pattern (for "NF0014-T1")
                        elif len(groups) >= 2 and expected and '-T' in expected:
                            match = f"{groups[0]}-T{groups[1]}"
                        # Strategy 3: Use first group only
                        elif len(groups) == 1:
                            match = str(groups[0])
                        # Strategy 4: Join all groups (fallback)
                        else:
                            match = ''.join(str(g) for g in groups if g)
                    else:
                        # No capture groups - use the full matched string
                        if expected and expected in full_match:
                            match = expected
                        else:
                            # Try removing common file extensions/suffixes
                            suffixes = ['.markdup.sorted.bam', '.fastq.gz', '_quant.sf', '_parquet']
                            match = full_match
                            for suffix in suffixes:
                                if match.endswith(suffix):
                                    match = match[:-len(suffix)]
                                    break
                            if expected and match != expected:
                                match = full_match
                else:
                    match = ''
                all_matches.append(match)
        except re.error as e:
            print(f"    Invalid regex pattern: {e}")
            return 0.0
        
        # Compare matches - calculate accuracy as fraction of correct matches
        if len(all_matches) == 0:
            return 0.0
        
        correct = sum(1 for i, match in enumerate(all_matches) 
                    if i < len(expected_matches) and match == expected_matches[i])
        
        return correct / len(expected_matches) if expected_matches else 0.0
    
    except Exception as e:
        print(f"Error scoring regex generation: {e}")
        return None

