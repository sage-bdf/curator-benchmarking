"""Custom scorer for search_query_generation task."""
import json
import re
from typing import Dict, Any, Optional
from urllib.parse import unquote, urlparse, parse_qs


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
    Score search_query_generation task by validating the generated query wrapper URL.

    Args:
        prediction: The model's prediction containing platform and queryWrapper
        ground_truth: Dictionary with "queryWrapper" key containing expected URL
        input_data: Dictionary with "queryPhrase" and "platform" fields

    Returns:
        Score between 0.0 and 1.0, or None on error
    """
    try:
        # Extract JSON from prediction
        json_str = _extract_json(prediction)

        # Parse prediction
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        # Extract queryWrapper from prediction
        pred_query_wrapper = pred_dict.get('queryWrapper', '')
        if not pred_query_wrapper:
            return 0.0

        # Get input platform
        platform = input_data.get('platform', '') if input_data else ''

        # Get expected query wrapper from ground truth
        expected_query_wrapper = ground_truth.get('queryWrapper', '')
        if not expected_query_wrapper:
            return 0.0

        score_value = 0.0

        # Component 1: Check correct base URL (40%)
        if platform == 'Bridge2AI':
            if pred_query_wrapper.startswith('https://b2ai.standards.synapse.org/Explore/?QueryWrapper0='):
                score_value += 0.4
        elif platform == 'NF Tools':
            if pred_query_wrapper.startswith('https://nf.synapse.org/Explore/Tools?QueryWrapper0='):
                score_value += 0.4

        # Component 2: Check if QueryWrapper0 parameter has valid JSON structure (40%)
        try:
            parsed_url = urlparse(pred_query_wrapper)
            query_params = parse_qs(parsed_url.query)

            if 'QueryWrapper0' in query_params:
                query_wrapper_json = query_params['QueryWrapper0'][0]
                # Try to parse the JSON
                wrapper_dict = json.loads(query_wrapper_json)

                # Check if it has a 'query' field
                if 'query' in wrapper_dict and wrapper_dict['query']:
                    score_value += 0.4
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            # Invalid JSON structure, no points for this component
            pass

        # Component 3: Check query similarity with ground truth (20%)
        try:
            # Extract queries from both URLs
            pred_parsed = urlparse(pred_query_wrapper)
            pred_params = parse_qs(pred_parsed.query)
            pred_wrapper_json = pred_params.get('QueryWrapper0', [''])[0]
            pred_wrapper_dict = json.loads(pred_wrapper_json)
            pred_query = pred_wrapper_dict.get('query', '').lower().strip()

            gt_parsed = urlparse(expected_query_wrapper)
            gt_params = parse_qs(gt_parsed.query)
            gt_wrapper_json = gt_params.get('QueryWrapper0', [''])[0]
            gt_wrapper_dict = json.loads(gt_wrapper_json)
            gt_query = gt_wrapper_dict.get('query', '').lower().strip()

            # Simple word overlap similarity
            if pred_query and gt_query:
                pred_words = set(pred_query.split())
                gt_words = set(gt_query.split())
                if pred_words and gt_words:
                    overlap = len(pred_words & gt_words)
                    union = len(pred_words | gt_words)
                    similarity = overlap / union if union > 0 else 0
                    score_value += 0.2 * similarity
        except (json.JSONDecodeError, ValueError, KeyError, IndexError):
            # Can't compare queries, no points for this component
            pass

        return score_value

    except Exception as e:
        print(f"Error scoring search query generation: {e}")
        return None

