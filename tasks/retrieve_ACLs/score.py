"""Scoring function for retrieve_ACLs task."""
import json
import re
import urllib.request
import urllib.error
import time
from typing import Dict, Any, Optional


def _fetch_restriction_info(entity_id: str, timeout: int = 30, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Fetch restriction information for an entity from Synapse REST API with retry logic.

    Args:
        entity_id: The Synapse entity ID (e.g., syn26462036)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with restriction information or None if fetch failed
    """
    api_url = "https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation"

    request_body = {
        "restrictableObjectType": "ENTITY",
        "objectId": entity_id
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                api_url,
                data=json.dumps(request_body).encode('utf-8'),
                headers=headers,
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result

        except urllib.error.HTTPError as e:
            print(f"    HTTP error fetching restriction info (attempt {attempt + 1}/{max_retries}): {e.code} {e.reason}")
            if attempt < max_retries - 1 and e.code >= 500:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
        except urllib.error.URLError as e:
            print(f"    Network error fetching restriction info (attempt {attempt + 1}/{max_retries}): {e.reason}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
        except Exception as e:
            print(f"    Error fetching restriction info (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

    return None


def _calculate_score(
    pred_has_reqs: bool,
    actual_has_reqs: bool
) -> float:
    """
    Calculate score based on binary classification of hasAccessRequirements.

    Args:
        pred_has_reqs: Predicted hasAccessRequirements
        actual_has_reqs: Actual hasAccessRequirements from API

    Returns:
        1.0 if prediction matches actual, 0.0 otherwise
    """
    return 1.0 if pred_has_reqs == actual_has_reqs else 0.0


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain extra content.

    Args:
        text: Text that may contain JSON along with explanatory text

    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    # First try direct parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text using regex
    # Look for patterns like {...} that span multiple lines
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    return None


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_sample: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Score the model's prediction by comparing with ground truth.

    Args:
        prediction: Model's output as a string (should be JSON)
        ground_truth: Ground truth dictionary with hasAccessRequirements field
        input_sample: Input sample with entityId

    Returns:
        Score between 0.0 and 1.0 based on ground truth accuracy, or None on error
    """
    try:
        # Parse prediction as JSON (with extraction for noisy responses)
        pred = _extract_json(prediction)
        if pred is None:
            print("    Prediction is not valid JSON")
            return 0.0

        # Get entityId from input
        entity_id = input_sample.get('entityId', 'unknown') if input_sample else 'unknown'

        # Extract prediction fields
        pred_has_access_reqs = pred.get("hasAccessRequirements", False)

        # Get expected value from ground truth
        expected_has_reqs = ground_truth.get("hasAccessRequirements", False)

        # Handle string "true"/"false" from TSV
        if isinstance(expected_has_reqs, str):
            expected_has_reqs = expected_has_reqs.lower() == "true"

        # Calculate score based on binary classification
        score_value = _calculate_score(pred_has_access_reqs, expected_has_reqs)

        print(f"    Ground Truth Comparison for {entity_id}:")
        print(f"      Predicted hasAccessRequirements: {pred_has_access_reqs}")
        print(f"      Expected hasAccessRequirements: {expected_has_reqs}")
        print(f"      Classification Correct: {pred_has_access_reqs == expected_has_reqs}")
        print(f"      Score: {score_value:.3f}")

        return score_value

    except Exception as e:
        print(f"Error scoring retrieve_ACLs: {e}")
        return None
