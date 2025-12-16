"""Scoring function for retrieve_access_restrictions task."""
import json
import urllib.request
import urllib.error
import time
from typing import Dict, Any, Optional, Set, Tuple


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


def _extract_restriction_fields(restriction_info: Dict[str, Any]) -> Tuple[bool, str, Set[str]]:
    """
    Extract key fields from restriction information API response.

    Args:
        restriction_info: The API response from /restrictionInformation

    Returns:
        Tuple of (hasRestrictions, restrictionLevel, requirements_set)
    """
    has_restrictions = restriction_info.get("hasUnmetAccessRequirement", False)

    # Determine restriction level
    restriction_level = "Open Access"
    requirements = set()

    if has_restrictions:
        # Check restriction requirements to determine level
        restriction_requirements = restriction_info.get("restrictionDetails", [])

        for req in restriction_requirements:
            req_type = req.get("type", "")
            requirements.add(req_type)

            # Determine restriction level based on access requirement type
            if "ManagedACTAccessRequirement" in req_type:
                restriction_level = "Managed Access"
            elif "ACTAccessRequirement" in req_type or "TermsOfUseAccessRequirement" in req_type:
                if restriction_level == "Open Access":
                    restriction_level = "Controlled Access"

    return has_restrictions, restriction_level, requirements


def _calculate_score(
    pred_level: str,
    actual_level: str
) -> float:
    """
    Calculate score based on restriction level match.

    Args:
        pred_level: Predicted restriction level
        actual_level: Actual restriction level from API

    Returns:
        1.0 if prediction matches actual, 0.0 otherwise
    """
    return 1.0 if pred_level.lower() == actual_level.lower() else 0.0


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_sample: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Score the model's prediction by comparing with ground truth.

    Args:
        prediction: Model's output as a string (should be JSON)
        ground_truth: Ground truth dictionary with restrictionLevel field
        input_sample: Input sample with entityId

    Returns:
        Score between 0.0 and 1.0 based on ground truth accuracy, or None on error
    """
    try:
        # Parse prediction as JSON
        try:
            pred = json.loads(prediction)
        except json.JSONDecodeError:
            print("    Prediction is not valid JSON")
            return 0.0

        # Get entityId from input
        entity_id = input_sample.get('entityId', 'unknown') if input_sample else 'unknown'

        # Extract prediction fields
        pred_restriction_level = pred.get("restrictionLevel", "")

        # Get expected value from ground truth
        expected_level = ground_truth.get("restrictionLevel", "")

        # Calculate score based on restriction level match
        score_value = _calculate_score(pred_restriction_level, expected_level)

        print(f"    Ground Truth Comparison for {entity_id}:")
        print(f"      Predicted restrictionLevel: {pred_restriction_level}")
        print(f"      Expected restrictionLevel: {expected_level}")
        print(f"      Level Match: {pred_restriction_level.lower() == expected_level.lower()}")
        print(f"      Score: {score_value:.3f}")

        return score_value

    except Exception as e:
        print(f"Error scoring retrieve_access_restrictions: {e}")
        return None
