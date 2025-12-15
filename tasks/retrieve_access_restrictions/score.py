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


def _calculate_metrics(
    pred_has_restr: bool,
    actual_has_restr: bool,
    pred_level: str,
    actual_level: str
) -> Tuple[int, int, int, int]:
    """
    Calculate TP, FP, FN, TN for restriction detection.

    Args:
        pred_has_restr: Predicted hasRestrictions
        actual_has_restr: Actual hasRestrictions from API
        pred_level: Predicted restriction level
        actual_level: Actual restriction level from API

    Returns:
        Tuple of (true_positive, false_positive, false_negative, true_negative)
    """
    # For hasRestrictions boolean
    if pred_has_restr and actual_has_restr:
        tp_restr = 1
        fp_restr = 0
        fn_restr = 0
        tn_restr = 0
    elif not pred_has_restr and not actual_has_restr:
        tp_restr = 0
        fp_restr = 0
        fn_restr = 0
        tn_restr = 1
    elif pred_has_restr and not actual_has_restr:
        tp_restr = 0
        fp_restr = 1
        fn_restr = 0
        tn_restr = 0
    else:  # not pred_has_restr and actual_has_restr
        tp_restr = 0
        fp_restr = 0
        fn_restr = 1
        tn_restr = 0

    # For restriction level (if restrictions exist)
    tp_level = 0
    fp_level = 0
    fn_level = 0

    if actual_has_restr:
        if pred_level.lower() == actual_level.lower():
            tp_level = 1
        else:
            fp_level = 1
            fn_level = 1

    return tp_restr + tp_level, fp_restr + fp_level, fn_restr + fn_level, tn_restr


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_sample: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Score the model's prediction by comparing with actual API results.

    Args:
        prediction: Model's output as a string (should be JSON)
        ground_truth: Ground truth dictionary (not used - we fetch actual data)
        input_sample: Input sample with entityId

    Returns:
        Score between 0.0 and 1.0 based on API result accuracy, or None on error
    """
    try:
        # Parse prediction as JSON
        try:
            pred = json.loads(prediction)
        except json.JSONDecodeError:
            print("    Prediction is not valid JSON")
            return 0.0

        # Get entityId from input
        if not input_sample or 'entityId' not in input_sample:
            print("    No entityId in input_sample")
            return None

        entity_id = input_sample['entityId']

        # Extract prediction fields
        pred_has_restrictions = pred.get("hasRestrictions", False)
        pred_restriction_level = pred.get("restrictionLevel", "")
        pred_data_use_summary = pred.get("dataUseSummary", "")

        # Fetch actual restriction information from API
        restriction_info = _fetch_restriction_info(entity_id)

        if restriction_info is None:
            print(f"    Could not fetch restriction info for {entity_id} - scoring as 0.0")
            return 0.0

        # Extract actual values from API response
        actual_has_restr, actual_level, actual_requirements = _extract_restriction_fields(restriction_info)

        # Calculate metrics
        tp, fp, fn, tn = _calculate_metrics(
            pred_has_restrictions,
            actual_has_restr,
            pred_restriction_level,
            actual_level
        )

        print(f"    API Results Comparison for {entity_id}:")
        print(f"      Predicted hasRestrictions: {pred_has_restrictions}")
        print(f"      Actual hasRestrictions: {actual_has_restr}")
        print(f"      Predicted restrictionLevel: {pred_restriction_level}")
        print(f"      Actual restrictionLevel: {actual_level}")
        print(f"      True Positives (TP): {tp}")
        print(f"      False Positives (FP): {fp}")
        print(f"      False Negatives (FN): {fn}")
        print(f"      True Negatives (TN): {tn}")

        # Calculate accuracy-based score
        total = tp + fp + fn + tn
        if total > 0:
            accuracy = (tp + tn) / total
            print(f"      Accuracy: {accuracy:.3f}")
            return accuracy
        else:
            return 0.0

    except Exception as e:
        print(f"Error scoring retrieve_access_restrictions: {e}")
        return None
