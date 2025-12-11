"""Scoring function for retrieve_ACLs task."""
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Set, List, Tuple


def _fetch_restriction_info(entity_id: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch restriction information for an entity from Synapse REST API.

    Args:
        entity_id: The Synapse entity ID (e.g., syn26462036)
        timeout: Request timeout in seconds

    Returns:
        Dictionary with restriction information or None if fetch failed
    """
    try:
        api_url = "https://repo-prod.prod.sagebase.org/repo/v1/restrictionInformation"

        request_body = {
            "restrictableObjectType": "ENTITY",
            "objectId": entity_id
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(request_body).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result

    except Exception as e:
        print(f"    Error fetching restriction info: {e}")
        return None


def _fetch_acl(requirement_id: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch ACL for an access requirement from Synapse REST API.

    Args:
        requirement_id: The access requirement ID
        timeout: Request timeout in seconds

    Returns:
        Dictionary with ACL information or None if fetch failed
    """
    try:
        api_url = f"https://repo-prod.prod.sagebase.org/repo/v1/accessRequirement/{requirement_id}/acl"

        headers = {
            'Accept': 'application/json'
        }

        req = urllib.request.Request(api_url, headers=headers)

        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result

    except urllib.error.HTTPError as e:
        if e.code == 404:
            # No ACL found for this requirement
            return None
        print(f"    Error fetching ACL for requirement {requirement_id}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"    Error fetching ACL for requirement {requirement_id}: {e}")
        return None


def _extract_acl_principals(restriction_info: Dict[str, Any]) -> Set[str]:
    """
    Extract all principal IDs that have ACL access from restriction information.

    Args:
        restriction_info: The API response from /restrictionInformation

    Returns:
        Set of principal IDs (user/team IDs) that appear in ACLs
    """
    principals = set()

    # Get access requirement IDs from restriction details
    restriction_details = restriction_info.get("restrictionDetails", [])

    for detail in restriction_details:
        requirement_id = detail.get("accessRequirementId")
        if requirement_id:
            # Fetch ACL for this requirement
            acl = _fetch_acl(str(requirement_id))
            if acl and "resourceAccess" in acl:
                for access_entry in acl["resourceAccess"]:
                    principal_id = access_entry.get("principalId")
                    if principal_id:
                        principals.add(str(principal_id))

    return principals


def _calculate_metrics(
    pred_has_reqs: bool,
    actual_has_reqs: bool,
    pred_principals: Set[str],
    actual_principals: Set[str]
) -> Tuple[int, int, int, int]:
    """
    Calculate TP, FP, FN, TN for ACL detection.

    Args:
        pred_has_reqs: Predicted hasAccessRequirements
        actual_has_reqs: Actual hasAccessRequirements from API
        pred_principals: Set of principal IDs from prediction
        actual_principals: Set of principal IDs from actual ACLs

    Returns:
        Tuple of (true_positive, false_positive, false_negative, true_negative)
    """
    # Metrics for hasAccessRequirements boolean
    if pred_has_reqs and actual_has_reqs:
        tp_reqs = 1
        fp_reqs = 0
        fn_reqs = 0
        tn_reqs = 0
    elif not pred_has_reqs and not actual_has_reqs:
        tp_reqs = 0
        fp_reqs = 0
        fn_reqs = 0
        tn_reqs = 1
    elif pred_has_reqs and not actual_has_reqs:
        tp_reqs = 0
        fp_reqs = 1
        fn_reqs = 0
        tn_reqs = 0
    else:  # not pred_has_reqs and actual_has_reqs
        tp_reqs = 0
        fp_reqs = 0
        fn_reqs = 1
        tn_reqs = 0

    # Metrics for principal overlap (if ACLs exist)
    tp_principals = len(pred_principals & actual_principals)
    fp_principals = len(pred_principals - actual_principals)
    fn_principals = len(actual_principals - pred_principals)

    return tp_reqs + tp_principals, fp_reqs + fp_principals, fn_reqs + fn_principals, tn_reqs


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
        pred_has_access_reqs = pred.get("hasAccessRequirements", False)
        pred_acl_summary = pred.get("aclSummary", [])

        # Extract principal IDs from predicted ACL summary
        pred_principals = set()
        if isinstance(pred_acl_summary, list):
            for acl_entry in pred_acl_summary:
                if isinstance(acl_entry, dict):
                    # Look for principal IDs in various possible fields
                    for key in ["principalId", "userId", "teamId", "id"]:
                        if key in acl_entry:
                            pred_principals.add(str(acl_entry[key]))

        # Fetch actual restriction information from API
        restriction_info = _fetch_restriction_info(entity_id)

        if restriction_info is None:
            print(f"    Could not fetch restriction info for {entity_id}")
            return None

        # Determine if there are actual access requirements
        actual_has_reqs = restriction_info.get("hasUnmetAccessRequirement", False)

        # Extract actual principal IDs from ACLs
        actual_principals = set()
        if actual_has_reqs:
            actual_principals = _extract_acl_principals(restriction_info)

        # Calculate metrics
        tp, fp, fn, tn = _calculate_metrics(
            pred_has_access_reqs,
            actual_has_reqs,
            pred_principals,
            actual_principals
        )

        print(f"    API Results Comparison for {entity_id}:")
        print(f"      Predicted hasAccessRequirements: {pred_has_access_reqs}")
        print(f"      Actual hasAccessRequirements: {actual_has_reqs}")
        print(f"      Predicted ACL principals: {len(pred_principals)} principal(s)")
        print(f"      Actual ACL principals: {len(actual_principals)} principal(s)")
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
            # If no ACLs exist and prediction correctly identified no ACLs
            if not actual_has_reqs and not pred_has_access_reqs:
                print(f"      Accuracy: 1.000 (correctly identified no requirements)")
                return 1.0
            return 0.0

    except Exception as e:
        print(f"Error scoring retrieve_ACLs: {e}")
        return None
