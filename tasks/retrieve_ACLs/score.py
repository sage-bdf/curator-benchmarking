"""Scoring function for retrieve_ACLs task."""
import json
from typing import Dict, Any, Optional


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_sample: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Score the model's prediction against ground truth.

    Args:
        prediction: Model's output as a string (should be JSON)
        ground_truth: Ground truth dictionary with aclId, hasResourceAccess, explanation
        input_sample: Input sample (not used for scoring)

    Returns:
        Dictionary with score (0.0-1.0) and explanation
    """
    try:
        # Parse prediction as JSON
        pred = json.loads(prediction)
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "explanation": "Prediction is not valid JSON"
        }

    # Extract fields from prediction
    pred_acl_id = pred.get("aclId")
    pred_resource_access = pred.get("resourceAccess", [])
    pred_explanation = pred.get("explanation", "")

    # Extract ground truth fields
    gt_acl_id = ground_truth.get("aclId")
    gt_has_resource_access_str = str(ground_truth.get("hasResourceAccess", "")).lower()
    gt_has_resource_access = gt_has_resource_access_str == "true"
    gt_explanation = ground_truth.get("explanation", "")

    # Scoring components
    score_components = []

    # 1. Check aclId is present (30% of score)
    if pred_acl_id:
        # Check if it matches expected (if available) or at least exists
        if gt_acl_id and str(pred_acl_id) == str(gt_acl_id):
            score_components.append(0.3)
        elif gt_acl_id:
            # Wrong ID
            score_components.append(0.0)
        else:
            # Ground truth doesn't specify ID, give credit for having one
            score_components.append(0.3)
    else:
        score_components.append(0.0)

    # 2. Check resourceAccess (40% of score)
    has_resource_access = isinstance(pred_resource_access, list) and len(pred_resource_access) > 0
    if has_resource_access == gt_has_resource_access:
        # Correct prediction about resource access
        score_components.append(0.3)

        # If resource access exists, verify structure
        if has_resource_access:
            valid_structure = True
            for ra in pred_resource_access:
                if not isinstance(ra, dict):
                    valid_structure = False
                    break
                if "principalId" not in ra or "accessType" not in ra:
                    valid_structure = False
                    break
                if not isinstance(ra.get("accessType"), list):
                    valid_structure = False
                    break

            if valid_structure:
                score_components.append(0.1)  # Bonus for correct structure
            else:
                score_components.append(0.0)
        else:
            score_components.append(0.0)
    else:
        score_components.append(0.0)
        score_components.append(0.0)

    # 3. Check explanation quality (30% of score)
    if pred_explanation and gt_explanation:
        # Check if key concepts from ground truth appear in prediction
        gt_lower = gt_explanation.lower()
        pred_lower = pred_explanation.lower()

        key_terms = []

        # Extract key terms from ground truth
        if "acl" in gt_lower:
            key_terms.append("acl")
        if "permission" in gt_lower or "manage" in gt_lower:
            key_terms.append("permission")
        if "principal" in gt_lower or "user" in gt_lower or "team" in gt_lower:
            key_terms.append(("principal", "user", "team"))  # Match any of these
        if "access request" in gt_lower or "review" in gt_lower:
            key_terms.append("request")
        if "administrat" in gt_lower:
            key_terms.append("administrat")

        # Count how many key terms are present
        if key_terms:
            matched_terms = 0
            for term in key_terms:
                if isinstance(term, tuple):
                    # Match any of the terms in the tuple
                    if any(t in pred_lower for t in term):
                        matched_terms += 1
                else:
                    if term in pred_lower:
                        matched_terms += 1

            key_terms_score = (matched_terms / len(key_terms)) * 0.3
        else:
            # If no specific key terms, give partial credit if explanation exists
            key_terms_score = 0.15

        score_components.append(key_terms_score)
    else:
        score_components.append(0.0)

    # Calculate total score
    total_score = sum(score_components)

    # Generate explanation
    details = []
    if score_components[0] == 0.3:
        details.append("✓ aclId present")
    else:
        details.append(f"✗ aclId missing or incorrect")

    if score_components[1] == 0.3:
        details.append("✓ resourceAccess correct")
    else:
        details.append(f"✗ resourceAccess incorrect (expected hasResourceAccess: {gt_has_resource_access})")

    if len(score_components) > 2 and score_components[2] == 0.1:
        details.append("✓ resourceAccess structure valid")

    if score_components[-1] > 0.2:
        details.append("✓ explanation quality good")
    elif score_components[-1] > 0:
        details.append("~ explanation quality partial")
    else:
        details.append("✗ explanation quality poor")

    return {
        "score": total_score,
        "explanation": " | ".join(details)
    }
