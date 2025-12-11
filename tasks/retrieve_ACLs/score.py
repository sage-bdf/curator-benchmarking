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
        ground_truth: Ground truth dictionary with hasAccessRequirements, hasACL, dataUseSummary
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
    pred_has_access_reqs = pred.get("hasAccessRequirements")
    pred_acl_summary = pred.get("aclSummary", [])
    pred_data_use_summary = pred.get("dataUseSummary", "")

    # Extract ground truth fields
    gt_has_access_reqs_str = str(ground_truth.get("hasAccessRequirements", "")).lower()
    gt_has_access_reqs = gt_has_access_reqs_str == "true"
    gt_has_acl_str = str(ground_truth.get("hasACL", "")).lower()
    gt_has_acl = gt_has_acl_str == "true"
    gt_data_use_summary = ground_truth.get("dataUseSummary", "")

    # Scoring components
    score_components = []

    # 1. Check hasAccessRequirements (30% of score)
    if pred_has_access_reqs == gt_has_access_reqs:
        score_components.append(0.3)
    else:
        score_components.append(0.0)

    # 2. Check aclSummary structure and presence (30% of score)
    has_acl_summary = isinstance(pred_acl_summary, list) and len(pred_acl_summary) > 0
    if has_acl_summary == gt_has_acl:
        # Correct prediction about ACL presence
        score_components.append(0.2)

        # If ACL summary exists, verify structure
        if has_acl_summary:
            valid_structure = True
            for acl_entry in pred_acl_summary:
                if not isinstance(acl_entry, dict):
                    valid_structure = False
                    break
                # Check for expected fields
                if "requirementId" not in acl_entry and "aclId" not in acl_entry:
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

    # 3. Check dataUseSummary quality (40% of score)
    if pred_data_use_summary and gt_data_use_summary:
        # Check if key concepts from ground truth appear in prediction
        gt_lower = gt_data_use_summary.lower()
        pred_lower = pred_data_use_summary.lower()

        key_terms = []

        # Extract key terms from ground truth based on context
        if "open" in gt_lower or "no" in gt_lower and "restriction" in gt_lower:
            key_terms.append(("open", "no restriction", "freely"))
        if "managed access" in gt_lower:
            key_terms.append("managed")
        if "request" in gt_lower or "submit" in gt_lower:
            key_terms.append(("request", "submit"))
        if "review" in gt_lower or "approv" in gt_lower:
            key_terms.append(("review", "approv"))
        if "team" in gt_lower or "administrat" in gt_lower:
            key_terms.append(("team", "administrat", "manager"))
        if "researcher" in gt_lower or "user" in gt_lower:
            key_terms.append(("researcher", "user"))
        if "certif" in gt_lower:
            key_terms.append("certif")

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

            key_terms_score = (matched_terms / len(key_terms)) * 0.4
        else:
            # If no specific key terms, give partial credit if summary exists
            key_terms_score = 0.2

        score_components.append(key_terms_score)
    else:
        score_components.append(0.0)

    # Calculate total score
    total_score = sum(score_components)

    # Generate explanation
    details = []
    if score_components[0] == 0.3:
        details.append("✓ hasAccessRequirements correct")
    else:
        details.append(f"✗ hasAccessRequirements incorrect (expected: {gt_has_access_reqs}, got: {pred_has_access_reqs})")

    if score_components[1] == 0.2:
        details.append("✓ aclSummary presence correct")
    else:
        details.append(f"✗ aclSummary incorrect (expected hasACL: {gt_has_acl})")

    if len(score_components) > 2 and score_components[2] == 0.1:
        details.append("✓ aclSummary structure valid")

    if score_components[-1] > 0.3:
        details.append("✓ dataUseSummary quality good")
    elif score_components[-1] > 0.1:
        details.append("~ dataUseSummary quality partial")
    else:
        details.append("✗ dataUseSummary quality poor")

    return {
        "score": total_score,
        "explanation": " | ".join(details)
    }
