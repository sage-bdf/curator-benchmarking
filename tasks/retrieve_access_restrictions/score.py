"""Scoring function for retrieve_access_restrictions task."""
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
        ground_truth: Ground truth dictionary with hasRestrictions, restrictionType, explanation
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
    pred_has_restrictions = pred.get("hasRestrictions")
    pred_restriction_type = pred.get("restrictionType")
    pred_explanation = pred.get("explanation", "")

    # Extract ground truth fields
    gt_has_restrictions_str = str(ground_truth.get("hasRestrictions", "")).lower()
    gt_has_restrictions = gt_has_restrictions_str == "true"
    gt_restriction_type = ground_truth.get("restrictionType")
    if gt_restriction_type and str(gt_restriction_type).lower() == "null":
        gt_restriction_type = None
    gt_explanation = ground_truth.get("explanation", "")

    # Scoring components
    score_components = []

    # 1. Check hasRestrictions (40% of score)
    if pred_has_restrictions == gt_has_restrictions:
        score_components.append(0.4)
    else:
        score_components.append(0.0)

    # 2. Check restrictionType (30% of score)
    if gt_restriction_type is None:
        # If no restriction type expected, check if prediction also has null/None
        if pred_restriction_type is None or str(pred_restriction_type).lower() in ["null", "none", ""]:
            score_components.append(0.3)
        else:
            score_components.append(0.0)
    else:
        # Compare restriction types (case-insensitive, allow partial matches)
        if pred_restriction_type and str(pred_restriction_type).lower() in str(gt_restriction_type).lower():
            score_components.append(0.3)
        elif pred_restriction_type and str(gt_restriction_type).lower() in str(pred_restriction_type).lower():
            score_components.append(0.3)
        else:
            score_components.append(0.0)

    # 3. Check explanation quality (30% of score)
    if pred_explanation and gt_explanation:
        # Check if key concepts from ground truth appear in prediction
        gt_lower = gt_explanation.lower()
        pred_lower = pred_explanation.lower()

        key_terms_score = 0
        key_terms = []

        # Extract key terms from ground truth
        if "managed access" in gt_lower:
            key_terms.append("managed")
        if "controlled access" in gt_lower:
            key_terms.append("controlled")
        if "certification" in gt_lower:
            key_terms.append("certif")  # Matches certification/certified
        if "request" in gt_lower or "approval" in gt_lower:
            key_terms.append("request")
        if "publicly accessible" in gt_lower or "open access" in gt_lower:
            key_terms.append("public")
        if "no restrictions" in gt_lower or "no access restrictions" in gt_lower:
            key_terms.append("no restriction")

        # Count how many key terms are present
        if key_terms:
            matched_terms = sum(1 for term in key_terms if term in pred_lower)
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
    if score_components[0] == 0.4:
        details.append("✓ hasRestrictions correct")
    else:
        details.append(f"✗ hasRestrictions incorrect (expected: {gt_has_restrictions}, got: {pred_has_restrictions})")

    if score_components[1] == 0.3:
        details.append("✓ restrictionType correct")
    else:
        details.append(f"✗ restrictionType incorrect (expected: {gt_restriction_type}, got: {pred_restriction_type})")

    if score_components[2] > 0.2:
        details.append("✓ explanation quality good")
    elif score_components[2] > 0:
        details.append("~ explanation quality partial")
    else:
        details.append("✗ explanation quality poor")

    return {
        "score": total_score,
        "explanation": " | ".join(details)
    }
