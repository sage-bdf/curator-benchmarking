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
        ground_truth: Ground truth dictionary with hasRestrictions, restrictionLevel, dataUseSummary
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
    pred_restriction_level = pred.get("restrictionLevel", "")
    pred_requirements = pred.get("requirements", [])
    pred_data_use_summary = pred.get("dataUseSummary", "")

    # Extract ground truth fields
    gt_has_restrictions_str = str(ground_truth.get("hasRestrictions", "")).lower()
    gt_has_restrictions = gt_has_restrictions_str == "true"
    gt_restriction_level = ground_truth.get("restrictionLevel", "")
    gt_data_use_summary = ground_truth.get("dataUseSummary", "")

    # Scoring components
    score_components = []

    # 1. Check hasRestrictions (30% of score)
    if pred_has_restrictions == gt_has_restrictions:
        score_components.append(0.3)
    else:
        score_components.append(0.0)

    # 2. Check restrictionLevel (30% of score)
    if pred_restriction_level and gt_restriction_level:
        # Case-insensitive comparison, allow partial matches
        pred_level_lower = str(pred_restriction_level).lower()
        gt_level_lower = str(gt_restriction_level).lower()

        if pred_level_lower == gt_level_lower:
            score_components.append(0.3)
        elif pred_level_lower in gt_level_lower or gt_level_lower in pred_level_lower:
            score_components.append(0.2)  # Partial credit for partial match
        else:
            score_components.append(0.0)
    elif not pred_restriction_level and not gt_restriction_level:
        score_components.append(0.3)
    else:
        score_components.append(0.0)

    # 3. Check dataUseSummary quality (40% of score)
    if pred_data_use_summary and gt_data_use_summary:
        # Check if key concepts from ground truth appear in prediction
        gt_lower = gt_data_use_summary.lower()
        pred_lower = pred_data_use_summary.lower()

        key_terms = []

        # Extract key terms from ground truth
        if "managed access" in gt_lower:
            key_terms.append("managed")
        if "controlled access" in gt_lower:
            key_terms.append("controlled")
        if "open access" in gt_lower or "openly" in gt_lower:
            key_terms.append(("open", "freely"))
        if "certification" in gt_lower or "certify" in gt_lower:
            key_terms.append("certif")
        if "request" in gt_lower or "submit" in gt_lower:
            key_terms.append(("request", "submit"))
        if "approv" in gt_lower:
            key_terms.append("approv")
        if "researcher" in gt_lower:
            key_terms.append("researcher")
        if "download" in gt_lower:
            key_terms.append("download")
        if "no restriction" in gt_lower or "without" in gt_lower:
            key_terms.append(("no restriction", "without"))
        if "terms" in gt_lower:
            key_terms.append("terms")

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
        details.append("✓ hasRestrictions correct")
    else:
        details.append(f"✗ hasRestrictions incorrect (expected: {gt_has_restrictions}, got: {pred_has_restrictions})")

    if score_components[1] == 0.3:
        details.append("✓ restrictionLevel correct")
    elif score_components[1] == 0.2:
        details.append("~ restrictionLevel partially correct")
    else:
        details.append(f"✗ restrictionLevel incorrect (expected: {gt_restriction_level}, got: {pred_restriction_level})")

    if score_components[2] > 0.3:
        details.append("✓ dataUseSummary quality good")
    elif score_components[2] > 0.1:
        details.append("~ dataUseSummary quality partial")
    else:
        details.append("✗ dataUseSummary quality poor")

    return {
        "score": total_score,
        "explanation": " | ".join(details)
    }
