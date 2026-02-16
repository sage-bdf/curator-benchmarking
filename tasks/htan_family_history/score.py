"""Custom scorer for HTAN data correction task using hybrid metric."""
import json
import re
from typing import Dict, Any, Optional, Set


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two text strings.

    Jaccard similarity = |A ∩ B| / |A ∪ B|
    where A and B are sets of words
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    # Normalize: lowercase, split on whitespace
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


def classify_field_type(prop_name: str, prop_schema: Dict[str, Any]) -> str:
    """
    Classify field as 'structured' or 'text'.

    Structured: enums, patterns, numeric types, arrays with enums
    Text: strings without strict constraints, *_OTHER_SPECIFY
    """
    # Check for enum
    if "enum" in prop_schema:
        return "structured"

    # Check for pattern (but not _OTHER_SPECIFY which are free-text)
    if "pattern" in prop_schema and not prop_name.endswith("_OTHER_SPECIFY"):
        return "structured"

    # Check for numeric/boolean types
    prop_type = prop_schema.get("type", "string")
    if prop_type in ["integer", "number", "boolean"]:
        return "structured"

    # Check for arrays with enum items
    if prop_type == "array":
        items = prop_schema.get("items", {})
        if "enum" in items:
            return "structured"

    # Default to text
    return "text"


def load_field_types(schema: Dict[str, Any]) -> Dict[str, str]:
    """Build field type mapping from schema properties."""
    field_types = {}
    properties = schema.get("properties", {})

    for prop_name, prop_schema in properties.items():
        field_types[prop_name] = classify_field_type(prop_name, prop_schema)

    return field_types


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()

    # Try to find JSON object boundaries
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
    Score HTAN data correction using hybrid metric.

    - Structured fields (enums, IDs, numbers): Exact match (field-level accuracy)
    - Free-text fields: Jaccard similarity (word overlap)

    Returns weighted average across all fields.
    """
    try:
        # Extract JSON from prediction
        json_str = _extract_json(prediction)
        if not json_str:
            return 0.0

        # Parse prediction
        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        # Load schema from input_data if available
        schema = {}
        if input_data and "_schema" in input_data:
            schema = input_data["_schema"]

        # Classify field types
        field_types = load_field_types(schema) if schema else {}

        # Calculate scores per field
        all_keys = set(pred_dict.keys()) | set(ground_truth.keys())
        if not all_keys:
            return 1.0

        field_scores = []
        for key in all_keys:
            pred_val = pred_dict.get(key)
            truth_val = ground_truth.get(key)

            # Determine field type (default to structured if schema not available)
            field_type = field_types.get(key, "structured")

            if field_type == "structured":
                # Exact match for structured fields
                if pred_val == truth_val:
                    field_scores.append(1.0)
                else:
                    field_scores.append(0.0)
            else:
                # Jaccard similarity for text fields
                similarity = jaccard_similarity(
                    str(pred_val) if pred_val is not None else '',
                    str(truth_val) if truth_val is not None else ''
                )
                field_scores.append(similarity)

        return sum(field_scores) / len(field_scores)

    except Exception as e:
        print(f"Error scoring prediction: {e}")
        return None
