"""Custom prompt formatter for HTAN correction tasks."""
import json
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format prompt with schema information for HTAN data correction.

    Includes:
    - Base prompt template
    - Simplified schema showing enum values and field types
    - Input data as JSON
    """
    # Build simplified schema showing key validation rules
    schema_text = ""
    if schema and "properties" in schema:
        simplified_schema = {
            "type": "object",
            "properties": {}
        }

        properties = schema["properties"]
        for prop_name, prop_def in properties.items():
            field_info = {
                "type": prop_def.get("type", "string")
            }

            # Add description (truncated)
            if "description" in prop_def:
                desc = prop_def["description"]
                field_info["description"] = desc[:100] + "..." if len(desc) > 100 else desc

            # Include enum values (limit to 20 if very large)
            if "enum" in prop_def:
                enum_values = prop_def["enum"]
                if len(enum_values) > 20:
                    field_info["enum_preview"] = enum_values[:20]
                    field_info["enum_count"] = len(enum_values)
                    field_info["enum_note"] = f"Controlled vocabulary with {len(enum_values)} values. First 20 shown."
                else:
                    field_info["enum"] = enum_values

            # Include pattern for ID validation
            if "pattern" in prop_def:
                field_info["pattern"] = prop_def["pattern"]

            # Include range constraints
            if "minimum" in prop_def:
                field_info["minimum"] = prop_def["minimum"]
            if "maximum" in prop_def:
                field_info["maximum"] = prop_def["maximum"]

            # Include array item constraints
            if "items" in prop_def and prop_def.get("type") == "array":
                items = prop_def["items"]
                if "enum" in items:
                    enum_values = items["enum"]
                    if len(enum_values) > 20:
                        field_info["items_enum_preview"] = enum_values[:20]
                        field_info["items_enum_count"] = len(enum_values)
                    else:
                        field_info["items_enum"] = enum_values

            simplified_schema["properties"][prop_name] = field_info

        # Add required fields info
        if "required" in schema:
            simplified_schema["required"] = schema["required"]

        schema_text = f"\n\nTarget Schema:\n{json.dumps(simplified_schema, indent=2)}"

    # Format input data
    sample_text = f"\n\nInput Data (with errors to correct):\n{json.dumps(sample, indent=2)}"

    return f"{prompt_template}{schema_text}{sample_text}"
