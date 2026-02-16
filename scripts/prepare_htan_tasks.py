"""
Prepare HTAN v1.2.0 benchmarking tasks for curator-benchmarking framework.

This script:
1. Iterates through HTAN v1.2.0 synthetic datasets
2. Creates task directories in curator-benchmarking/tasks/
3. Copies input_data.tsv and ground_truth.tsv
4. Fetches or copies JSON schemas
5. Generates default_prompt.txt, format_prompt.py, and score.py for each task
"""

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Any

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SYNTHETIC_DATA_ROOT = REPO_ROOT / "benchmarking/sim-input/synthetic-data/htan2/v1.2.0"
TASKS_ROOT = REPO_ROOT / "curator-benchmarking/tasks"
SCRIPTS_DIR = Path(__file__).parent

# Schema cache to avoid repeated downloads
_schema_cache: Dict[str, Dict[str, Any]] = {}


def fetch_schema(uri: str) -> Dict[str, Any]:
    """Fetch JSON schema from URI with caching."""
    if uri in _schema_cache:
        return _schema_cache[uri]

    try:
        with urllib.request.urlopen(uri) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status} fetching schema from {uri}")
            schema = json.load(response)
    except Exception as exc:
        raise Exception(f"Failed to fetch schema from {uri}: {exc}")

    _schema_cache[uri] = schema
    return schema


def classify_field_types(schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Classify each field as 'structured' or 'text'.

    Structured: enums, patterns (non-_OTHER_SPECIFY), integer/number/boolean types
    Text: strings without strict constraints, *_OTHER_SPECIFY fields
    """
    field_types = {}
    properties = schema.get("properties", {})

    for prop_name, prop_schema in properties.items():
        # Check for enum
        if "enum" in prop_schema:
            field_types[prop_name] = "structured"
            continue

        # Check for pattern (but not _OTHER_SPECIFY which are free-text)
        if "pattern" in prop_schema and not prop_name.endswith("_OTHER_SPECIFY"):
            field_types[prop_name] = "structured"
            continue

        # Check for numeric/boolean types
        prop_type = prop_schema.get("type", "string")
        if prop_type in ["integer", "number", "boolean"]:
            field_types[prop_name] = "structured"
            continue

        # Check for arrays with enum items
        if prop_type == "array":
            items = prop_schema.get("items", {})
            if "enum" in items:
                field_types[prop_name] = "structured"
                continue

        # Default to text
        field_types[prop_name] = "text"

    return field_types


def generate_score_py(schema: Dict[str, Any]) -> str:
    """Generate score.py content with hybrid scorer."""
    return '''"""Custom scorer for HTAN data correction task using hybrid metric."""
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
    text = re.sub(r'```json\\s*\\n?', '', text)
    text = re.sub(r'```\\s*\\n?', '', text)
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
'''


def generate_format_prompt_py() -> str:
    """Generate format_prompt.py content."""
    return '''"""Custom prompt formatter for HTAN correction tasks."""
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

        schema_text = f"\\n\\nTarget Schema:\\n{json.dumps(simplified_schema, indent=2)}"

    # Format input data
    sample_text = f"\\n\\nInput Data (with errors to correct):\\n{json.dumps(sample, indent=2)}"

    return f"{prompt_template}{schema_text}{sample_text}"
'''


def generate_default_prompt_txt(schema_type: str, metadata: Dict[str, Any]) -> str:
    """Generate default_prompt.txt content."""
    complexity = metadata.get("complexity", "medium")
    error_notes = metadata.get("coverageNotes", "Various error types present")

    return f'''You are a metadata curation assistant specializing in HTAN biomedical data quality.

Your task: Correct errors in {schema_type.replace("_", " ").title()} metadata records.

Dataset Complexity: {complexity}
Common error types in the input:
{error_notes[:500]}

General error types to watch for:
- Enum values with wrong case (e.g., "male" should be "Male")
- Invalid enum values not in controlled vocabulary
- Missing required fields
- Invalid ID formats (HTAN IDs must match specific patterns)
- Out-of-range numeric values (negative ages, percentages > 100)
- Leading/trailing whitespace
- Wrong array separators (semicolons instead of commas, or vice versa)
- Conditional validation failures (e.g., "Other" selected without specification)

INSTRUCTIONS:
1. Carefully review the input record and identify ALL errors
2. Correct each error by:
   - Matching exact enum values from the Target Schema (case-sensitive)
   - Fixing ID formats to match HTAN patterns
   - Ensuring numeric values are within valid ranges
   - Removing extra whitespace
   - Filling required fields when context allows
   - Using proper separators for array values (check schema)
3. For free-text fields (*_OTHER_SPECIFY), preserve meaning while fixing formatting
4. Return ONLY the corrected record as valid JSON
5. Preserve all original fields even if they don't need correction

CRITICAL RULES:
- Use EXACT enum values from the schema (case-sensitive matching required)
- Do not add fields that weren't in the input
- Do not remove fields unless they're completely invalid
- Ensure correct data types (integer vs number vs string vs array)
- For arrays, use the proper separator (usually commas between items)

Output Format:
```json
{{
  "FIELD_1": "corrected_value",
  "FIELD_2": 123,
  "ARRAY_FIELD": ["value1", "value2"],
  ...
}}
```

Return ONLY the JSON. No explanation needed.'''


def create_task(dataset_dir: Path, task_name: str):
    """Create a single HTAN benchmarking task."""
    print(f"\nProcessing {task_name}...")

    # Create task directory
    task_dir = TASKS_ROOT / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created task directory: {task_dir}")

    # Load metadata
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: metadata.json not found in {dataset_dir}")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text())

    # Copy input_data.tsv and ground_truth.tsv
    for filename in ["input_data.tsv", "ground_truth.tsv"]:
        src = dataset_dir / filename
        dst = task_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
        else:
            print(f"  WARNING: {filename} not found in {dataset_dir}")

    # Fetch or copy schema
    schema_uri = metadata.get("schemaUri")
    if not schema_uri:
        print(f"  WARNING: schemaUri not found in metadata")
        schema = {}
    else:
        # Check if schema.json exists locally
        local_schema_path = dataset_dir / "schema.json"
        if local_schema_path.exists():
            print(f"  Found local schema.json, copying...")
            shutil.copy2(local_schema_path, task_dir / "schema.json")
            schema = json.loads(local_schema_path.read_text())
        else:
            print(f"  Fetching schema from {schema_uri}...")
            schema = fetch_schema(schema_uri)
            # Save to task directory
            (task_dir / "schema.json").write_text(json.dumps(schema, indent=2))
            print(f"  Saved schema.json")

    # Generate score.py
    score_content = generate_score_py(schema)
    (task_dir / "score.py").write_text(score_content)
    print(f"  Generated score.py")

    # Generate format_prompt.py
    format_prompt_content = generate_format_prompt_py()
    (task_dir / "format_prompt.py").write_text(format_prompt_content)
    print(f"  Generated format_prompt.py")

    # Generate default_prompt.txt
    schema_type = task_name.replace("htan_", "")
    prompt_content = generate_default_prompt_txt(schema_type, metadata)
    (task_dir / "default_prompt.txt").write_text(prompt_content)
    print(f"  Generated default_prompt.txt")

    print(f"  ✓ Task {task_name} created successfully")


def main():
    """Main function to prepare all HTAN tasks."""
    print("=" * 70)
    print("HTAN v1.2.0 Task Preparation")
    print("=" * 70)

    if not SYNTHETIC_DATA_ROOT.exists():
        print(f"ERROR: Synthetic data root not found: {SYNTHETIC_DATA_ROOT}")
        return

    # Get all dataset directories
    dataset_dirs = [d for d in SYNTHETIC_DATA_ROOT.iterdir() if d.is_dir()]
    dataset_dirs.sort()

    print(f"\nFound {len(dataset_dirs)} datasets:")
    for d in dataset_dirs:
        print(f"  - {d.name}")

    # Create tasks
    print("\nCreating tasks...")
    for dataset_dir in dataset_dirs:
        schema_type = dataset_dir.name
        task_name = f"htan_{schema_type}"
        try:
            create_task(dataset_dir, task_name)
        except Exception as e:
            print(f"  ERROR creating task {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Task preparation complete!")
    print("=" * 70)
    print(f"\nTasks created in: {TASKS_ROOT}")
    print("\nNext steps:")
    print("  1. cd curator-benchmarking")
    print("  2. python -m src.cli list | grep htan")
    print("  3. python -m src.cli run htan_demographics")


if __name__ == "__main__":
    main()
