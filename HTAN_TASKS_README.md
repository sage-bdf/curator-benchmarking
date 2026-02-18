# HTAN v1.2.0 Benchmarking Tasks

## Overview

This implementation creates 13 benchmarking tasks from HTAN v1.2.0 synthetic data to evaluate LLM performance on clinical metadata error correction. Each task tests the model's ability to correct realistic data entry errors using a hybrid scoring approach.

## Implementation Summary

### Created Files

1. **scripts/prepare_htan_tasks.py** (main automation script)
   - Iterates through HTAN v1.2.0 synthetic datasets
   - Creates task directories with all required files
   - Fetches or copies JSON schemas from schemaUri
   - Generates custom scorers, prompt formatters, and prompt templates

2. **scripts/generate_report.py** (PDF report generator)
   - Creates comprehensive 18-page PDF reports
   - Includes executive summary, charts, detailed results, and recommendations
   - Requires: `fpdf2>=2.7.0`, `matplotlib>=3.7.0`

3. **tasks/htan_*/** (13 task directories)
   - `input_data.tsv` - Records with realistic errors
   - `ground_truth.tsv` - Corrected reference data
   - `schema.json` - JSON schema for validation
   - `default_prompt.txt` - Error correction instructions
   - `format_prompt.py` - Schema integration into prompts
   - `score.py` - Hybrid scorer (exact match + Jaccard similarity)

### Task List (13 total)

✅ Successfully Created:
- htan_biospecimen (25 samples, high complexity)
- htan_demographics (25 samples, low complexity)
- htan_diagnosis (30 samples, high complexity)
- htan_digital_pathology (20 samples, medium complexity)
- htan_exposure (20 samples, medium complexity)
- htan_family_history (19 samples, medium complexity)
- htan_follow_up (20 samples, medium complexity)
- htan_molecular_test (15 samples, medium complexity)
- htan_multiplex_microscopy_level2 (20 samples, high complexity)
- htan_multiplex_microscopy_level3 (20 samples, high complexity)

⚠️ Parsed with Issues (TSV format inconsistencies):
- htan_bulk_wes_level1 (20 samples, high complexity)
- htan_bulk_wes_level2 (20 samples, high complexity)
- htan_bulk_wes_level3 (25 samples, high complexity)

## Hybrid Scoring Methodology

The scoring system uses two different metrics based on field type:

### 1. Field-Level Accuracy (Structured Data)
- **Used for**: Enums, controlled vocabularies, numeric values, IDs with patterns
- **Metric**: Exact matching (1.0 if exact match, 0.0 otherwise)
- **Examples**:
  - `GENDER_IDENTITY`: "Male" vs "Male" → 1.0
  - `AGE_IN_DAYS_AT_SPECIMEN_COLLECTION`: 25 vs 30 → 0.0

### 2. Jaccard Similarity (Free-Text Data)
- **Used for**: Free-text fields (*_OTHER_SPECIFY, description fields)
- **Metric**: Word overlap similarity (|A ∩ B| / |A ∪ B|)
- **Example**:
  - Predicted: "family history"
  - Ground truth: "family history of cancer"
  - Words A = {family, history}, B = {family, history, of, cancer}
  - Score = 2/4 = 0.5

### Final Score
- Average of all field scores across the record
- Task score = mean across all records

## Usage

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate all HTAN tasks (if not already done)
python scripts/prepare_htan_tasks.py
```

### Running Benchmarks

```bash
# List all tasks
python -m src.cli list | grep htan

# Run single task
python -m src.cli run htan_demographics

# Run with specific model
python -m src.cli run htan_biospecimen \
    --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --temperature 0.0

# Run all working HTAN tasks
for task in htan_biospecimen htan_demographics htan_diagnosis htan_digital_pathology htan_exposure htan_family_history htan_follow_up htan_molecular_test htan_multiplex_microscopy_level2 htan_multiplex_microscopy_level3; do
    python -m src.cli run $task
done
```

### Generating PDF Reports

```bash
# Generate report from latest experiment
python scripts/generate_report.py

# Generate report for specific experiment
python scripts/generate_report.py --experiment-id abc123...

# Specify custom output path
python scripts/generate_report.py --output my_report.pdf
```

## Known Issues

### TSV Parsing Errors

Three tasks (bulk_wes_level1, bulk_wes_level2, bulk_wes_level3) have TSV files with inconsistent field counts:
- **Issue**: Line 6 has 22 fields instead of expected 21
- **Root cause**: Likely unescaped tab characters or line breaks in field values in the original synthetic data
- **Impact**: Tasks cannot be loaded by the curator-benchmarking framework
- **Workaround**:
  1. Manually fix TSV files by escaping special characters
  2. Or regenerate synthetic data with proper TSV escaping
  3. Or use a more robust parser that handles quoted fields

To fix, you could:
```python
import pandas as pd

# Read with different options
df = pd.read_csv('input_data.tsv', sep='\t', quoting=3, on_bad_lines='skip')
# or
df = pd.read_csv('input_data.tsv', sep='\t', quoting=0, escapechar='\\')
```

### Schema Injection for Scoring

The custom scorers need access to the schema to classify field types. Current implementation expects schema in `input_data['_schema']` parameter, but the framework doesn't inject this by default.

**Workaround options**:
1. Modify `src/task.py` to inject schema: `sample['_schema'] = self.schema`
2. Pre-compute field types and store in `field_types.json` per task
3. Load schema directly in scorer from task directory

## API Keys

The framework uses API keys from `.secret` file in the repo root:
- `ANTHROPIC_API_KEY` - For Claude models
- `AWS_BEARER_TOKEN_BEDROCK` - For AWS Bedrock models
- `OPENROUTER_API_KEY` - For OpenRouter models

## Report Contents

The generated PDF report includes:
1. **Executive Summary** - Overall performance metrics
2. **Task Performance Overview** - Tables and charts comparing all tasks
3. **Detailed Results** - One page per task with sample-level breakdowns
4. **Error Type Analysis** - Performance across error categories
5. **Scoring Methodology** - Explanation of hybrid approach
6. **Recommendations** - Insights and improvement suggestions

## Schema Sources

All schemas are fetched from:
```
https://raw.githubusercontent.com/ncihtan/htan2-data-model/main/JSON_Schemas/v1.2.0/
```

Schemas include:
- Field types (string, integer, number, boolean, array)
- Enums (controlled vocabularies)
- Patterns (regex for IDs and formats)
- Range constraints (minimum, maximum)
- Required fields
- Conditional dependencies

## Error Types Covered

The synthetic data includes realistic errors:
- ✓ Case sensitivity (e.g., "male" vs "Male")
- ✓ Invalid enum values
- ✓ Missing required fields
- ✓ Invalid ID formats
- ✓ Out-of-range numeric values
- ✓ Leading/trailing whitespace
- ✓ Wrong array separators
- ✓ Conditional validation failures
- ✓ Type mismatches (string vs number)
- ✓ Typos and misspellings
- ✓ Synonym usage (e.g., "Caucasian" vs "White")

## Next Steps

1. **Fix TSV parsing issues** in bulk_wes_level* tasks
2. **Implement schema injection** for scorers
3. **Run benchmarks** across multiple models
4. **Generate comparative reports** to analyze model performance
5. **Fine-tune prompts** based on error analysis
6. **Validate hybrid scoring** matches expected behavior

## Contact

For questions or issues, refer to the implementation plan at:
`/Users/jmoon/.claude/plans/virtual-juggling-walrus.md`
