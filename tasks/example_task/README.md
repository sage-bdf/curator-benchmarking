# Example Task

This is an example task structure. Each task should have:

1. **Input data** - CSV or TSV file with the data to process
2. **Ground truth** (optional) - CSV or TSV file with expected results for scoring
3. **Default prompt** - Text file with the prompt template

## File Naming

The framework will automatically detect:
- Input files: `input*.csv`, `input*.tsv`, or any CSV/TSV that's not ground truth
- Ground truth: Files containing "ground" and "truth" in the name (case-insensitive)
- Default prompt: `default_prompt.txt`

## Prompt Formatting

The prompt should be a template that will be combined with input data. The framework will automatically append the input data as JSON to your prompt.

Example prompt:
```
Please correct the following metadata entry. Return the corrected version as JSON with the same structure.
```

The framework will format it as:
```
Please correct the following metadata entry. Return the corrected version as JSON with the same structure.

Input data:
{
  "field1": "value1",
  "field2": "value2"
}
```

