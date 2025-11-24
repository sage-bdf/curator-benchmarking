# Grid Request Task

This task evaluates the model's ability to generate correct grid queries and updates based on user requests.

## System Instructions

The system instructions for this task are pre-compiled to include:
1.  Base instructions (persona, operational context, examples)
2.  Data Schema (fetched from a remote URL)
3.  OpenAPI Specification (fetched from a remote URL)

### Updating System Instructions

If you need to update the base instructions or the source URLs for the schema/OpenAPI spec, modify `compile_system_instructions.py`.

To regenerate the `system_instructions.txt` file (which is used by the benchmark), run:

```bash
python tasks/grid_request/compile_system_instructions.py
```

This script will:
1.  Use the embedded base instructions.
2.  Fetch the latest Schema and OpenAPI spec.
3.  Combine them and overwrite `tasks/grid_request/system_instructions.txt`.

## Running with OpenRouter

To run this task using an OpenRouter model, set your API key and specify the model ID:

```bash
export OPENROUTER_API_KEY=your_key_here
python -m src.cli run grid_request --model x-ai/grok-4.1-fast:free
```
## Development Notes

### Dataset generation

Dataset was drafted with https://github.com/nf-osi/agent-at-work/tree/main/recipes/benchmarking and validated and improved as needed with human review. The parameters are stored in `metadata.json`. If the dataset is updated, please rerun the `compile_system_instructions.py` script so that the correct schemas are injected from references in `metadata.json`.

### Scoring

Currently, `score.py` tries to not overly penalize for differences that are not semantically important unless they are explicitly marked as important in `ground_truth.tsv`. 

For example, though syntactically different, these are considered equivalent and accepted by the API:
```json
{"set": [...]}
{"set": [...], "filters": []}
```

The `score.py` scorer applies these normalizations before comparison:

1. **Limit parameter**: Removed when `limit_required=false` (default in `ground_truth.tsv`) (in the future, we may be stricter)
2. **Empty filters arrays**: `"filters": []` is removed and treated as equivalent to omitting the filters field entirely
