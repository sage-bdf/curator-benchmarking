# LLM Metadata Curation Benchmarking Framework

A framework for benchmarking different LLMs on metadata curation tasks using AWS Bedrock.

## Overview

This framework allows you to:
- Test different LLM models on metadata curation tasks
- Permute key variables: model endpoint, system instructions, temperature, and thinking mode
- Run experiments and automatically score results against ground truth
- Track experiment results over time via GitHub Issues and a web dashboard

---

## Contributing a Task

Tasks define specific metadata curation challenges that LLMs are evaluated on. Each task consists of input data, expected outputs (ground truth), and a prompt template.

### Task Directory Structure

Create a new directory under `tasks/` with the following structure:

```
tasks/
└── your_task_name/
    ├── input_data.csv          # Required: Input data (CSV or TSV)
    ├── ground_truth.tsv        # Optional: Expected results for scoring
    ├── default_prompt.txt      # Required: Default prompt template
    ├── format_prompt.py        # Optional: Custom prompt formatter
    ├── score.py                # Optional: Custom scoring function
    └── schema.json             # Optional: JSON schema for validation
```

### Required Files

#### 1. Input Data (`input_data.csv` or `input_data.tsv`)

The input data file contains the samples that will be processed by the LLM. Each row represents one sample.

- **Format**: CSV or TSV
- **Encoding**: UTF-8
- **Headers**: First row should contain column names
- **Example**:
  ```csv
  id,field1,field2
  1,value1,value2
  2,value3,value4
  ```

#### 2. Default Prompt (`default_prompt.txt`)

The prompt template that will be used to format each input sample. The framework will automatically append the input data as JSON to your prompt.

- **Format**: Plain text file
- **Example**:
  ```
  Please correct the following metadata entry according to the task requirements.
  Return the corrected version as JSON with the same structure as the input.
  ```

The framework will format it as:
```
Please correct the following metadata entry according to the task requirements.
Return the corrected version as JSON with the same structure as the input.

Input data:
{
  "id": "1",
  "field1": "value1",
  "field2": "value2"
}
```

### Optional Files

#### 3. Ground Truth (`ground_truth.tsv` or `ground_truth.csv`)

Expected results for each input sample. Used for automatic scoring.

- **Format**: CSV or TSV
- **Must match**: Same number of rows as input data
- **Columns**: Should contain the expected output fields

#### 4. Custom Prompt Formatter (`format_prompt.py`)

If you need custom prompt formatting logic beyond the default behavior, create a `format_prompt.py` file with a `format_prompt()` function:

```python
def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the prompt for this task.
    
    Args:
        prompt_template: The base prompt from default_prompt.txt
        sample: Input sample data (dict)
        ground_truth: Ground truth sample (optional)
        schema: Optional JSON schema (optional)
        
    Returns:
        Formatted prompt string
    """
    # Your custom formatting logic here
    sample_str = json.dumps(sample, indent=2)
    return f"{prompt_template}\n\nInput data:\n{sample_str}"
```

See `tasks/column_enumeration/format_prompt.py` for an example.

#### 5. Custom Scorer (`score.py`)

If you need custom scoring logic (beyond the default field-level or text similarity scoring), create a `score.py` file with a `score()` function:

```python
def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Score a prediction against ground truth.
    
    Args:
        prediction: The LLM's output (string, may contain JSON)
        ground_truth: Expected result (dict)
        input_data: Original input data (optional)
        
    Returns:
        Score between 0.0 and 1.0, or None if scoring failed
    """
    # Your custom scoring logic here
    # Return 1.0 for perfect match, 0.0 for no match
    pass
```

See `tasks/column_enumeration/score.py` for an example.

#### 6. JSON Schema (`schema.json`)

If your task requires structured JSON output, you can provide a JSON schema for validation. This is useful for tasks that require specific fields or controlled vocabularies.

- **Format**: JSON Schema (draft-07)
- **Example**: See `tasks/correction_of_typos/schema.json`

### Example Task

See `tasks/example_task/` for a minimal example, or `tasks/column_enumeration/` for a complete example with custom formatting and scoring.

### Testing Your Task

After creating your task, test it locally:

```bash
# List tasks to verify yours is detected
python -m src.cli list

# Run a test experiment
python -m src.cli run your_task_name
```

---

## Submitting an Experiment

Experiments test LLM models on one or more tasks with specific configurations (model, system instructions, temperature, etc.). The recommended way to submit experiments is through GitHub Issues.

### Via GitHub Issues (Recommended)

1. **Create a new issue** using the "Experiment Submission" template
   - Go to the Issues tab in the repository
   - Click "New Issue"
   - Select "Experiment Submission" template

2. **Fill out the form**:
   - **Model Endpoint**: Select from dropdown or choose "Other" to specify a custom model
     - Default: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
     - Other options include Claude Haiku, Claude Sonnet variants, Amazon Nova, DeepSeek, and OpenAI models
   - **Custom Model Endpoint**: If you selected "Other", enter your model endpoint here
   - **System Instructions**: Optional custom system instructions
     - Reference a file: `file:path/to/instructions.txt`
     - Or paste instructions directly
     - Leave empty to use default
   - **Temperature**: Sampling temperature (0.0-1.0, leave empty for default 0.0)
   - **Thinking Mode**: Enable thinking/reasoning mode for supported models (e.g., Claude Sonnet 4.5)
   - **Experiment Description**: Optional description of what you're testing

3. **Submit the issue**
   - The `experiment` label is automatically applied
   - The GitHub Action will automatically run the experiment across **all available tasks**

4. **Monitor progress**
   - Watch progress in the Actions tab
   - Results will be posted as a comment when complete
   - The issue will be automatically closed on success

### File References

You can reference files in the repository for system instructions:

```
file:prompts/my_custom_prompt.txt
file:instructions/metadata_curation_v2.txt
```

Paths are relative to the repository root.

### Example Issue

```
### Model Endpoint
global.anthropic.claude-3-5-sonnet-20241022-v2:0

### System Instructions
file:instructions/custom_instructions.txt

### Temperature
0.1

### Thinking Mode
false

### Experiment Description
Testing Claude 3.5 Sonnet on metadata correction with custom instructions and lower temperature.
```

### Important Notes

- **Experiments run on ALL tasks**: When you submit an experiment, it will automatically run across all available tasks in the `tasks/` directory
- **GitHub Actions setup**: For GitHub Actions to work, you need:
  - Repository secrets: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
  - GitHub Actions enabled with permissions to write comments and close issues
- **Updating task list**: When new tasks are added, update the issue template:
  ```bash
  python scripts/update_issue_template.py
  ```

### Viewing Results

Results are automatically posted as a comment on the issue. You can also:

- View the **GitHub Pages dashboard** (if enabled): `https://<username>.github.io/curator-benchmarking/`
- Check the `docs/results/` directory for JSON result files
- Review the experiment log: `docs/results/experiments_log.jsonl`

See the [Experiment Results](#experiment-results) section below for more details.

---

## Local Setup

### Prerequisites

- Python 3.9 or higher
- AWS account with Bedrock access
- AWS credentials configured

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd curator-benchmarking
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or use the setup script:
   ```bash
   ./setup.sh
   ```

4. **Configure AWS credentials**:
   - Set environment variables:
     ```bash
     export AWS_ACCESS_KEY_ID=your_key
     export AWS_SECRET_ACCESS_KEY=your_secret
     ```
   - Or use AWS CLI: `aws configure`
   - Ensure your AWS account has access to the Bedrock models you want to use

5. **Review configuration** (optional):
   - Edit `config/defaults.yaml` to change:
     - Default model endpoint
     - AWS region (default: `us-east-1`)
     - Default system instructions
     - Experiment parameters (temperature, max_tokens, retries)

---

## Command-Line Usage

### List Available Tasks

```bash
python -m src.cli list
```

### Run a Single Experiment

Run an experiment with default settings:
```bash
python -m src.cli run <task_name>
```

Run with custom parameters:
```bash
python -m src.cli run <task_name> \
    --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --system-instructions custom_instructions.txt \
    --prompt custom_prompt.txt \
    --temperature 0.1
```

### Run a Suite of Experiments

Run multiple experiments with different parameter combinations:
```bash
python -m src.cli suite <task_name> \
    --models \
        global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        us.anthropic.claude-3-5-sonnet-20240620-v1:0 \
    --system-instructions \
        instructions_v1.txt \
        instructions_v2.txt \
    --prompts \
        prompt_v1.txt \
        prompt_v2.txt
```

This will run all combinations of models × instructions × prompts (2 × 2 × 2 = 8 experiments).

### Process an Issue Manually

If you have saved an issue body to a file, you can process it manually:
```bash
python -m src.issue_processor <issue_body_file.txt>
```

---

## Experiment Results

### Result Files

Results are saved in the `docs/results/` directory:
- **Individual experiment results**: `<experiment_id>_<task_name>.json`
- **Experiment log**: `experiments_log.jsonl` (append-only log of all experiments)

Each experiment result includes:
- Experiment metadata (ID, timestamp, parameters)
- Individual sample results (input, prediction, ground truth, score)
- Aggregate metrics (success rate, average score, etc.)

### GitHub Pages Dashboard

A web-based dashboard is available to explore experiment results:

1. **Update results for GitHub Pages**:
   ```bash
   ./scripts/update_gh_pages.sh
   # or
   python scripts/update_gh_pages.py
   ```

2. **Enable GitHub Pages** in repository settings:
   - Settings → Pages
   - Source: Deploy from a branch
   - Branch: `main` (or your default branch)
   - Folder: `/docs`

3. **Access the dashboard** at: `https://<username>.github.io/curator-benchmarking/`

The dashboard provides:
- Visual overview of all experiments
- Filtering by task or model
- Sorting by date, score, or task name
- Key metrics and experimental parameters
- Responsive, modern interface

See `docs/README.md` for more details.

---

## Configuration

Default configuration is in `config/defaults.yaml`. Key settings:

- **AWS region**: Default region for Bedrock API calls
- **Default model**: Model endpoint used when not specified
- **Default system instructions**: Base instructions for all experiments
- **Experiment parameters**: Temperature, max_tokens, retries, etc.

You can override these settings:
- Via command-line arguments
- Via GitHub issue fields
- By editing `config/defaults.yaml`

---

## Scoring

The framework automatically scores predictions against ground truth when available:

- **Default scoring**:
  - Structured data: Field-level accuracy (exact match)
  - Text data: Word overlap similarity (Jaccard similarity)
- **Custom scoring**: If a task includes `score.py`, the custom scorer is used
- **Scores range**: 0.0 to 1.0, where 1.0 is a perfect match

---

## Repository Structure

```
.
├── config/
│   └── defaults.yaml          # Default configuration
├── tasks/                      # Task directories
│   └── <task_name>/
│       ├── input_data.csv      # Input data (CSV or TSV)
│       ├── ground_truth.tsv    # Expected results (optional)
│       ├── default_prompt.txt  # Default prompt template
│       ├── format_prompt.py    # Custom formatter (optional)
│       ├── score.py            # Custom scorer (optional)
│       └── schema.json         # JSON schema (optional)
├── docs/                       # GitHub Pages dashboard
│   ├── index.html              # Main dashboard interface
│   └── results/                # Results (auto-generated)
├── src/                        # Framework source code
│   ├── config.py
│   ├── bedrock_client.py
│   ├── task.py
│   ├── experiment.py
│   ├── scorer.py
│   ├── cli.py
│   ├── issue_processor.py
│   └── issue_processor_github.py
├── scripts/                    # Utility scripts
│   ├── update_gh_pages.py
│   ├── update_issue_template.py
│   └── file_experiment_issues.py
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── experiment.yml       # GitHub issue template
│   └── workflows/
│       └── run_experiment.yml   # GitHub Actions workflow
└── requirements.txt
```

---

## Additional Resources

- **Quick Start Guide**: See `QUICKSTART.md` for a condensed setup and usage guide
- **Experiment Workflow**: See `EXPERIMENT_WORKFLOW.md` for detailed GitHub issue workflow documentation
- **Example Task**: See `tasks/example_task/README.md` for task structure examples

---

## Troubleshooting

### GitHub Actions Issues

- **Issue not triggering workflow**: Ensure the issue has the `experiment` label and follows the template format
- **Experiment fails**: Check Actions logs, verify AWS credentials in repository secrets, ensure task directory exists
- **Results not posted**: Verify workflow has permission to write comments (Settings → Actions → General → Workflow permissions)

### Local Execution Issues

- **AWS credentials**: Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set
- **Model access**: Verify your AWS account has access to the Bedrock models you're using
- **Task not found**: Check that the task directory exists and contains required files (`input_data.csv` or `input_data.tsv`, `default_prompt.txt`)
