# LLM Metadata Curation Benchmarking Framework

A framework for benchmarking different LLMs on metadata curation tasks using AWS Bedrock.

## Overview

This framework allows you to:
- Test different LLM models on metadata curation tasks
- Permute three key variables: model endpoint, system instructions, and prompts
- Run experiments and automatically score results against ground truth
- Track experiment results over time

## Structure

```
.
├── config/
│   └── defaults.yaml          # Default configuration
├── tasks/                      # Task directories
│   └── <task_name>/
│       ├── input_data.csv      # Input data for the task
│       ├── ground_truth.csv    # Expected results
│       └── default_prompt.txt  # Default prompt for the task
├── results/                    # Experiment results (auto-generated)
│   ├── <experiment_id>_results.json
│   └── experiments_log.jsonl
├── src/                        # Framework source code
│   ├── config.py
│   ├── bedrock_client.py
│   ├── task.py
│   ├── experiment.py
│   ├── scorer.py
│   ├── cli.py
│   ├── issue_processor.py
│   └── issue_processor_github.py
├── docs/                        # GitHub Pages dashboard
│   ├── index.html              # Main dashboard interface
│   └── results/                # Results copied for web access
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── experiment.yml       # GitHub issue template for experiments
│   └── workflows/
│       └── run_experiment.yml   # GitHub Actions workflow
└── requirements.txt
```

## Setup

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure AWS credentials:
   - Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables
   - Or configure using `aws configure`

3. Review and adjust `config/defaults.yaml` if needed:
   - Default model: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
   - Region: `us-east-1`
   - Default system instructions

## Task Structure

Each task should be organized in its own directory under `tasks/`:

```
tasks/
└── my_task/
    ├── input_data.csv          # Required: Input data (CSV or TSV)
    ├── ground_truth.csv        # Optional: Expected results for scoring
    ├── default_prompt.txt      # Required: Default prompt template
    └── task_config.yaml        # Optional: Task-specific configuration
```

The input data should be in CSV or TSV format. The framework will automatically detect and load it.

## Usage

### Submitting Experiments via GitHub Issues (Recommended)

The recommended way to submit experiments is through GitHub issues:

1. **Create a new issue** using the "Experiment Submission" template
2. **Fill out the form** with:
   - Task name
   - Model endpoint (optional, uses default if not specified)
   - System instructions (optional, can reference a file or paste directly)
   - Prompt (optional, can reference a file or paste directly)
   - Experiment description
3. **Label the issue** with the `experiment` label (automatically applied by template)
4. **The GitHub Action will automatically**:
   - Run the experiment when the issue is opened or edited
   - Post results as a comment on the issue
   - Close the issue when the experiment completes successfully

**Note:** For GitHub Actions to work, you need to:
- Set up repository secrets: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
- Ensure the GitHub Actions workflow has permission to write comments and close issues
- Update the issue template with available tasks: `python scripts/update_issue_template.py`

### Command-Line Usage

#### List Available Tasks

```bash
python -m src.cli list
```

#### Run a Single Experiment

Run an experiment with default settings:
```bash
python -m src.cli run <task_name>
```

Run with custom parameters:
```bash
python -m src.cli run <task_name> \
    --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --system-instructions custom_instructions.txt \
    --prompt custom_prompt.txt
```

#### Run a Suite of Experiments

Run multiple experiments with different parameter combinations:
```bash
python -m src.cli suite <task_name> \
    --models model1 model2 model3 \
    --system-instructions instructions1.txt instructions2.txt \
    --prompts prompt1.txt prompt2.txt
```

This will run all combinations of models × instructions × prompts.

#### Process an Issue Manually

If you have saved an issue body to a file, you can process it manually:
```bash
python -m src.issue_processor <issue_body_file.txt>
```

## Experiment Results

Results are saved in the `results/` directory:
- Individual experiment results: `<experiment_id>_results.json`
- Experiment log: `experiments_log.jsonl` (append-only log of all experiments)

Each experiment result includes:
- Experiment metadata (ID, timestamp, parameters)
- Individual sample results
- Aggregate metrics (success rate, average score, etc.)

### GitHub Pages Dashboard

A web-based dashboard is available to explore experiment results:

1. **Update results for GitHub Pages:**
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

## Configuration

Default configuration is in `config/defaults.yaml`. Key settings:
- AWS region and default model
- Default system instructions
- Experiment parameters (temperature, max_tokens, retries)

## Scoring

The framework automatically scores predictions against ground truth when available:
- Structured data: Field-level accuracy
- Text data: Word overlap similarity

Scores range from 0.0 to 1.0, where 1.0 is a perfect match.

