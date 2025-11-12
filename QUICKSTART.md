# Quick Start Guide

## Setup (One Time)

1. **Install dependencies:**
   ```bash
   ./setup.sh
   # Or manually:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure AWS credentials:**
   ```bash
   export AWS_ACCESS_KEY_ID=your_key
   export AWS_SECRET_ACCESS_KEY=your_secret
   ```
   Or use `aws configure` if you have AWS CLI installed.

## Running Experiments

### 1. List Available Tasks
```bash
python -m src.cli list
```

### 2. Run a Single Experiment (Default Settings)
```bash
python -m src.cli run example_task
```

### 3. Run with Custom Parameters
```bash
python -m src.cli run example_task \
    --model global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --system-instructions my_instructions.txt \
    --prompt my_prompt.txt
```

### 4. Run a Suite of Experiments
Test multiple combinations:
```bash
python -m src.cli suite example_task \
    --models \
        global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
        global.anthropic.claude-3-5-sonnet-20241022-v2:0 \
    --system-instructions \
        instructions_v1.txt \
        instructions_v2.txt \
    --prompts \
        prompt_v1.txt \
        prompt_v2.txt
```

This will run 2 × 2 × 2 = 8 experiments.

## Creating a New Task

1. Create a task directory:
   ```bash
   mkdir -p tasks/my_new_task
   ```

2. Add required files:
   - `input_data.csv` or `input_data.tsv` - Your input data
   - `default_prompt.txt` - The prompt template
   - `ground_truth.csv` (optional) - Expected results for scoring

3. Example task structure:
   ```
   tasks/my_new_task/
   ├── input_data.csv
   ├── ground_truth.csv
   └── default_prompt.txt
   ```

## Viewing Results

Results are saved in `results/`:
- Individual experiments: `results/<experiment_id>_results.json`
- Experiment log: `results/experiments_log.jsonl`

View the log:
```bash
cat results/experiments_log.jsonl | jq .
```

## Configuration

Edit `config/defaults.yaml` to change:
- Default model endpoint
- AWS region
- Default system instructions
- Experiment parameters (temperature, max_tokens, etc.)

