"""Command-line interface for running experiments."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from .config import Config
from .task import Task
from .experiment import Experiment


def list_tasks(tasks_dir: Path):
    """List all available tasks."""
    tasks_dir = Path(tasks_dir)
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        return
    
    task_dirs = [d for d in tasks_dir.iterdir() if d.is_dir() and d.name != 'example_task']
    
    if not task_dirs:
        print("No tasks found.")
        return
    
    print("Available tasks:")
    for task_dir in sorted(task_dirs):
        task = Task(task_dir)
        print(f"  - {task.name}")
        print(f"    Input samples: {len(task.input_data)}")
        print(f"    Ground truth: {'Yes' if task.ground_truth is not None else 'No'}")


def run_experiment(
    tasks_dir: Path,
    model_id: Optional[str] = None,
    system_instructions_file: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking: Optional[bool] = None,
    config: Optional[Config] = None
):
    """Run an experiment across all tasks."""
    if config is None:
        config = Config()
    
    tasks_dir = Path(tasks_dir)
    
    # Load custom system instructions if provided
    system_instructions = None
    if system_instructions_file:
        with open(system_instructions_file, 'r') as f:
            system_instructions = f.read()
    
    # Use default model if not specified
    model_id = model_id or config.default_model
    
    # Create and run experiment (runs all tasks)
    experiment = Experiment(
        tasks_dir=tasks_dir,
        model_id=model_id,
        system_instructions=system_instructions,
        temperature=temperature,
        thinking=thinking,
        config=config
    )
    
    result = experiment.run()
    
    print("\n" + "="*60)
    print("Experiment Complete")
    print("="*60)
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Overall Metrics: {result['overall_metrics']}")
    print("="*60)


def run_experiment_suite(
    tasks_dir: Path,
    models: List[str],
    system_instructions_files: Optional[List[str]] = None,
    config: Optional[Config] = None
):
    """Run a suite of experiments with different parameter combinations."""
    if config is None:
        config = Config()
    
    tasks_dir = Path(tasks_dir)
    
    # Default to using task defaults if not provided
    system_instructions_list = [None]
    if system_instructions_files:
        system_instructions_list = []
        for file in system_instructions_files:
            with open(file, 'r') as f:
                system_instructions_list.append(f.read())
    
    # Run all combinations (each experiment runs all tasks)
    total_experiments = len(models) * len(system_instructions_list)
    print(f"Running {total_experiments} experiments...")
    
    experiment_num = 0
    for model_id in models:
        for system_instructions in system_instructions_list:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Running experiment...")
            
            experiment = Experiment(
                tasks_dir=tasks_dir,
                model_id=model_id,
                system_instructions=system_instructions,
                config=config
            )
            
            experiment.run()


def update_all_experiments(
    tasks_dir: Path,
    config: Optional[Config] = None
):
    """Update all existing experiments with changed/new tasks."""
    if config is None:
        config = Config()
    
    tasks_dir = Path(tasks_dir)
    results_dir = Path(__file__).parent.parent / "docs" / "results"
    
    if not results_dir.exists():
        print("No results directory found. No experiments to update.")
        return
    
    # Load experiments from the log file
    log_file = results_dir / "experiments_log.jsonl"
    print(f"Looking for experiments log at: {log_file}")
    print(f"Results directory exists: {results_dir.exists()}")
    print(f"Log file exists: {log_file.exists()}")
    
    if not log_file.exists():
        print(f"No experiments log found at {log_file}. No experiments to update.")
        return
    
    experiments = []
    lines_read = 0
    lines_skipped = 0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                lines_read += 1
                line = line.strip()
                if not line:
                    continue
                # Skip merge conflict markers
                if line.startswith('<<<<<<<') or line.startswith('=======') or line.startswith('>>>>>>>'):
                    lines_skipped += 1
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get('experiment_id'):
                        experiments.append(entry)
                    else:
                        lines_skipped += 1
                except Exception as parse_error:
                    # Skip invalid JSON lines
                    lines_skipped += 1
                    continue
    except Exception as e:
        print(f"Error reading experiments log: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Read {lines_read} lines, parsed {len(experiments)} valid experiments, skipped {lines_skipped} lines")
    
    if not experiments:
        print("No existing experiments found in log.")
        return
    
    # Deduplicate by experiment_id, keeping the most recent entry
    experiment_map = {}
    for exp in experiments:
        exp_id = exp.get('experiment_id')
        if not exp_id:
            continue
        existing = experiment_map.get(exp_id)
        if not existing or exp.get('timestamp', '') > existing.get('timestamp', ''):
            experiment_map[exp_id] = exp
    
    unique_experiments = list(experiment_map.values())
    
    if not unique_experiments:
        print("No valid experiments found after deduplication.")
        return
    
    print(f"Found {len(unique_experiments)} unique experiments to check for updates...\n")
    
    updated_count = 0
    for exp_entry in unique_experiments:
        try:
            experiment_id = exp_entry.get('experiment_id')
            if not experiment_id:
                continue
            
            # Recreate experiment from stored parameters
            experiment = Experiment(
                tasks_dir=tasks_dir,
                model_id=exp_entry.get('model_id', config.default_model),
                system_instructions=exp_entry.get('system_instructions'),
                temperature=exp_entry.get('temperature'),
                thinking=exp_entry.get('thinking', False),
                config=config
            )
            
            # Run will automatically detect and update changed/new tasks
            experiment.run()
            updated_count += 1
            
        except Exception as e:
            print(f"Error updating experiment {exp_entry.get('experiment_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Update complete: {updated_count}/{len(experiments)} experiments processed")
    print(f"{'='*60}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmarking framework for LLM metadata curation tasks"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List tasks command
    list_parser = subparsers.add_parser('list', help='List available tasks')
    list_parser.add_argument(
        '--tasks-dir',
        type=str,
        default='tasks',
        help='Directory containing tasks'
    )
    
    # Run experiment command
    run_parser = subparsers.add_parser('run', help='Run an experiment across all tasks')
    run_parser.add_argument(
        '--tasks-dir',
        type=str,
        default='tasks',
        help='Directory containing tasks'
    )
    run_parser.add_argument(
        '--model',
        type=str,
        help='Model endpoint ID (default: from config)'
    )
    run_parser.add_argument(
        '--system-instructions',
        type=str,
        help='Path to custom system instructions file'
    )
    run_parser.add_argument(
        '--temperature',
        type=float,
        help='Sampling temperature (default: from config, usually 0.0)'
    )
    run_parser.add_argument(
        '--thinking',
        action='store_true',
        help='Enable thinking/reasoning mode (for supported models)'
    )
    
    # Run suite command
    suite_parser = subparsers.add_parser('suite', help='Run a suite of experiments across all tasks')
    suite_parser.add_argument(
        '--tasks-dir',
        type=str,
        default='tasks',
        help='Directory containing tasks'
    )
    suite_parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='Model endpoint IDs to test'
    )
    suite_parser.add_argument(
        '--system-instructions',
        nargs='+',
        help='Paths to system instructions files'
    )
    
    # Update all experiments command
    update_parser = subparsers.add_parser('update-all', help='Update all existing experiments with changed/new tasks')
    update_parser.add_argument(
        '--tasks-dir',
        type=str,
        default='tasks',
        help='Directory containing tasks'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_tasks(Path(args.tasks_dir))
    
    elif args.command == 'run':
        run_experiment(
            tasks_dir=Path(args.tasks_dir),
            model_id=args.model,
            system_instructions_file=args.system_instructions,
            temperature=args.temperature,
            thinking=args.thinking
        )
    
    elif args.command == 'suite':
        run_experiment_suite(
            tasks_dir=Path(args.tasks_dir),
            models=args.models,
            system_instructions_files=args.system_instructions
        )
    
    elif args.command == 'update-all':
        update_all_experiments(
            tasks_dir=Path(args.tasks_dir)
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

