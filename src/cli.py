"""Command-line interface for running experiments."""
import argparse
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
    
    task_dirs = [d for d in tasks_dir.iterdir() if d.is_dir()]
    
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
    task_name: str,
    tasks_dir: Path,
    model_id: Optional[str] = None,
    system_instructions_file: Optional[str] = None,
    prompt_file: Optional[str] = None,
    config: Optional[Config] = None
):
    """Run a single experiment."""
    if config is None:
        config = Config()
    
    tasks_dir = Path(tasks_dir)
    task_dir = tasks_dir / task_name
    
    if not task_dir.exists():
        print(f"Task not found: {task_name}")
        sys.exit(1)
    
    task = Task(task_dir)
    
    # Load custom system instructions if provided
    system_instructions = None
    if system_instructions_file:
        with open(system_instructions_file, 'r') as f:
            system_instructions = f.read()
    
    # Load custom prompt if provided
    prompt = None
    if prompt_file:
        with open(prompt_file, 'r') as f:
            prompt = f.read()
    
    # Use default model if not specified
    model_id = model_id or config.default_model
    
    # Create and run experiment
    experiment = Experiment(
        task=task,
        model_id=model_id,
        system_instructions=system_instructions,
        prompt=prompt,
        config=config
    )
    
    result = experiment.run()
    
    print("\n" + "="*60)
    print("Experiment Complete")
    print("="*60)
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Metrics: {result['metrics']}")
    print("="*60)


def run_experiment_suite(
    task_name: str,
    tasks_dir: Path,
    models: List[str],
    system_instructions_files: Optional[List[str]] = None,
    prompt_files: Optional[List[str]] = None,
    config: Optional[Config] = None
):
    """Run a suite of experiments with different parameter combinations."""
    if config is None:
        config = Config()
    
    tasks_dir = Path(tasks_dir)
    task_dir = tasks_dir / task_name
    
    if not task_dir.exists():
        print(f"Task not found: {task_name}")
        sys.exit(1)
    
    task = Task(task_dir)
    
    # Default to using task defaults if not provided
    system_instructions_list = [None]
    if system_instructions_files:
        system_instructions_list = []
        for file in system_instructions_files:
            with open(file, 'r') as f:
                system_instructions_list.append(f.read())
    
    prompt_list = [None]
    if prompt_files:
        prompt_list = []
        for file in prompt_files:
            with open(file, 'r') as f:
                prompt_list.append(f.read())
    
    # Run all combinations
    total_experiments = len(models) * len(system_instructions_list) * len(prompt_list)
    print(f"Running {total_experiments} experiments...")
    
    experiment_num = 0
    for model_id in models:
        for system_instructions in system_instructions_list:
            for prompt in prompt_list:
                experiment_num += 1
                print(f"\n[{experiment_num}/{total_experiments}] Running experiment...")
                
                experiment = Experiment(
                    task=task,
                    model_id=model_id,
                    system_instructions=system_instructions,
                    prompt=prompt,
                    config=config
                )
                
                experiment.run()


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
    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('task', help='Task name to run')
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
        '--prompt',
        type=str,
        help='Path to custom prompt file'
    )
    
    # Run suite command
    suite_parser = subparsers.add_parser('suite', help='Run a suite of experiments')
    suite_parser.add_argument('task', help='Task name to run')
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
    suite_parser.add_argument(
        '--prompts',
        nargs='+',
        help='Paths to prompt files'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_tasks(Path(args.tasks_dir))
    
    elif args.command == 'run':
        run_experiment(
            task_name=args.task,
            tasks_dir=Path(args.tasks_dir),
            model_id=args.model,
            system_instructions_file=args.system_instructions,
            prompt_file=args.prompt
        )
    
    elif args.command == 'suite':
        run_experiment_suite(
            task_name=args.task,
            tasks_dir=Path(args.tasks_dir),
            models=args.models,
            system_instructions_files=args.system_instructions,
            prompt_files=args.prompts
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

