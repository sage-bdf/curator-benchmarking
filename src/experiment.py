"""Experiment management and execution."""
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from .bedrock_client import BedrockClient
from .task import Task
from .scorer import Scorer
from .config import Config


def compute_task_hash(task: Task) -> str:
    """
    Compute a hash of a task's content to detect changes.
    
    Args:
        task: The Task object to hash
        
    Returns:
        MD5 hash string representing the task's content
    """
    # Hash all relevant task files
    task_dir = task.task_dir
    files_to_hash = []
    
    # Include all files that affect task behavior
    for pattern in ['*.tsv', '*.csv', '*.txt', '*.json', '*.yaml', '*.yml']:
        files_to_hash.extend(list(task_dir.glob(pattern)))
    
    # Sort files for consistent hashing
    files_to_hash.sort()
    
    # Compute combined hash
    combined_content = []
    for file_path in files_to_hash:
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                combined_content.append(f"{file_path.name}:{hashlib.md5(content).hexdigest()}")
        except Exception:
            # If file can't be read, include its name
            combined_content.append(f"{file_path.name}:error")
    
    # Hash the combined content
    combined_str = "\n".join(combined_content)
    return hashlib.md5(combined_str.encode()).hexdigest()


class Experiment:
    """Represents and executes a benchmarking experiment across all tasks."""
    
    def __init__(
        self,
        tasks_dir: Path,
        model_id: str,
        system_instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        thinking: Optional[bool] = None,
        config: Optional[Config] = None
    ):
        """Initialize experiment with parameters."""
        self.config = config or Config()
        self.tasks_dir = Path(tasks_dir)
        self.model_id = model_id
        self.system_instructions = system_instructions or self.config.default_system_instructions
        
        # Get temperature from parameter or config default
        experiment_config = self.config.experiment_config
        self.temperature = temperature if temperature is not None else experiment_config.get('temperature', 0.0)
        
        # Get thinking mode from parameter or config default
        self.thinking = thinking if thinking is not None else experiment_config.get('thinking', False)
        
        self.bedrock_client = BedrockClient(self.config)
        self.scorer = Scorer()
        
        # Generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Results storage - use docs/results/ as the main results directory
        self.results_dir = Path(__file__).parent.parent / "docs" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID based on parameters."""
        params_str = f"{self.model_id}_{self.system_instructions}_{self.temperature}_{self.thinking}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _get_all_tasks(self) -> List[Task]:
        """Get all available tasks (excluding example_task)."""
        tasks = []
        for task_dir in sorted(self.tasks_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name != 'example_task':
                try:
                    task = Task(task_dir)
                    tasks.append(task)
                except Exception as e:
                    print(f"Warning: Could not load task {task_dir.name}: {e}")
                    continue
        return tasks
    
    def _get_task_hashes(self, tasks: List[Task]) -> Dict[str, str]:
        """Get hash for each task."""
        task_hashes = {}
        for task in tasks:
            try:
                task_hashes[task.name] = compute_task_hash(task)
            except Exception as e:
                print(f"Warning: Could not compute hash for task {task.name}: {e}")
                # Use a placeholder hash if computation fails
                task_hashes[task.name] = "unknown"
        return task_hashes
    
    def _run_task(self, task: Task) -> Dict[str, Any]:
        """Run a single task and return results."""
        task_start_time = time.time()
        
        input_samples = task.get_input_samples()
        ground_truth_samples = task.get_ground_truth_samples()
        
        results = []
        experiment_config = self.config.experiment_config
        
        # Include schema in prompt if available
        schema_text = ""
        if task.schema:
            # Extract just the relevant properties from the schema
            schema_properties = {}
            if 'properties' in task.schema:
                schema_properties = task.schema['properties']
            
            if schema_properties:
                # Create a simplified schema with just the properties and their enums
                simplified_schema = {
                    "type": "object",
                    "properties": {}
                }
                enum_count = 0
                for prop_name, prop_def in schema_properties.items():
                    simplified_schema["properties"][prop_name] = {
                        "description": prop_def.get("description", ""),
                        "type": prop_def.get("type", "string")
                    }
                    if "enum" in prop_def:
                        simplified_schema["properties"][prop_name]["enum"] = prop_def["enum"]
                        enum_count += 1
                
                schema_text = f"\n\nTarget Schema (controlled terminology - use these exact values where enums are specified):\n{json.dumps(simplified_schema, indent=2)}"
                print(f"    Schema loaded: {len(schema_properties)} properties, {enum_count} with controlled terminology (enums)")
                # Debug: show schema snippet
                schema_preview = json.dumps(simplified_schema, indent=2)[:200]
                print(f"    Schema preview: {schema_preview}...")
            else:
                print(f"    Warning: Schema found but no properties defined")
        else:
            print(f"    No schema file found for this task")
        
        for idx, sample in enumerate(input_samples):
            print(f"    Processing sample {idx + 1}/{len(input_samples)}...", end=' ', flush=True)
            
            # Get ground truth for this sample if available
            ground_truth = None
            if ground_truth_samples and idx < len(ground_truth_samples):
                ground_truth = ground_truth_samples[idx]
            
            # Use task's format_prompt method (handles custom formatters if they exist)
            formatted_prompt = task.format_prompt(
                sample=sample,
                ground_truth=ground_truth,
                schema_text=schema_text
            )
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                model_id=self.model_id,
                prompt=formatted_prompt,
                system_instructions=self.system_instructions,
                temperature=self.temperature,
                thinking=self.thinking,
                max_tokens=experiment_config.get('max_tokens', 4096),
                max_retries=experiment_config.get('max_retries', 3)
            )
            
            # Initialize score to None
            score = None
            
            if not response.get('success', False):
                print(f"✗ Failed: {response.get('error', 'Unknown error')}")
            else:
                # Score if ground truth available
                if ground_truth_samples and idx < len(ground_truth_samples):
                    prediction_content = response.get('content', '')
                    ground_truth_dict = ground_truth_samples[idx]
                    
                    # Debug output for first sample of first task
                    if idx == 0 and task.name == list(self._get_all_tasks())[0].name:
                        print(f"\n    [DEBUG] Sample {idx + 1} - Prediction (first 200 chars):")
                        print(f"    {prediction_content[:200] if prediction_content else '(empty)'}...")
                        print(f"    [DEBUG] Ground truth: {ground_truth_dict}")
                        # Show raw response structure if content is empty
                        if not prediction_content and 'raw_response' in response:
                            print(f"    [DEBUG] Raw response keys: {list(response.get('raw_response', {}).keys())}")
                            print(f"    [DEBUG] Raw response (first 500 chars): {str(response.get('raw_response', {}))[:500]}")
                        # Use scorer's extraction method to show what will be scored
                        json_str = self.scorer._extract_json(prediction_content)
                        try:
                            pred_parsed = json.loads(json_str)
                            print(f"    [DEBUG] Extracted and parsed prediction: {pred_parsed}")
                        except Exception as e:
                            print(f"    [DEBUG] Failed to parse extracted JSON: {e}")
                            print(f"    [DEBUG] Extracted JSON string: {json_str[:200] if json_str else '(empty)'}")
                    
                    score = self.scorer.score(
                        prediction=prediction_content,
                        ground_truth=ground_truth_dict,
                        input_data=sample
                    )
                    if score is not None:
                        print(f"✓ Score: {(score * 100):.2f}%")
                    else:
                        print(f"✓ Score: N/A (scoring failed)")
                else:
                    print("✓ (no ground truth for scoring)")
            
            # Extract token usage from response
            usage = response.get('usage', {})
            input_tokens = usage.get('inputTokens') or usage.get('input_tokens') or 0
            output_tokens = usage.get('outputTokens') or usage.get('output_tokens') or 0
            total_tokens = usage.get('totalTokens') or usage.get('total_tokens') or (input_tokens + output_tokens)
            
            results.append({
                'sample_index': idx,
                'input': sample,
                'response': response,
                'score': score,
                'ground_truth': ground_truth_samples[idx] if ground_truth_samples else None,
                'token_usage': {
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'total_tokens': total_tokens
                }
            })
        
        # Calculate metrics for this task
        metrics = self._calculate_metrics(results)
        
        # Aggregate token usage for this task
        total_input_tokens = sum(r.get('token_usage', {}).get('input_tokens', 0) for r in results)
        total_output_tokens = sum(r.get('token_usage', {}).get('output_tokens', 0) for r in results)
        total_tokens = sum(r.get('token_usage', {}).get('total_tokens', 0) for r in results)
        
        task_end_time = time.time()
        task_duration_seconds = task_end_time - task_start_time
        
        return {
            'task_name': task.name,
            'num_samples': len(input_samples),
            'results': results,
            'metrics': metrics,
            'duration_seconds': task_duration_seconds,
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens
            }
        }
    
    def _get_experiment_task_file(self, task_name: str) -> Path:
        """Get the file path for an experiment-task result."""
        return self.results_dir / f"{self.experiment_id}_{task_name}.json"
    
    def _load_experiment_task_result(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Load an experiment-task result if it exists."""
        task_file = self._get_experiment_task_file(task_name)
        if task_file.exists():
            try:
                with open(task_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Warning: Could not load {task_file.name}: {e}")
        return None
    
    def _save_experiment_task_result(self, task_name: str, task_result: Dict[str, Any], task_hash: str):
        """Save an experiment-task result to its own file."""
        task_file = self._get_experiment_task_file(task_name)
        
        # Prepare the result with experiment metadata
        result_data = {
            'experiment_id': self.experiment_id,
            'task_name': task_name,
            'task_hash': task_hash,
            'model_id': self.model_id,
            'system_instructions': self.system_instructions,
            'temperature': self.temperature,
            'thinking': self.thinking,
            'timestamp': datetime.now().isoformat(),
            'task_result': task_result
        }
        
        with open(task_file, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def _experiment_exists_in_log(self) -> bool:
        """Check if this experiment exists in the experiments log."""
        log_file = self.results_dir / "experiments_log.jsonl"
        if not log_file.exists():
            return False
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('<<<<<<<') or line.startswith('=======') or line.startswith('>>>>>>>'):
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get('experiment_id') == self.experiment_id:
                            return True
                    except:
                        continue
        except:
            pass
        return False
    
    def _log_experiment(self):
        """Log experiment metadata to experiments_log.jsonl."""
        log_file = self.results_dir / "experiments_log.jsonl"
        summary = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'system_instructions': self.system_instructions,
            'temperature': self.temperature,
            'thinking': self.thinking
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
    
    def run(self, update_other_experiments: bool = True) -> Dict[str, Any]:
        """
        Execute the experiment across all tasks and return aggregated results.
        
        Each task result is stored in a separate file: experiment_id_taskname.json
        """
        # Get all current tasks and their hashes
        tasks = self._get_all_tasks()
        if not tasks:
            raise ValueError("No tasks found to run")
        
        current_task_hashes = self._get_task_hashes(tasks)
        
        # Check if experiment exists in log
        experiment_exists = self._experiment_exists_in_log()
        
        if experiment_exists:
            print(f"\n{'='*60}")
            print(f"Experiment {self.experiment_id} exists")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Running new experiment {self.experiment_id}")
            print(f"{'='*60}")
            print(f"  Model: {self.model_id}")
            print(f"  System Instructions: {self.system_instructions[:50]}...")
            print(f"  Temperature: {self.temperature}")
            print(f"  Thinking: {self.thinking}")
            print(f"  Timestamp: {datetime.now().isoformat()}")
            print(f"\n  Found {len(tasks)} tasks to run:")
            for task in tasks:
                print(f"    - {task.name} ({len(task.get_input_samples())} samples)")
            print(f"\n  Starting experiment execution...\n")
        
        # Determine which tasks need to be run
        tasks_to_run = []
        for task in tasks:
            task_name = task.name
            current_hash = current_task_hashes.get(task_name)
            
            # Load existing task result if it exists
            existing_task_result = self._load_experiment_task_result(task_name)
            
            if existing_task_result:
                stored_hash = existing_task_result.get('task_hash')
                if current_hash == stored_hash:
                    print(f"  Task '{task_name}' unchanged (hash: {current_hash[:8]}), skipping")
                    continue
                else:
                    stored_hash_str = stored_hash[:8] if stored_hash else "none"
                    current_hash_str = current_hash[:8] if current_hash else "none"
                    print(f"  Task '{task_name}' has changed (hash: {stored_hash_str} -> {current_hash_str}), will re-run")
                    tasks_to_run.append(task)
            else:
                # Task file doesn't exist - need to run it
                if experiment_exists:
                    print(f"  Task '{task_name}' missing, will run")
                else:
                    print(f"  Task '{task_name}' will run")
                tasks_to_run.append(task)
        
        if not tasks_to_run:
            print(f"  All tasks are up to date, no re-running needed")
            print(f"{'='*60}\n")
            # Update experiment log if this is a new experiment
            if not self._experiment_exists_in_log():
                self._log_experiment()
            # Load and return aggregated results
            return self._aggregate_experiment_results(tasks, current_task_hashes)
        
        # Run tasks that need updating
        print(f"\n  Running {len(tasks_to_run)} task(s)...\n")
        
        for task_idx, task in enumerate(tasks_to_run, 1):
            print(f"{'='*60}")
            print(f"Task {task_idx}/{len(tasks_to_run)}: {task.name}")
            print(f"{'='*60}")
            try:
                task_result = self._run_task(task)
                task_hash = current_task_hashes.get(task.name)
                
                # Save individual task result to its own file
                self._save_experiment_task_result(task.name, task_result, task_hash)
                
                print(f"  ✓ Task {task.name} completed successfully")
                avg_score = task_result['metrics'].get('average_score')
                if avg_score is not None:
                    print(f"    Accuracy: {(avg_score * 100):.2f}%")
                else:
                    print(f"    Accuracy: N/A (no scores calculated)")
                print(f"    Samples: {task_result['metrics']['total_samples']}")
                duration = task_result.get('duration_seconds', 0)
                print(f"    Duration: {duration:.2f}s")
                token_usage = task_result.get('token_usage', {})
                if token_usage.get('total_tokens', 0) > 0:
                    print(f"    Tokens: {token_usage.get('total_tokens', 0):,} total ({token_usage.get('input_tokens', 0):,} in, {token_usage.get('output_tokens', 0):,} out)")
                print()
                
            except Exception as e:
                print(f"  ✗ Error running task {task.name}: {e}")
                import traceback
                traceback.print_exc()
                # Save error result
                error_result = {
                    'task_name': task.name,
                    'error': str(e),
                    'metrics': {
                        'total_samples': 0,
                        'successful_runs': 0,
                        'failed_runs': 0,
                        'success_rate': 0,
                        'average_score': None
                    },
                    'duration_seconds': 0
                }
                task_hash = current_task_hashes.get(task.name)
                self._save_experiment_task_result(task.name, error_result, task_hash)
                print()
        
        # Update experiment log if this is a new experiment
        # (Check again in case it was just created)
        if not self._experiment_exists_in_log():
            self._log_experiment()
        
        # Aggregate and return results
        return self._aggregate_experiment_results(tasks, current_task_hashes)
    
    def _aggregate_experiment_results(self, tasks: List[Task], current_task_hashes: Dict[str, str]) -> Dict[str, Any]:
        """
        Aggregate results from all experiment-task files for this experiment.
        
        Returns a dictionary in the same format as before for compatibility.
        """
        task_results = {}
        all_scores = []
        total_samples = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_duration_seconds = 0
        tasks_completed = 0
        tasks_failed = 0
        
        for task in tasks:
            task_name = task.name
            task_file_data = self._load_experiment_task_result(task_name)
            
            if task_file_data:
                task_result = task_file_data.get('task_result', {})
                task_results[task_name] = task_result
                
                if 'error' not in task_result:
                    tasks_completed += 1
                    metrics = task_result.get('metrics', {})
                    total_samples += metrics.get('total_samples', 0)
                    
                    avg_score = metrics.get('average_score')
                    if avg_score is not None:
                        all_scores.append(avg_score)
                    
                    token_usage = task_result.get('token_usage', {})
                    total_input_tokens += token_usage.get('input_tokens', 0)
                    total_output_tokens += token_usage.get('output_tokens', 0)
                    total_tokens += token_usage.get('total_tokens', 0)
                    total_duration_seconds += task_result.get('duration_seconds', 0)
                else:
                    tasks_failed += 1
        
        overall_metrics = {
            'total_samples': total_samples,
            'tasks_completed': tasks_completed,
            'tasks_failed': tasks_failed,
            'average_accuracy': sum(all_scores) / len(all_scores) if all_scores else None,
            'task_metrics': {name: result.get('metrics', {}) for name, result in task_results.items()},
            'duration_seconds': total_duration_seconds,
            'token_usage': {
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_tokens': total_tokens
            }
        }
        
        print(f"\n{'='*60}")
        print(f"Experiment {self.experiment_id} Summary")
        print(f"{'='*60}")
        print(f"  Tasks completed: {tasks_completed}")
        print(f"  Tasks failed: {tasks_failed}")
        print(f"  Total samples: {total_samples}")
        if overall_metrics['average_accuracy'] is not None:
            print(f"  Average accuracy: {(overall_metrics['average_accuracy'] * 100):.2f}%")
        if total_tokens > 0:
            print(f"  Total tokens: {total_tokens:,} ({total_input_tokens:,} input, {total_output_tokens:,} output)")
        print(f"  Total duration: {total_duration_seconds:.2f}s")
        print(f"{'='*60}\n")
        
        return {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'system_instructions': self.system_instructions,
            'temperature': self.temperature,
            'thinking': self.thinking,
            'task_results': task_results,
            'overall_metrics': overall_metrics
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results."""
        scores = [r['score'] for r in results if r['score'] is not None]
        successful_runs = [r for r in results if r['response'].get('success', False)]
        
        metrics = {
            'total_samples': len(results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(results) - len(successful_runs),
            'success_rate': len(successful_runs) / len(results) if results else 0
        }
        
        if scores:
            metrics['average_score'] = sum(scores) / len(scores)
            metrics['min_score'] = min(scores)
            metrics['max_score'] = max(scores)
            metrics['num_scored'] = len(scores)
        else:
            metrics['average_score'] = None
            metrics['min_score'] = None
            metrics['max_score'] = None
            metrics['num_scored'] = 0
        
        return metrics
    
    def _update_other_experiments_for_new_tasks(
        self,
        new_task_names: List[str],
        current_task_hashes: Dict[str, str]
    ):
        """
        Update all other existing experiments to include new tasks.
        
        Args:
            new_task_names: List of task names that are new
            current_task_hashes: Current task hashes for all tasks
        """
        if not new_task_names:
            return
        
        print(f"\n{'='*60}")
        print(f"New tasks detected: {', '.join(new_task_names)}")
        print(f"Updating other existing experiments...")
        print(f"{'='*60}")
        
        # Find all other experiment result files
        result_files = list(self.results_dir.glob("*_results.json"))
        other_experiments = [
            f for f in result_files 
            if f.name != f"{self.experiment_id}_results.json"
        ]
        
        if not other_experiments:
            print(f"  No other experiments found to update")
            return
        
        print(f"  Found {len(other_experiments)} other experiment(s) to check")
        
        updated_count = 0
        for result_file in other_experiments:
            try:
                with open(result_file, 'r') as f:
                    experiment_data = json.load(f)
                
                other_experiment_id = experiment_data.get('experiment_id')
                if not other_experiment_id:
                    continue
                
                # Check if this experiment is missing any of the new tasks
                existing_task_names = set(experiment_data.get('task_results', {}).keys())
                missing_tasks = [t for t in new_task_names if t not in existing_task_names]
                
                if not missing_tasks:
                    continue  # This experiment already has all new tasks
                
                print(f"\n  Updating experiment {other_experiment_id[:16]}... (missing: {', '.join(missing_tasks)})")
                
                # Recreate the experiment from stored parameters
                other_experiment = Experiment(
                    tasks_dir=self.tasks_dir,
                    model_id=experiment_data.get('model_id', self.config.default_model),
                    system_instructions=experiment_data.get('system_instructions'),
                    temperature=experiment_data.get('temperature'),
                    thinking=experiment_data.get('thinking', False),
                    config=self.config
                )
                
                # Run will automatically detect and add the new tasks
                other_experiment.run(update_other_experiments=False)  # Prevent recursive updates
                updated_count += 1
                
            except Exception as e:
                print(f"  Error updating experiment {result_file.name}: {e}")
                continue
        
        if updated_count > 0:
            print(f"\n  ✓ Updated {updated_count} other experiment(s) with new tasks")
        else:
            print(f"\n  All other experiments already have the new tasks")
        print(f"{'='*60}\n")
    
