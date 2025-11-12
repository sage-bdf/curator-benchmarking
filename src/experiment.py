"""Experiment management and execution."""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from .bedrock_client import BedrockClient
from .task import Task
from .scorer import Scorer
from .config import Config


class Experiment:
    """Represents and executes a benchmarking experiment across all tasks."""
    
    def __init__(
        self,
        tasks_dir: Path,
        model_id: str,
        system_instructions: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize experiment with parameters."""
        self.config = config or Config()
        self.tasks_dir = Path(tasks_dir)
        self.model_id = model_id
        self.system_instructions = system_instructions or self.config.default_system_instructions
        self.bedrock_client = BedrockClient(self.config)
        self.scorer = Scorer()
        
        # Generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Results storage
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID based on parameters."""
        params_str = f"{self.model_id}_{self.system_instructions}"
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
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
    
    def _run_task(self, task: Task) -> Dict[str, Any]:
        """Run a single task and return results."""
        prompt = task.default_prompt
        input_samples = task.get_input_samples()
        ground_truth_samples = task.get_ground_truth_samples()
        
        results = []
        experiment_config = self.config.experiment_config
        
        # Include schema in prompt if available
        schema_text = ""
        if task.schema:
            schema_text = f"\n\nTarget Schema (controlled terminology):\n{json.dumps(task.schema, indent=2)}"
        
        for idx, sample in enumerate(input_samples):
            print(f"    Processing sample {idx + 1}/{len(input_samples)}...", end=' ', flush=True)
            
            # Format prompt with sample data and schema
            sample_str = json.dumps(sample, indent=2)
            formatted_prompt = f"{prompt}{schema_text}\n\nInput data:\n{sample_str}"
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                model_id=self.model_id,
                prompt=formatted_prompt,
                system_instructions=self.system_instructions,
                temperature=experiment_config.get('temperature', 0.0),
                max_tokens=experiment_config.get('max_tokens', 4096),
                max_retries=experiment_config.get('max_retries', 3)
            )
            
            if not response.get('success', False):
                print(f"✗ Failed: {response.get('error', 'Unknown error')}")
            else:
                # Score if ground truth available
                score = None
                if ground_truth_samples and idx < len(ground_truth_samples):
                    score = self.scorer.score(
                        prediction=response.get('content', ''),
                        ground_truth=ground_truth_samples[idx]
                    )
                    print(f"✓ Score: {(score * 100):.2f}%")
                else:
                    print("✓ (no ground truth for scoring)")
            
            results.append({
                'sample_index': idx,
                'input': sample,
                'response': response,
                'score': score,
                'ground_truth': ground_truth_samples[idx] if ground_truth_samples else None
            })
        
        # Calculate metrics for this task
        metrics = self._calculate_metrics(results)
        
        return {
            'task_name': task.name,
            'num_samples': len(input_samples),
            'results': results,
            'metrics': metrics
        }
    
    def run(self) -> Dict[str, Any]:
        """Execute the experiment across all tasks and return results."""
        print(f"\n{'='*60}")
        print(f"Running experiment {self.experiment_id}")
        print(f"{'='*60}")
        print(f"  Model: {self.model_id}")
        print(f"  System Instructions: {self.system_instructions[:50]}...")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        
        tasks = self._get_all_tasks()
        if not tasks:
            raise ValueError("No tasks found to run")
        
        print(f"\n  Found {len(tasks)} tasks to run:")
        for task in tasks:
            print(f"    - {task.name} ({len(task.get_input_samples())} samples)")
        
        print(f"\n  Starting experiment execution...\n")
        
        task_results = {}
        overall_scores = []
        total_samples = 0
        
        for task_idx, task in enumerate(tasks, 1):
            print(f"{'='*60}")
            print(f"Task {task_idx}/{len(tasks)}: {task.name}")
            print(f"{'='*60}")
            try:
                task_result = self._run_task(task)
                task_results[task.name] = task_result
                
                # Collect scores for overall metrics
                if task_result['metrics'].get('average_score') is not None:
                    overall_scores.append(task_result['metrics']['average_score'])
                total_samples += task_result['metrics']['total_samples']
                
                print(f"  ✓ Task {task.name} completed successfully")
                print(f"    Accuracy: {(task_result['metrics'].get('average_score', 0) * 100):.2f}%")
                print(f"    Samples: {task_result['metrics']['total_samples']}")
                print()
                
            except Exception as e:
                print(f"  ✗ Error running task {task.name}: {e}")
                import traceback
                traceback.print_exc()
                task_results[task.name] = {
                    'task_name': task.name,
                    'error': str(e),
                    'metrics': {
                        'total_samples': 0,
                        'successful_runs': 0,
                        'failed_runs': 0,
                        'success_rate': 0,
                        'average_score': None
                    }
                }
                print()
        
        # Calculate overall metrics
        overall_metrics = {
            'total_samples': total_samples,
            'tasks_completed': len([r for r in task_results.values() if 'error' not in r]),
            'tasks_failed': len([r for r in task_results.values() if 'error' in r]),
            'average_accuracy': sum(overall_scores) / len(overall_scores) if overall_scores else None,
            'task_metrics': {name: result['metrics'] for name, result in task_results.items()}
        }
        
        # Prepare experiment result
        experiment_result = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'model_id': self.model_id,
            'system_instructions': self.system_instructions,
            'task_results': task_results,
            'overall_metrics': overall_metrics
        }
        
        # Save results
        self._save_results(experiment_result)
        
        print(f"\n{'='*60}")
        print(f"Experiment {self.experiment_id} Complete!")
        print(f"{'='*60}")
        print(f"  Tasks completed: {overall_metrics['tasks_completed']}")
        print(f"  Tasks failed: {overall_metrics['tasks_failed']}")
        print(f"  Total samples: {overall_metrics['total_samples']}")
        if overall_metrics['average_accuracy'] is not None:
            print(f"  Average accuracy: {(overall_metrics['average_accuracy'] * 100):.2f}%")
        print(f"{'='*60}\n")
        
        return experiment_result
    
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
    
    def _save_results(self, experiment_result: Dict[str, Any]):
        """Save experiment results to disk."""
        # Save full results as JSON
        results_file = self.results_dir / f"{self.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        # Save summary to experiments log (one entry per experiment)
        summary = {
            'experiment_id': experiment_result['experiment_id'],
            'timestamp': experiment_result['timestamp'],
            'model_id': experiment_result['model_id'],
            'system_instructions': experiment_result['system_instructions'],
            'overall_metrics': experiment_result['overall_metrics']
        }
        
        log_file = self.results_dir / "experiments_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        
        print(f"  Results saved to {results_file}")
        print(f"  Log entry added to experiments_log.jsonl")
