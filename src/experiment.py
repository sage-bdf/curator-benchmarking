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
            
            # Format prompt with sample data and schema
            sample_str = json.dumps(sample, indent=2)
            formatted_prompt = f"{prompt}{schema_text}\n\nInput data:\n{sample_str}"
            
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
        # Check if experiment already exists
        results_file = self.results_dir / f"{self.experiment_id}_results.json"
        if results_file.exists():
            print(f"\n{'='*60}")
            print(f"Experiment {self.experiment_id} already exists - skipping")
            print(f"{'='*60}")
            print(f"  Loading existing results from: {results_file}")
            
            # Load and return existing results
            with open(results_file, 'r') as f:
                existing_result = json.load(f)
            
            print(f"  Model: {existing_result.get('model_id', 'N/A')}")
            print(f"  Temperature: {existing_result.get('temperature', 'N/A')}")
            print(f"  Thinking: {existing_result.get('thinking', 'N/A')}")
            print(f"  Original timestamp: {existing_result.get('timestamp', 'N/A')}")
            print(f"  Tasks completed: {existing_result.get('overall_metrics', {}).get('tasks_completed', 'N/A')}")
            if existing_result.get('overall_metrics', {}).get('average_accuracy') is not None:
                avg_acc = existing_result['overall_metrics']['average_accuracy']
                print(f"  Average accuracy: {(avg_acc * 100):.2f}%")
            print(f"{'='*60}\n")
            
            return existing_result
        
        print(f"\n{'='*60}")
        print(f"Running experiment {self.experiment_id}")
        print(f"{'='*60}")
        print(f"  Model: {self.model_id}")
        print(f"  System Instructions: {self.system_instructions[:50]}...")
        print(f"  Temperature: {self.temperature}")
        print(f"  Thinking: {self.thinking}")
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
                avg_score = task_result['metrics'].get('average_score')
                if avg_score is not None:
                    print(f"    Accuracy: {(avg_score * 100):.2f}%")
                else:
                    print(f"    Accuracy: N/A (no scores calculated)")
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
            'temperature': self.temperature,
            'thinking': self.thinking,
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
