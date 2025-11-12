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
    """Represents and executes a benchmarking experiment."""
    
    def __init__(
        self,
        task: Task,
        model_id: str,
        system_instructions: Optional[str] = None,
        prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize experiment with parameters."""
        self.config = config or Config()
        self.task = task
        self.model_id = model_id
        self.system_instructions = system_instructions or self.config.default_system_instructions
        self.prompt = prompt or task.default_prompt
        self.bedrock_client = BedrockClient(self.config)
        self.scorer = Scorer()
        
        # Generate experiment ID
        self.experiment_id = self._generate_experiment_id()
        
        # Results storage
        self.results_dir = Path(__file__).parent.parent / "results"
        self.results_dir.mkdir(exist_ok=True)
    
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID based on parameters."""
        params_str = f"{self.task.name}_{self.model_id}_{self.system_instructions}_{self.prompt}"
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def run(self) -> Dict[str, Any]:
        """Execute the experiment and return results."""
        print(f"Running experiment {self.experiment_id}")
        print(f"  Task: {self.task.name}")
        print(f"  Model: {self.model_id}")
        print(f"  System Instructions: {self.system_instructions[:50]}...")
        print(f"  Prompt: {self.prompt[:50]}...")
        
        input_samples = self.task.get_input_samples()
        ground_truth_samples = self.task.get_ground_truth_samples()
        
        results = []
        experiment_config = self.config.experiment_config
        
        for idx, sample in enumerate(input_samples):
            print(f"  Processing sample {idx + 1}/{len(input_samples)}")
            
            # Format prompt with sample data
            formatted_prompt = self._format_prompt(sample)
            
            # Invoke model
            response = self.bedrock_client.invoke_model(
                model_id=self.model_id,
                prompt=formatted_prompt,
                system_instructions=self.system_instructions,
                temperature=experiment_config.get('temperature', 0.0),
                max_tokens=experiment_config.get('max_tokens', 4096),
                max_retries=experiment_config.get('max_retries', 3)
            )
            
            # Score if ground truth available
            score = None
            if ground_truth_samples and idx < len(ground_truth_samples):
                score = self.scorer.score(
                    prediction=response.get('content', ''),
                    ground_truth=ground_truth_samples[idx]
                )
            
            results.append({
                'sample_index': idx,
                'input': sample,
                'response': response,
                'score': score,
                'ground_truth': ground_truth_samples[idx] if ground_truth_samples else None
            })
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(results)
        
        # Prepare experiment result
        experiment_result = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'task_name': self.task.name,
            'model_id': self.model_id,
            'system_instructions': self.system_instructions,
            'prompt': self.prompt,
            'num_samples': len(input_samples),
            'results': results,
            'metrics': metrics
        }
        
        # Save results
        self._save_results(experiment_result)
        
        return experiment_result
    
    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format the prompt with sample data."""
        # Convert sample to a readable format
        sample_str = json.dumps(sample, indent=2)
        return f"{self.prompt}\n\nInput data:\n{sample_str}"
    
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
        
        # Save summary to experiments log
        summary = {
            'experiment_id': experiment_result['experiment_id'],
            'timestamp': experiment_result['timestamp'],
            'task_name': experiment_result['task_name'],
            'model_id': experiment_result['model_id'],
            'metrics': experiment_result['metrics']
        }
        
        log_file = self.results_dir / "experiments_log.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        
        print(f"  Results saved to {results_file}")
        print(f"  Metrics: {experiment_result['metrics']}")

