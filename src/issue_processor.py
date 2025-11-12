"""Process GitHub issues to run experiments."""
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .config import Config
from .task import Task
from .experiment import Experiment


class IssueProcessor:
    """Processes GitHub issues to extract experiment parameters and run them."""
    
    def __init__(self, tasks_dir: Path = None, config: Config = None):
        """Initialize issue processor."""
        self.config = config or Config()
        if tasks_dir is None:
            tasks_dir = Path(__file__).parent.parent / "tasks"
        self.tasks_dir = Path(tasks_dir)
    
    def parse_issue_body(self, issue_body: str) -> Dict[str, Any]:
        """
        Parse GitHub issue body to extract experiment parameters.
        
        Args:
            issue_body: The body text of the GitHub issue
            
        Returns:
            Dictionary with parsed parameters
        """
        params = {}
        
        # Task selection removed - experiments now run all tasks automatically
        
        # Extract model
        model_match = re.search(r'### Model Endpoint\s*\n\n([^\n]+)', issue_body)
        if model_match:
            model = model_match.group(1).strip()
            # Handle "Default" option from dropdown
            if model and model not in ['', '-', 'Default (global.anthropic.claude-sonnet-4-5-20250929-v1:0)']:
                # Extract model ID if it's in the format "Default (model_id)"
                if model.startswith('Default'):
                    # Use default from config
                    pass
                else:
                    params['model'] = model
        
        # Extract system instructions (may span multiple lines)
        sys_inst_match = re.search(r'### System Instructions\s*\n\n(.*?)(?=\n###|\Z)', issue_body, re.DOTALL)
        if sys_inst_match:
            sys_inst = sys_inst_match.group(1).strip()
            if sys_inst and sys_inst not in ['', '-']:
                try:
                    params['system_instructions'] = self._resolve_content(sys_inst)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    # If file not found, treat as direct content
                    params['system_instructions'] = sys_inst
        
        # Prompt is always task default, so ignore any prompt field in issue
        
        # Extract description
        desc_match = re.search(r'### Experiment Description\s*\n\n(.*?)(?=\n###|\Z)', issue_body, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip()
            if desc and desc not in ['', '-']:
                params['description'] = desc
        
        return params
    
    def _resolve_content(self, content: str) -> str:
        """
        Resolve content that may be a file reference or direct content.
        
        If content starts with "file:", it's treated as a file path.
        Otherwise, it's treated as direct content.
        """
        content = content.strip()
        
        if content.startswith('file:'):
            file_path = content[5:].strip()
            # Resolve relative to project root
            full_path = Path(__file__).parent.parent / file_path
            if full_path.exists():
                with open(full_path, 'r') as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Referenced file not found: {file_path}")
        
        return content
    
    def run_experiment_from_issue(
        self,
        issue_body: str,
        issue_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Parse an issue and run the experiment.
        
        Args:
            issue_body: The body text of the GitHub issue
            issue_number: Optional issue number for tracking
            
        Returns:
            Experiment result dictionary
        """
        params = self.parse_issue_body(issue_body)
        
        model_id = params.get('model') or self.config.default_model
        system_instructions = params.get('system_instructions')
        
        print(f"Running experiment from issue #{issue_number}" if issue_number else "Running experiment")
        print(f"  Model: {model_id}")
        if system_instructions:
            print(f"  Custom system instructions: Yes")
        print(f"  Running all tasks...")
        
        experiment = Experiment(
            tasks_dir=self.tasks_dir,
            model_id=model_id,
            system_instructions=system_instructions,
            config=self.config
        )
        
        result = experiment.run()
        
        # Add issue metadata to result
        result['issue_number'] = issue_number
        result['issue_params'] = params
        
        return result


def process_issue_file(issue_file: Path) -> Dict[str, Any]:
    """
    Process a saved issue file (for testing or manual processing).
    
    Args:
        issue_file: Path to a file containing issue body text
        
    Returns:
        Experiment result dictionary
    """
    with open(issue_file, 'r') as f:
        issue_body = f.read()
    
    processor = IssueProcessor()
    return processor.run_experiment_from_issue(issue_body)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.issue_processor <issue_body_file>")
        sys.exit(1)
    
    issue_file = Path(sys.argv[1])
    result = process_issue_file(issue_file)
    
    print("\n" + "="*60)
    print("Experiment Complete")
    print("="*60)
    print(f"Experiment ID: {result['experiment_id']}")
    print(f"Metrics: {result['metrics']}")
    print("="*60)

