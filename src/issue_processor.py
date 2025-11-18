"""Process GitHub issues to run experiments."""
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .config import Config
from .task import Task
from .experiment import Experiment
from .tool import ToolRegistry


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
        
        # Extract model - handle both "Model" and "Model Endpoint" labels
        model_match = re.search(r'### Model(?: Endpoint)?\s*\n\n([^\n]+)', issue_body)
        if model_match:
            model = model_match.group(1).strip()
            # Skip separator lines in the dropdown
            if model.startswith('---'):
                model = None
            # Handle "Default" option from dropdown
            if model and model not in ['', '-', 'Default (global.anthropic.claude-sonnet-4-5-20250929-v1:0)', 'Default']:
                # Extract model ID if it's in the format "Default (model_id)"
                if model.startswith('Default'):
                    # Use default from config
                    pass
                elif model == 'Other':
                    model = None  # Will check custom_model below
                else:
                    params['model'] = model
        
        # Check for custom model endpoint if model not set yet
        if 'model' not in params:
            custom_model_match = re.search(r'### Custom Model Endpoint\s*\n\n([^\n]+)', issue_body)
            if custom_model_match:
                custom_model = custom_model_match.group(1).strip()
                if custom_model and custom_model not in ['', '-', '_No response_']:
                    params['model'] = custom_model
        
        # Extract system instructions (may span multiple lines)
        sys_inst_match = re.search(r'### System Instructions\s*\n\n(.*?)(?=\n###|\Z)', issue_body, re.DOTALL)
        if sys_inst_match:
            sys_inst = sys_inst_match.group(1).strip()
            # Treat "default" (case-insensitive) and "_No response_" as empty to use default instructions
            if sys_inst and sys_inst.lower() not in ['default', '_no response_'] and sys_inst not in ['', '-', '_No response_']:
                try:
                    params['system_instructions'] = self._resolve_content(sys_inst)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    # If file not found, treat as direct content
                    params['system_instructions'] = sys_inst
        
        # Extract temperature
        temp_match = re.search(r'### Temperature\s*\n\n([^\n]+)', issue_body)
        if temp_match:
            temp_str = temp_match.group(1).strip()
            # Treat "_No response_" as empty to use default temperature
            if temp_str and temp_str not in ['', '-', '_No response_']:
                try:
                    params['temperature'] = float(temp_str)
                except ValueError:
                    print(f"Warning: Invalid temperature value: {temp_str}")
        
        # Extract thinking mode
        thinking_match = re.search(r'### Thinking Mode\s*\n\n([^\n]+)', issue_body)
        if thinking_match:
            thinking_str = thinking_match.group(1).strip().lower()
            if thinking_str in ['true', 'yes', '1', 'enabled']:
                params['thinking'] = True
            elif thinking_str in ['false', 'no', '0', 'disabled', '']:
                params['thinking'] = False
        
        # Extract tools (checkboxes format: "- [x] tool_name" for selected, "- [ ] tool_name" for unselected)
        tools_match = re.search(r'### Tools\s*\n\n(.*?)(?=\n###|\Z)', issue_body, re.DOTALL)
        if tools_match:
            tools_section = tools_match.group(1)
            # Find all checked tools (format: "- [x] tool_name")
            checked_tools = re.findall(r'- \[x\]\s+(\S+)', tools_section)
            if checked_tools:
                params['tool_names'] = checked_tools
        
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
        temperature = params.get('temperature')
        thinking = params.get('thinking')
        tool_names = params.get('tool_names', [])
        
        # Load tools if specified
        tools = None
        tool_registry = None
        if tool_names:
            tool_registry = ToolRegistry()
            tools_dir = Path(__file__).parent.parent / "tools"
            tools = tool_registry.load_tools_by_names(tool_names, tools_dir)
            if tools:
                print(f"  Loaded {len(tools)} tool(s): {', '.join([t.name for t in tools])}")
            else:
                print(f"  Warning: No tools loaded from names: {tool_names}")
        
        print(f"Running experiment from issue #{issue_number}" if issue_number else "Running experiment")
        print(f"  Model: {model_id}")
        if system_instructions:
            print(f"  Custom system instructions: Yes")
        if temperature is not None:
            print(f"  Temperature: {temperature}")
        if thinking is not None:
            print(f"  Thinking mode: {thinking}")
        if tools:
            print(f"  Tools: {', '.join([t.name for t in tools])}")
        print(f"  Running all tasks...")
        
        experiment = Experiment(
            tasks_dir=self.tasks_dir,
            model_id=model_id,
            system_instructions=system_instructions,
            temperature=temperature,
            thinking=thinking,
            config=self.config,
            tools=tools,
            tool_registry=tool_registry
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

