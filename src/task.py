"""Task management for benchmarking."""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


class Task:
    """Represents a benchmarking task."""
    
    def __init__(self, task_dir: Path):
        """Initialize task from directory."""
        self.task_dir = Path(task_dir)
        self.name = self.task_dir.name
        
        # Load task configuration
        self.config = self._load_config()
        
        # Load input data
        self.input_data = self._load_input_data()
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Load default prompt
        self.default_prompt = self._load_default_prompt()
        
        # Load system instructions if they exist
        self.system_instructions = self._load_system_instructions()
        
        # Load schema if it exists
        self.schema = self._load_schema()
        
        # Load custom prompt formatter if it exists
        self.format_prompt_func = self._load_prompt_formatter()
        
        # Load custom system instructions formatter if it exists
        self.format_system_instructions_func = self._load_system_instructions_formatter()
        
        # Load custom scorer if it exists
        self.score_func = self._load_scorer()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load task configuration if it exists."""
        config_path = self.task_dir / "task_config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_input_data(self) -> pd.DataFrame:
        """Load input data from CSV or TSV files."""
        # Look for input data files
        input_files = list(self.task_dir.glob("input*.csv")) + \
                     list(self.task_dir.glob("input*.tsv"))
        
        if not input_files:
            # Fallback: look for any CSV/TSV that's not ground truth
            all_files = list(self.task_dir.glob("*.csv")) + \
                       list(self.task_dir.glob("*.tsv"))
            ground_truth_files = list(self.task_dir.glob("*ground*truth*")) + \
                                list(self.task_dir.glob("*Ground*Truth*"))
            input_files = [f for f in all_files if f not in ground_truth_files]
        
        if not input_files:
            raise ValueError(f"No input data found in {self.task_dir}")
        
        # Load the first input file found
        input_file = input_files[0]
        if input_file.suffix == '.tsv':
            return pd.read_csv(input_file, sep='\t')
        else:
            return pd.read_csv(input_file)
    
    def _load_ground_truth(self) -> Optional[pd.DataFrame]:
        """Load ground truth data."""
        ground_truth_files = list(self.task_dir.glob("*ground*truth*")) + \
                            list(self.task_dir.glob("*Ground*Truth*"))
        
        if not ground_truth_files:
            return None
        
        ground_truth_file = ground_truth_files[0]
        if ground_truth_file.suffix == '.tsv':
            return pd.read_csv(ground_truth_file, sep='\t')
        else:
            return pd.read_csv(ground_truth_file)
    
    def _load_default_prompt(self) -> str:
        """Load default prompt for this task."""
        prompt_path = self.task_dir / "default_prompt.txt"
        if prompt_path.exists():
            with open(prompt_path, 'r') as f:
                return f.read()
        
        # Return a generic prompt if none exists
        return "Please process the following metadata according to the task requirements."
    
    def _load_system_instructions(self) -> Optional[str]:
        """Load system instructions for this task."""
        sys_prompt_path = self.task_dir / "system_instructions.txt"
        if sys_prompt_path.exists():
            with open(sys_prompt_path, 'r') as f:
                return f.read()
        return None
    
    def get_system_instructions(self) -> Optional[str]:
        """
        Get system instructions, optionally formatted dynamically.
        
        If a custom format_system_instructions function exists, use it.
        Otherwise, return the static system instructions.
        """
        if self.format_system_instructions_func and self.system_instructions:
            try:
                print(f"DEBUG: Calling format_system_instructions for {self.name}")
                return self.format_system_instructions_func(self.system_instructions)
            except Exception as e:
                print(f"Warning: Error formatting system instructions for {self.name}: {e}")
                return self.system_instructions
        return self.system_instructions
    
    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load JSON schema if it exists."""
        schema_path = self.task_dir / "schema.json"
        if schema_path.exists():
            import json
            with open(schema_path, 'r') as f:
                return json.load(f)
        return None
    
    def get_input_samples(self) -> list:
        """Get input samples as a list of dictionaries."""
        return self.input_data.to_dict('records')
    
    def get_ground_truth_samples(self) -> Optional[list]:
        """Get ground truth samples as a list of dictionaries."""
        if self.ground_truth is None:
            return None
        return self.ground_truth.to_dict('records')
    
    def _load_prompt_formatter(self) -> Optional[callable]:
        """Load custom prompt formatter if it exists."""
        format_prompt_path = self.task_dir / "format_prompt.py"
        if format_prompt_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"{self.name}_format_prompt",
                    format_prompt_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'format_prompt'):
                    return module.format_prompt
            except Exception as e:
                print(f"Warning: Could not load format_prompt.py for {self.name}: {e}")
        return None
    
    def _load_system_instructions_formatter(self) -> Optional[callable]:
        """Load custom system instructions formatter if it exists."""
        format_prompt_path = self.task_dir / "format_prompt.py"
        if format_prompt_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"{self.name}_format_prompt",
                    format_prompt_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'format_system_instructions'):
                    print(f"DEBUG: Loaded format_system_instructions for {self.name}")
                    return module.format_system_instructions
            except Exception as e:
                print(f"Warning: Could not load format_system_instructions from format_prompt.py for {self.name}: {e}")
        else:
            print(f"DEBUG: format_prompt.py not found at {format_prompt_path}")
        return None
    
    def format_prompt(
        self,
        sample: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        schema_text: str = ""
    ) -> str:
        """
        Format the prompt for a given sample.
        
        If a custom format_prompt function exists, use it.
        Otherwise, use the default formatting.
        """
        if self.format_prompt_func:
            # Use custom formatter (schema_text is ignored for custom formatters)
            return self.format_prompt_func(
                self.default_prompt,
                sample,
                ground_truth,
                self.schema
            )
        else:
            # Default formatting: prompt + schema + input data
            sample_str = json.dumps(sample, indent=2)
            return f"{self.default_prompt}{schema_text}\n\nInput data:\n{sample_str}"
    
    def _load_scorer(self) -> Optional[callable]:
        """Load custom scorer if it exists."""
        score_path = self.task_dir / "score.py"
        if score_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"{self.name}_score",
                    score_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'score'):
                    return module.score
            except Exception as e:
                print(f"Warning: Could not load score.py for {self.name}: {e}")
        return None

