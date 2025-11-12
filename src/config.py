"""Configuration management for the benchmarking framework."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Manages configuration for the benchmarking framework."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def aws_region(self) -> str:
        """Get AWS region."""
        return self._config['aws']['region']
    
    @property
    def default_model(self) -> str:
        """Get default model endpoint."""
        return self._config['aws']['default_model']
    
    @property
    def default_system_instructions(self) -> str:
        """Get default system instructions."""
        return self._config['default_system_instructions']
    
    @property
    def experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self._config['experiment']
    
    def get_aws_access_key(self) -> str:
        """Get AWS access key from environment."""
        return os.getenv('AWS_ACCESS_KEY_ID', '')
    
    def get_aws_secret_key(self) -> str:
        """Get AWS secret key from environment."""
        return os.getenv('AWS_SECRET_ACCESS_KEY', '')

