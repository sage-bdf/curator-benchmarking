"""Configuration management for the benchmarking framework."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Manages configuration for the benchmarking framework."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "defaults.yaml"
        
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Load credentials from .aws/creds.yaml if it exists
        self._creds = self._load_creds()
    
    def _load_creds(self) -> Optional[Dict[str, Any]]:
        """Load credentials from .aws/creds.yaml if it exists."""
        creds_path = Path(__file__).parent.parent / ".aws" / "creds.yaml"
        if creds_path.exists():
            try:
                with open(creds_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception:
                return None
        return None
    
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
    
    def get_bearer_token(self) -> Optional[str]:
        """Get AWS Bedrock bearer token from .aws/creds.yaml or environment."""
        # First check environment variable
        env_token = os.getenv('AWS_BEARER_TOKEN_BEDROCK')
        if env_token:
            return env_token
        
        # Then check creds file
        if self._creds and 'AWS_BEARER_TOKEN_BEDROCK' in self._creds:
            return self._creds['AWS_BEARER_TOKEN_BEDROCK']
        
        return None

