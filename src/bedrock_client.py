"""AWS Bedrock client for running LLM inference."""
import json
import time
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
from .config import Config


class BedrockClient:
    """Client for interacting with AWS Bedrock."""
    
    def __init__(self, config: Config):
        """Initialize Bedrock client with configuration."""
        self.config = config
        
        # Build client kwargs - use credentials from env if available, otherwise use default chain
        client_kwargs = {'region_name': config.aws_region}
        
        aws_key = config.get_aws_access_key()
        aws_secret = config.get_aws_secret_key()
        
        if aws_key and aws_secret:
            client_kwargs['aws_access_key_id'] = aws_key
            client_kwargs['aws_secret_access_key'] = aws_secret
        
        self.bedrock_runtime = boto3.client('bedrock-runtime', **client_kwargs)
    
    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        system_instructions: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model with the given parameters.
        
        Args:
            model_id: The model endpoint identifier
            prompt: The user prompt
            system_instructions: Optional system instructions (uses default if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            
        Returns:
            Dictionary containing the response and metadata
        """
        if system_instructions is None:
            system_instructions = self.config.default_system_instructions
        
        # Prepare the request body for Claude models
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if system_instructions:
            body["system"] = system_instructions
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                response_body = json.loads(response['body'].read())
                
                return {
                    'success': True,
                    'content': response_body.get('content', [{}])[0].get('text', ''),
                    'model_id': model_id,
                    'usage': response_body.get('usage', {}),
                    'attempt': attempt + 1
                }
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'error_code': error_code,
                        'model_id': model_id,
                        'attempt': attempt + 1
                    }
            
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'model_id': model_id,
                    'attempt': attempt + 1
                }
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'model_id': model_id
        }

