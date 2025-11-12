"""AWS Bedrock client for running LLM inference."""
import json
import time
import boto3
import requests
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
from .config import Config


class BedrockClient:
    """Client for interacting with AWS Bedrock."""
    
    def __init__(self, config: Config):
        """Initialize Bedrock client with configuration."""
        self.config = config
        self.bearer_token = config.get_bearer_token()
        
        # Set bearer token as environment variable so boto3 can use it
        if self.bearer_token:
            import os
            os.environ['AWS_BEARER_TOKEN_BEDROCK'] = self.bearer_token
        
        # Always use boto3 - it will automatically use the bearer token from environment
        # if available, otherwise fall back to AWS credentials
        client_kwargs = {'region_name': config.aws_region}
        
        # Only set AWS credentials if bearer token is not available
        if not self.bearer_token:
            aws_key = config.get_aws_access_key()
            aws_secret = config.get_aws_secret_key()
            
            if aws_key and aws_secret:
                client_kwargs['aws_access_key_id'] = aws_key
                client_kwargs['aws_secret_access_key'] = aws_secret
        
        self.bedrock_runtime = boto3.client('bedrock-runtime', **client_kwargs)
        self.use_bearer_token = False  # We use boto3, which handles bearer token via env var
    
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
        
        # Detect model provider and prepare appropriate request body
        if model_id.startswith('openai.'):
            # OpenAI format
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            if system_instructions:
                # Add system message as first message
                body["messages"].insert(0, {
                    "role": "system",
                    "content": system_instructions
                })
        else:
            # Anthropic format (default)
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
                # Always use boto3 - it automatically uses bearer token from AWS_BEARER_TOKEN_BEDROCK env var
                # if available, otherwise uses AWS credentials
                
                # Try invoke_model first (for older models)
                try:
                    response = self.bedrock_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps(body)
                    )
                    response_body = json.loads(response['body'].read())
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_message = e.response.get('Error', {}).get('Message', '')
                    
                    # If invoke_model fails with ValidationException about on-demand throughput,
                    # try using converse API instead (for newer models)
                    if error_code == 'ValidationException' and 'on-demand throughput' in error_message:
                        # Convert to converse API format
                        converse_messages = []
                        for msg in body.get('messages', []):
                            converse_messages.append({
                                'role': msg.get('role', 'user'),
                                'content': [{'text': msg.get('content', '')}]
                            })
                        
                        converse_kwargs = {
                            'modelId': model_id,
                            'messages': converse_messages
                        }
                        
                        if body.get('system'):
                            converse_kwargs['system'] = [{'text': body['system']}]
                        
                        if 'max_tokens' in body:
                            converse_kwargs['inferenceConfig'] = {
                                'maxTokens': body['max_tokens'],
                                'temperature': body.get('temperature', 0.0)
                            }
                        
                        response = self.bedrock_runtime.converse(**converse_kwargs)
                        # Converse API returns response directly, not wrapped in 'body'
                        response_body = response
                    else:
                        raise e
                
                # Extract content based on model response format
                content = ''
                
                # Try converse API format first (output.message.content array)
                if 'output' in response_body:
                    output = response_body.get('output', {})
                    if 'message' in output:
                        message = output['message']
                        if 'content' in message:
                            content_list = message.get('content', [])
                            if content_list and isinstance(content_list, list):
                                content = content_list[0].get('text', '')
                
                # Try Anthropic invoke_model format (content array with text)
                if not content and 'content' in response_body:
                    content_list = response_body.get('content', [])
                    if content_list and isinstance(content_list, list):
                        content = content_list[0].get('text', '')
                
                # Try OpenAI format (choices array)
                if not content and 'choices' in response_body:
                    choices = response_body.get('choices', [])
                    if choices and isinstance(choices, list):
                        choice = choices[0]
                        if 'message' in choice:
                            content = choice['message'].get('content', '')
                        elif 'text' in choice:
                            content = choice.get('text', '')
                
                # Fallback: try direct text field
                if not content:
                    content = response_body.get('text', response_body.get('output', ''))
                
                return {
                    'success': True,
                    'content': content,
                    'model_id': model_id,
                    'usage': response_body.get('usage', {}),
                    'attempt': attempt + 1,
                    'raw_response': response_body  # Include for debugging
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
            
            except requests.exceptions.HTTPError as e:
                # Handle HTTP errors from bearer token requests
                error_code = None
                error_message = str(e)
                try:
                    error_body = e.response.json()
                    error_code = error_body.get('__type', '')
                    error_message = error_body.get('message', error_body.get('error', str(e)))
                    # Include full error details for debugging
                    print(f"    [ERROR] HTTP {e.response.status_code}: {error_message}")
                    if error_body:
                        print(f"    [ERROR] Full error response: {error_body}")
                except:
                    pass
                
                if 'Throttling' in str(e) or '429' in str(e.response.status_code):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                
                return {
                    'success': False,
                    'error': error_message,
                    'error_code': error_code or f'HTTP_{e.response.status_code}',
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

