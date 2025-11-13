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
        thinking: bool = False,
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
        elif model_id.startswith('us.amazon.') or model_id.startswith('amazon.'):
            # Amazon Nova format - use converse API format
            # Include system instructions in the user message (Nova doesn't support system role)
            full_prompt = prompt
            if system_instructions:
                full_prompt = f"{system_instructions}\n\n{prompt}"
            # Use converse API format for Nova
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": full_prompt}]
                    }
                ],
                "inferenceConfig": inference_config
            }
        elif model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.'):
            # DeepSeek format - use converse API format
            # DeepSeek models support system instructions via the system parameter in Converse API
            inference_config = {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ],
                "inferenceConfig": inference_config
            }
            # Add system instructions if provided (DeepSeek supports system role in Converse API)
            if system_instructions:
                body["system"] = [{"text": system_instructions}]
        else:
            # Anthropic format (default)
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            if system_instructions:
                body["system"] = system_instructions
            
            # Add thinking mode if enabled (per AWS docs: https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html)
            # Thinking mode requires a thinking object with type: "enabled" and budget_tokens
            # Note: temperature is not compatible with thinking mode, so we omit it when thinking is enabled
            if thinking:
                # Set a reasonable thinking budget (minimum is 1024, we'll use 4096 as a default)
                # The budget should be less than max_tokens
                thinking_budget = min(4096, max_tokens - 100)  # Leave some room for text output
                if thinking_budget < 1024:
                    thinking_budget = 1024  # Minimum required
                
                body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            else:
                # Only set temperature when thinking is NOT enabled (they're incompatible)
                body["temperature"] = temperature
        
        for attempt in range(max_retries):
            try:
                # Always use boto3 - it automatically uses bearer token from AWS_BEARER_TOKEN_BEDROCK env var
                # if available, otherwise uses AWS credentials
                
                # For Amazon Nova and DeepSeek models, use converse API directly
                if model_id.startswith('us.amazon.') or model_id.startswith('amazon.') or model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.'):
                    try:
                        # Debug: print the request structure
                        model_type = "Nova" if (model_id.startswith('us.amazon.') or model_id.startswith('amazon.')) else "DeepSeek"
                        print(f"    [DEBUG] Calling {model_type} model: {model_id}")
                        print(f"    [DEBUG] Messages: {len(body['messages'])} message(s)")
                        print(f"    [DEBUG] InferenceConfig: {body.get('inferenceConfig', {})}")
                        if 'system' in body:
                            print(f"    [DEBUG] System instructions: {len(body['system'])} system message(s)")
                        
                        # Build converse API call parameters
                        converse_kwargs = {
                            'modelId': model_id,
                            'messages': body['messages'],
                            'inferenceConfig': body.get('inferenceConfig', {})
                        }
                        # Add system instructions if present (for DeepSeek models)
                        if 'system' in body:
                            converse_kwargs['system'] = body['system']
                        
                        response = self.bedrock_runtime.converse(**converse_kwargs)
                        # Converse API returns response directly as a dict
                        response_body = response
                        
                        # Debug: print response structure
                        print(f"    [DEBUG] Response keys: {list(response_body.keys()) if isinstance(response_body, dict) else 'Not a dict'}")
                        if isinstance(response_body, dict) and 'output' in response_body:
                            print(f"    [DEBUG] Output keys: {list(response_body['output'].keys()) if isinstance(response_body.get('output'), dict) else 'Not a dict'}")
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code', '')
                        error_message = e.response.get('Error', {}).get('Message', '')
                        # Print detailed error for debugging
                        print(f"    [ERROR] Nova model invocation failed: {error_code} - {error_message}")
                        print(f"    [ERROR] Model ID: {model_id}")
                        print(f"    [ERROR] InferenceConfig: {body.get('inferenceConfig', {})}")
                        print(f"    [ERROR] Full error response: {e.response}")
                        raise e
                    except Exception as e:
                        # Catch any other exceptions
                        print(f"    [ERROR] Unexpected error calling Nova model: {type(e).__name__}: {str(e)}")
                        print(f"    [ERROR] Model ID: {model_id}")
                        raise e
                else:
                    # Use invoke_model for Anthropic models (supports thinking mode via thinking object in body)
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
                            
                            inference_config = {
                                'maxTokens': body.get('max_tokens', max_tokens)
                            }
                            
                            # Note: thinking mode and temperature are incompatible
                            # If thinking is enabled, don't set temperature
                            if not thinking:
                                inference_config['temperature'] = body.get('temperature', temperature)
                            
                            converse_kwargs['inferenceConfig'] = inference_config
                            
                            # Note: thinking mode may not be supported in converse API fallback
                            # If thinking was requested, log a warning
                            if thinking:
                                print(f"Warning: Thinking mode requested but falling back to converse API which may not support it")
                            
                            response = self.bedrock_runtime.converse(**converse_kwargs)
                            # Converse API returns response directly, not wrapped in 'body'
                            response_body = response
                        else:
                            raise e
                
                # Extract content based on model response format
                content = ''
                
                # Helper function to extract text from content array (filters out thinking blocks)
                def extract_text_from_content_array(content_list):
                    """Extract text from content array, skipping thinking blocks."""
                    if not content_list or not isinstance(content_list, list):
                        return ''
                    text_parts = []
                    for item in content_list:
                        if isinstance(item, dict):
                            # Handle standard text type blocks
                            if item.get('type') == 'text' and 'text' in item:
                                text_parts.append(item['text'])
                            # Handle DeepSeek R1 reasoningContent format
                            elif 'reasoningContent' in item:
                                reasoning_content = item.get('reasoningContent', {})
                                if 'reasoningText' in reasoning_content:
                                    reasoning_text = reasoning_content.get('reasoningText', {})
                                    if isinstance(reasoning_text, dict) and 'text' in reasoning_text:
                                        text_parts.append(reasoning_text['text'])
                                    elif isinstance(reasoning_text, str):
                                        text_parts.append(reasoning_text)
                            # Handle DeepSeek R1 textContent format (if present)
                            elif 'textContent' in item:
                                text_content = item.get('textContent', {})
                                if isinstance(text_content, dict) and 'text' in text_content:
                                    text_parts.append(text_content['text'])
                                elif isinstance(text_content, str):
                                    text_parts.append(text_content)
                            # Fallback: if item has 'text' key directly
                            elif 'text' in item and not item.get('type') == 'thinking':
                                text_parts.append(item['text'])
                    return ''.join(text_parts)
                
                # Try converse API format first (output.message.content array)
                if 'output' in response_body:
                    output = response_body.get('output', {})
                    if 'message' in output:
                        message = output['message']
                        if 'content' in message:
                            content_list = message.get('content', [])
                            content = extract_text_from_content_array(content_list)
                
                # Try Anthropic invoke_model format (content array with text and thinking blocks)
                if not content and 'content' in response_body:
                    content_list = response_body.get('content', [])
                    content = extract_text_from_content_array(content_list)
                
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
                
                # Warn if content is still empty after all parsing attempts
                if not content:
                    print(f"    [WARNING] No content extracted from response. Response structure: {list(response_body.keys()) if isinstance(response_body, dict) else type(response_body)}")
                    if isinstance(response_body, dict):
                        print(f"    [WARNING] Full response (truncated): {str(response_body)[:500]}")
                
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

