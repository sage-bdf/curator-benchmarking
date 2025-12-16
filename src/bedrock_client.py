"""AWS Bedrock client for running LLM inference."""
import json
import time
import boto3
import requests
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError
from .config import Config
from .tool import Tool
from .tool_executor import ToolExecutor


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
    
    def _convert_tools_to_bedrock_format(self, tools: List[Tool], model_id: str) -> List[Dict[str, Any]]:
        """
        Convert Tool objects to Bedrock API format.
        
        Args:
            tools: List of Tool objects
            model_id: Model ID to determine format (Anthropic vs Converse API vs OpenAI)
            
        Returns:
            List of tool definitions in appropriate format
        """
        is_openai = model_id.startswith('openai.')
        use_converse_api = (
            model_id.startswith('us.amazon.') or model_id.startswith('amazon.') or
            model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.') or
            model_id.startswith('us.meta.') or model_id.startswith('meta.')
        )
        
        if is_openai:
            # OpenAI format - uses "functions" instead of "tools"
            result = []
            for tool in tools:
                schema = tool.get_schema()
                # OpenAI function format
                result.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema  # OpenAI uses "parameters" instead of "input_schema"
                })
            return result
        elif use_converse_api:
            # Converse API format
            return [tool.to_bedrock_format() for tool in tools]
        else:
            # Anthropic format
            result = []
            for tool in tools:
                schema = tool.get_schema()
                result.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": schema
                })
            return result
    
    def _extract_tool_calls_from_response(self, response_body: Dict[str, Any], model_id: str = None) -> List[Dict[str, Any]]:
        """Extract tool calls from a Bedrock response."""
        tool_calls = []
        is_openai = model_id and model_id.startswith('openai.')
        
        if is_openai:
            # OpenAI format - check for function_call in choices
            if 'choices' in response_body:
                for choice in response_body.get('choices', []):
                    message = choice.get('message', {})
                    if 'function_call' in message:
                        func_call = message['function_call']
                        # Parse arguments if it's a string
                        arguments = func_call.get('arguments', '{}')
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except:
                                arguments = {}
                        tool_calls.append({
                            'toolUseId': func_call.get('id') or f"call_{len(tool_calls)}",
                            'name': func_call.get('name'),
                            'input': arguments
                        })
        else:
            # Check Converse API format (output.message.content)
            if 'output' in response_body:
                output = response_body.get('output', {})
                if 'message' in output:
                    message = output.get('message', {})
                    content = message.get('content', [])
                    for item in content:
                        if isinstance(item, dict) and item.get('toolUse'):
                            tool_calls.append(item['toolUse'])
            
            # Check Anthropic format (content array)
            if 'content' in response_body:
                for item in response_body.get('content', []):
                    if isinstance(item, dict) and item.get('type') == 'tool_use':
                        tool_calls.append({
                            'toolUseId': item.get('id'),
                            'name': item.get('name'),
                            'input': item.get('input', {})
                        })
        
        return tool_calls
    
    def _invoke_model_with_tools(
        self,
        model_id: str,
        prompt: str,
        system_instructions: str,
        temperature: float,
        thinking: bool,
        max_tokens: int,
        max_retries: int,
        tools: List[Tool],
        tool_executor: ToolExecutor
    ) -> Dict[str, Any]:
        """
        Invoke model with tools, handling tool use flow.

        This method handles the multi-turn conversation where the model may request
        tool calls, we execute them, and continue the conversation.
        """
        print(f"    [DEBUG] _invoke_model_with_tools called with {len(tools)} tool(s)")
        # Convert tools to Bedrock format
        bedrock_tools = self._convert_tools_to_bedrock_format(tools, model_id)
        print(f"    [DEBUG] Converted to {len(bedrock_tools)} Bedrock format tool(s)")
        
        # Build initial messages
        messages = []
        
        # Determine model type and format
        is_openai = model_id.startswith('openai.')
        use_converse_api = (
            model_id.startswith('us.amazon.') or model_id.startswith('amazon.') or
            model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.') or
            model_id.startswith('us.meta.') or model_id.startswith('meta.')
        )
        
        if is_openai:
            # OpenAI format - build messages with system instruction
            messages = []
            if system_instructions:
                messages.append({
                    "role": "system",
                    "content": system_instructions
                })
            messages.append({
                "role": "user",
                "content": prompt
            })
        elif use_converse_api:
            # Converse API format
            messages.append({
                "role": "user",
                "content": [{"text": prompt}]
            })
        else:
            # Anthropic format
            messages.append({
                "role": "user",
                "content": prompt
            })
        
        # Maximum tool use iterations to prevent infinite loops
        max_tool_iterations = 10
        tool_iterations = 0
        all_tool_calls = []
        
        for attempt in range(max_retries):
            try:
                while tool_iterations < max_tool_iterations:
                    # Build API call parameters
                    if is_openai:
                        # OpenAI format
                        body = {
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "functions": bedrock_tools  # OpenAI uses "functions" instead of "tools"
                        }
                        
                        response = self.bedrock_runtime.invoke_model(
                            modelId=model_id,
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                    elif use_converse_api:
                        converse_kwargs = {
                            'modelId': model_id,
                            'messages': messages,
                            'inferenceConfig': {
                                'maxTokens': max_tokens,
                                'temperature': temperature
                            },
                            'toolConfig': {
                                'tools': bedrock_tools
                            }
                        }
                        if system_instructions:
                            converse_kwargs['system'] = [{"text": system_instructions}]
                        
                        response = self.bedrock_runtime.converse(**converse_kwargs)
                        response_body = response
                    else:
                        # Anthropic format
                        body = {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": max_tokens,
                            "messages": messages,
                            "tools": bedrock_tools
                        }
                        # Debug: Print tool info
                        print(f"    [DEBUG] Passing {len(bedrock_tools)} tool(s) to Bedrock API")
                        if bedrock_tools:
                            print(f"    [DEBUG] Tool names: {[t.get('name') for t in bedrock_tools]}")
                        if system_instructions:
                            body["system"] = system_instructions
                        if thinking:
                            thinking_budget = min(4096, max_tokens - 100)
                            if thinking_budget < 1024:
                                thinking_budget = 1024
                            body["thinking"] = {
                                "type": "enabled",
                                "budget_tokens": thinking_budget
                            }
                        else:
                            body["temperature"] = temperature
                        
                        response = self.bedrock_runtime.invoke_model(
                            modelId=model_id,
                            body=json.dumps(body)
                        )
                        response_body = json.loads(response['body'].read())
                    
                    # Check for tool calls
                    tool_calls = self._extract_tool_calls_from_response(response_body, model_id)
                    
                    if not tool_calls:
                        # No tool calls - we have the final response
                        break
                    
                    # Execute tool calls
                    tool_results = tool_executor.execute_tool_calls(tool_calls)
                    all_tool_calls.extend(tool_calls)
                    
                    # Add assistant message with tool use
                    if is_openai:
                        # OpenAI format - add assistant message with function_call
                        assistant_message = {
                            "role": "assistant",
                            "content": None  # OpenAI may return null content when calling functions
                        }
                        # Add function_call if present in response
                        if 'choices' in response_body and len(response_body['choices']) > 0:
                            choice = response_body['choices'][0]
                            if 'message' in choice and 'function_call' in choice['message']:
                                assistant_message['function_call'] = choice['message']['function_call']
                        messages.append(assistant_message)
                        
                        # Add tool results as function message
                        for result in tool_results:
                            tool_use_id = result.get('toolUseId')
                            tool_name = None
                            # Find the tool name from the tool call
                            for tool_call in tool_calls:
                                if tool_call.get('toolUseId') == tool_use_id:
                                    tool_name = tool_call.get('name')
                                    break
                            
                            # Extract result content
                            result_content = ""
                            for content_item in result.get('content', []):
                                if isinstance(content_item, dict):
                                    result_content += content_item.get('text', '')
                                elif isinstance(content_item, str):
                                    result_content += content_item
                            
                            messages.append({
                                "role": "function",
                                "name": tool_name,
                                "content": result_content
                            })
                    elif use_converse_api:
                        assistant_content = []
                        for tool_call in tool_calls:
                            assistant_content.append({"toolUse": tool_call})
                        messages.append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        
                        # Add tool results
                        tool_result_content = []
                        for result in tool_results:
                            tool_result_content.append({"toolResult": result})
                        messages.append({
                            "role": "user",
                            "content": tool_result_content
                        })
                    else:
                        # Anthropic format
                        assistant_content = []
                        for tool_call in tool_calls:
                            assistant_content.append({
                                "type": "tool_use",
                                "id": tool_call.get('toolUseId'),
                                "name": tool_call.get('name'),
                                "input": tool_call.get('input', {})
                            })
                        messages.append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        
                        # Add tool results
                        tool_result_content = []
                        for result in tool_results:
                            # Convert content to proper format - each item needs a type field
                            content_items = []
                            for content_item in result.get('content', []):
                                if isinstance(content_item, dict):
                                    # If already has type, use as is
                                    if 'type' in content_item:
                                        content_items.append(content_item)
                                    else:
                                        # Otherwise, wrap text in proper format
                                        content_items.append({
                                            "type": "text",
                                            "text": content_item.get('text', str(content_item))
                                        })
                                elif isinstance(content_item, str):
                                    content_items.append({
                                        "type": "text",
                                        "text": content_item
                                    })
                            
                            tool_result_content.append({
                                "type": "tool_result",
                                "tool_use_id": result.get('toolUseId'),
                                "content": content_items
                            })
                        messages.append({
                            "role": "user",
                            "content": tool_result_content
                        })
                    
                    tool_iterations += 1
                
                # Extract final content
                content = ''
                if is_openai:
                    # OpenAI format
                    if 'choices' in response_body and len(response_body['choices']) > 0:
                        choice = response_body['choices'][0]
                        message = choice.get('message', {})
                        # Check if there's content (not a function call)
                        if 'content' in message and message['content']:
                            content = message['content']
                        elif 'function_call' not in message:
                            # No function call and no content - might be empty
                            content = message.get('content', '')
                elif use_converse_api:
                    if 'output' in response_body:
                        output = response_body.get('output', {})
                        if 'message' in output:
                            message = output.get('message', {})
                            content_list = message.get('content', [])
                            for item in content_list:
                                if isinstance(item, dict) and item.get('text'):
                                    content += item['text']
                else:
                    # Anthropic format
                    content_list = response_body.get('content', [])
                    for item in content_list:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            content += item.get('text', '')
                
                return {
                    'success': True,
                    'content': content,
                    'model_id': model_id,
                    'usage': response_body.get('usage', {}),
                    'attempt': attempt + 1,
                    'raw_response': response_body,
                    'tool_calls': all_tool_calls,
                    'tool_execution_history': tool_executor.get_execution_history()
                }
            
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'error_code': error_code,
                        'model_id': model_id,
                        'attempt': attempt + 1,
                        'tool_calls': all_tool_calls
                    }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'model_id': model_id,
                    'attempt': attempt + 1,
                    'tool_calls': all_tool_calls
                }
        
        return {
            'success': False,
            'error': 'Max retries exceeded',
            'model_id': model_id,
            'tool_calls': all_tool_calls
        }
    
    def invoke_model(
        self,
        model_id: str,
        prompt: str,
        system_instructions: Optional[str] = None,
        temperature: float = 0.0,
        thinking: bool = False,
        max_tokens: int = 4096,
        max_retries: int = 3,
        tools: Optional[List[Tool]] = None,
        tool_executor: Optional[ToolExecutor] = None
    ) -> Dict[str, Any]:
        """
        Invoke a Bedrock model with the given parameters.
        
        Args:
            model_id: The model endpoint identifier
            prompt: The user prompt
            system_instructions: Optional system instructions (uses default if None)
            temperature: Sampling temperature
            thinking: Enable thinking mode
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            tools: Optional list of Tool objects to make available to the model
            tool_executor: Optional ToolExecutor for handling tool calls (required if tools provided)
            
        Returns:
            Dictionary containing the response and metadata, including tool usage information
        """
        if system_instructions is None:
            system_instructions = self.config.default_system_instructions

        # Debug: Check tool status
        print(f"    [DEBUG] invoke_model called: tools={len(tools) if tools else 0}, tool_executor={tool_executor is not None}")

        # If tools are provided, use tool-aware invocation
        if tools and tool_executor:
            print(f"    [DEBUG] Using tool-aware invocation")
            return self._invoke_model_with_tools(
                model_id=model_id,
                prompt=prompt,
                system_instructions=system_instructions,
                temperature=temperature,
                thinking=thinking,
                max_tokens=max_tokens,
                max_retries=max_retries,
                tools=tools,
                tool_executor=tool_executor
            )
        else:
            print(f"    [DEBUG] Using standard invocation (no tools)")
        
        # Otherwise, use standard invocation (existing code)
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
        elif model_id.startswith('us.meta.') or model_id.startswith('meta.'):
            # Meta Llama format - use converse API format
            # Meta Llama models support system instructions via the system parameter in Converse API
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
            # Add system instructions if provided (Meta Llama supports system role in Converse API)
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
                
                # For Amazon Nova, DeepSeek, and Meta models, use converse API directly
                if model_id.startswith('us.amazon.') or model_id.startswith('amazon.') or model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.') or model_id.startswith('us.meta.') or model_id.startswith('meta.'):
                    try:
                        # Debug: print the request structure
                        if model_id.startswith('us.amazon.') or model_id.startswith('amazon.'):
                            model_type = "Nova"
                        elif model_id.startswith('us.deepseek.') or model_id.startswith('deepseek.'):
                            model_type = "DeepSeek"
                        elif model_id.startswith('us.meta.') or model_id.startswith('meta.'):
                            model_type = "Meta"
                        else:
                            model_type = "Unknown"
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

