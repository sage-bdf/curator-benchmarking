"""Tool management for LLM experiments."""
import json
from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
from pathlib import Path


class Tool(ABC):
    """
    Base class for tools that can be used by LLMs in experiments.
    
    Tools are resources that the LLM can leverage to improve performance.
    Each tool must define its schema (OpenAPI format) and implement execution logic.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a tool.
        
        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the OpenAPI schema for this tool.
        
        Returns:
            Dictionary containing the tool schema in OpenAPI format
        """
        pass
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given parameters.
        
        Args:
            parameters: Dictionary of parameters from the LLM's tool call
            
        Returns:
            The result of tool execution (will be serialized to JSON)
        """
        pass
    
    def to_bedrock_format(self) -> Dict[str, Any]:
        """
        Convert tool to Bedrock API format.
        
        Returns:
            Tool definition in Bedrock Converse API format
        """
        schema = self.get_schema()
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": schema
            }
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tools
        """
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """
        Get names of all registered tools.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def load_from_config(self, config_path: Path) -> List[Tool]:
        """
        Load tools from a configuration file.
        
        Args:
            config_path: Path to tool configuration file (JSON or YAML)
            
        Returns:
            List of loaded Tool instances
        """
        if not config_path.exists():
            return []
        
        tools = []
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Config format: list of tool definitions
        tool_defs = config.get('tools', [])
        for tool_def in tool_defs:
            tool = self._create_tool_from_def(tool_def)
            if tool:
                tools.append(tool)
                self.register(tool)
        
        return tools
    
    def discover_all_tools(self, tools_dir: Path) -> Dict[str, Dict[str, Any]]:
        """
        Discover all available tools from config files in a directory.
        
        Args:
            tools_dir: Directory containing tool configuration files
            
        Returns:
            Dictionary mapping tool names to their definitions and source files
        """
        tools_map = {}
        
        if not tools_dir.exists():
            return tools_map
        
        # Scan for JSON and YAML config files
        for config_file in tools_dir.glob('*.json'):
            if config_file.name in ['example_tools.json', 'ols_tools.json', 'suggested_tools.json']:
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    tool_defs = config.get('tools', [])
                    for tool_def in tool_defs:
                        tool_name = tool_def.get('name')
                        if tool_name:
                            tools_map[tool_name] = {
                                'definition': tool_def,
                                'source_file': str(config_file)
                            }
                except Exception as e:
                    print(f"Warning: Could not read {config_file}: {e}")
        
        return tools_map
    
    def load_tools_by_names(self, tool_names: List[str], tools_dir: Path) -> List[Tool]:
        """
        Load specific tools by name from available tool configurations.
        
        Args:
            tool_names: List of tool names to load
            tools_dir: Directory containing tool configuration files
            
        Returns:
            List of loaded Tool instances
        """
        if not tool_names:
            return []
        
        # Discover all available tools
        all_tools = self.discover_all_tools(tools_dir)
        
        tools = []
        for tool_name in tool_names:
            if tool_name in all_tools:
                tool_def = all_tools[tool_name]['definition']
                tool = self._create_tool_from_def(tool_def)
                if tool:
                    tools.append(tool)
                    self.register(tool)
            else:
                print(f"Warning: Tool '{tool_name}' not found in available tools")
        
        return tools
    
    def _create_tool_from_def(self, tool_def: Dict[str, Any]) -> Optional[Tool]:
        """
        Create a Tool instance from a definition dictionary.
        
        Args:
            tool_def: Tool definition dictionary
            
        Returns:
            Tool instance or None if creation fails
        """
        tool_type = tool_def.get('type', 'function')
        name = tool_def.get('name')
        description = tool_def.get('description', '')
        
        if not name:
            return None
        
        if tool_type == 'function':
            # Function-based tool - requires a Python function
            func_path = tool_def.get('function_path')
            if func_path:
                # Load function from file
                return self._load_function_tool(name, description, func_path, tool_def)
            else:
                # Inline function definition (for simple tools)
                return self._create_inline_function_tool(name, description, tool_def)
        elif tool_type == 'api':
            # API-based tool
            return self._create_api_tool(name, description, tool_def)
        else:
            print(f"Warning: Unknown tool type '{tool_type}' for tool '{name}'")
            return None
    
    def _load_function_tool(self, name: str, description: str, func_path: str, tool_def: Dict[str, Any]) -> Optional[Tool]:
        """Load a function-based tool from a Python file."""
        try:
            import importlib.util
            path = Path(func_path)
            if not path.is_absolute():
                # Assume relative to current working directory or config file location
                path = Path.cwd() / path
            
            spec = importlib.util.spec_from_file_location(f"tool_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            func = getattr(module, tool_def.get('function_name', 'execute'), None)
            if not func:
                print(f"Warning: Function not found in {func_path}")
                return None
            
            schema = tool_def.get('schema', {})
            return FunctionTool(name, description, schema, func)
        except Exception as e:
            print(f"Warning: Could not load function tool '{name}' from {func_path}: {e}")
            return None
    
    def _create_inline_function_tool(self, name: str, description: str, tool_def: Dict[str, Any]) -> Optional[Tool]:
        """Create a function tool from inline definition."""
        schema = tool_def.get('schema', {})
        # For inline tools, we'd need the function code - this is a placeholder
        # In practice, inline tools should use function_path
        print(f"Warning: Inline function tools not yet supported for '{name}'")
        return None
    
    def _create_api_tool(self, name: str, description: str, tool_def: Dict[str, Any]) -> Optional[Tool]:
        """Create an API-based tool."""
        schema = tool_def.get('schema', {})
        api_url = tool_def.get('api_url')
        api_method = tool_def.get('api_method', 'GET')
        
        if not api_url:
            print(f"Warning: API tool '{name}' missing api_url")
            return None
        
        return APITool(name, description, schema, api_url, api_method)


class FunctionTool(Tool):
    """Tool that executes a Python function."""
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any], func: Callable):
        """
        Initialize a function-based tool.
        
        Args:
            name: Tool name
            description: Tool description
            schema: OpenAPI schema for the tool
            func: Python function to execute
        """
        super().__init__(name, description)
        self.schema = schema
        self.func = func
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema."""
        return self.schema
    
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the function with given parameters."""
        try:
            return self.func(**parameters)
        except Exception as e:
            return {"error": str(e)}


class APITool(Tool):
    """Tool that makes an API call."""
    
    def __init__(self, name: str, description: str, schema: Dict[str, Any], api_url: str, api_method: str = 'GET'):
        """
        Initialize an API-based tool.
        
        Args:
            name: Tool name
            description: Tool description
            schema: OpenAPI schema for the tool
            api_url: API endpoint URL
            api_method: HTTP method (GET, POST, etc.)
        """
        super().__init__(name, description)
        self.schema = schema
        self.api_url = api_url
        self.api_method = api_method
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema."""
        return self.schema
    
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """Execute the API call with given parameters."""
        try:
            import requests
            
            if self.api_method.upper() == 'GET':
                response = requests.get(self.api_url, params=parameters, timeout=10)
            elif self.api_method.upper() == 'POST':
                response = requests.post(self.api_url, json=parameters, timeout=10)
            else:
                return {"error": f"Unsupported HTTP method: {self.api_method}"}
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

