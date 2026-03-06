"""Tool registry with auto-discovery and provider-specific format conversion."""

import importlib
import logging
import pkgutil
from typing import Dict, List, Any, Optional

from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Loads and manages tool plugins."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def load_builtin_tools(self) -> None:
        """Auto-discover and load all tool modules in this package."""
        package_path = __path__
        for importer, modname, ispkg in pkgutil.iter_modules(package_path):
            if modname in ("__init__", "base"):
                continue
            try:
                mod = importlib.import_module(f"tools.{modname}")
                if hasattr(mod, "tool"):
                    self.register(mod.tool)
                    logger.info(f"Loaded tool: {mod.tool.name}")
                elif hasattr(mod, "Tool"):
                    instance = mod.Tool()
                    self.register(instance)
                    logger.info(f"Loaded tool: {instance.name}")
            except Exception as e:
                logger.error(f"Failed to load tool module '{modname}': {e}")

    def to_gemini_tools(self):
        """Convert all tools to Gemini FunctionDeclaration format."""
        from google.genai import types

        declarations = []
        for tool in self._tools.values():
            declarations.append(types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters_schema,
            ))
        return [types.Tool(function_declarations=declarations)]

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI Realtime function format."""
        result = []
        for tool in self._tools.values():
            result.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            })
        return result
