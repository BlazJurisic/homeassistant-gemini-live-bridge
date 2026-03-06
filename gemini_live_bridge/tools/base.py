"""Base class for tool plugins."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """A single tool that can be called by any voice provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g., 'control_device')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for parameters (provider-agnostic)."""
        ...

    @property
    def is_session_ender(self) -> bool:
        """If True, executing this tool should end the conversation session."""
        return False

    @abstractmethod
    async def execute(self, args: Dict[str, Any], ha_client) -> Dict[str, Any]:
        """Execute the tool and return a result dict."""
        ...
