"""Provider factory."""

from typing import Dict, Any

from .base import BaseProvider


def create_provider(name: str, config: Dict[str, Any], ha_client, tool_registry, session_id: str) -> BaseProvider:
    """Create a provider instance by name."""
    if name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(config, ha_client, tool_registry, session_id)
    elif name == "openai":
        from .openai_realtime import OpenAIRealtimeProvider
        return OpenAIRealtimeProvider(config, ha_client, tool_registry, session_id)
    elif name == "hybrid":
        from .hybrid import HybridProvider
        return HybridProvider(config, ha_client, tool_registry, session_id)
    else:
        raise ValueError(f"Unknown provider: {name}. Available: gemini, openai, hybrid")
