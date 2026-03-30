"""Base class for voice providers."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from audio import convert_32bit_stereo_to_16bit_mono

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Abstract base for all V2V / hybrid voice providers.

    Lifecycle: __init__ -> start() -> [audio streaming + tool calls] -> stop()

    The provider owns:
      - Connection to the upstream API (Gemini WS, OpenAI WS, or STT+LLM+TTS)
      - Two asyncio.Queues for audio I/O with the device layer
      - Tool execution delegation
    """

    def __init__(self, config: Dict[str, Any], ha_client, tool_registry, session_id: str):
        self.config = config
        self.ha_client = ha_client
        self.tool_registry = tool_registry
        self.session_id = session_id

        # Audio queues - DeviceConnection reads/writes these
        self.audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=500)  # provider -> device
        self.audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=100) # device -> provider

        self.active: bool = False
        self.playing: bool = False  # True while audio is being sent to speaker

    @abstractmethod
    async def start(self) -> None:
        """Connect to upstream API and run send/receive loops.
        Must set self.active = True, then run until self.active = False."""
        ...

    async def stop(self) -> None:
        """Signal the provider to shut down gracefully."""
        self.active = False

    def convert_device_audio(self, data: bytes) -> bytes:
        """Convert ESP32 audio (stereo 32-bit 16kHz) to provider's input format.
        Default: mono 16-bit 16kHz. Override if provider needs different format."""
        return convert_32bit_stereo_to_16bit_mono(data)

    def convert_provider_audio(self, data: bytes) -> bytes:
        """Convert provider output audio to ESP32 format (24kHz 16-bit mono raw).
        Default: passthrough. Override if provider outputs different format."""
        return data

    async def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name. Returns result dict."""
        tool = self.tool_registry.get(name)
        if not tool:
            logger.error(f"Unknown tool: {name}")
            return {"error": f"Unknown tool: {name}"}

        logger.info(f"Executing tool: {name}({args})")
        try:
            result = await tool.execute(args, self.ha_client)
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            result = {"error": str(e)}

        # Check if this tool ends the session
        if tool.is_session_ender:
            logger.info(f"Tool {name} is session ender, stopping provider")
            self.active = False

        return result
