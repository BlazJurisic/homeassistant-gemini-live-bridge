#!/usr/bin/env python3
"""
Voice Assistant Bridge for Home Assistant
Multi-provider: Gemini Live, OpenAI Realtime, Hybrid (Soniox STT + LLM + TTS)
"""

import os
import sys
import asyncio
import json
import logging

from ha_client import HomeAssistantClient
from device_connection import DeviceConnection
from tools import ToolRegistry

# Configuration from add-on options
CONFIG_PATH = "/data/options.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
else:
    config = {}

SERVER_PORT = int(config.get("server_port", 9872))
LOG_LEVEL = config.get("log_level", "INFO").upper()

# Home Assistant connection
_supervisor_token = os.environ.get("SUPERVISOR_TOKEN", "")
if not _supervisor_token:
    for token_path in [
        "/run/s6/container_environment/SUPERVISOR_TOKEN",
        "/var/run/s6/container_environment/SUPERVISOR_TOKEN",
    ]:
        if os.path.exists(token_path):
            with open(token_path) as f:
                _supervisor_token = f.read().strip()
            if _supervisor_token:
                break

if _supervisor_token:
    HA_URL = "http://supervisor/core"
    HA_TOKEN = _supervisor_token
else:
    HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
    HA_TOKEN = os.environ.get("HA_TOKEN", "")

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("websockets").setLevel(logging.WARNING)


class BridgeServer:
    """Main bridge server."""

    def __init__(self, port: int, bridge_config: dict, ha_client: HomeAssistantClient,
                 tool_registry: ToolRegistry):
        self.port = port
        self.config = bridge_config
        self.ha_client = ha_client
        self.tool_registry = tool_registry

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        connection = DeviceConnection(reader, writer, self.ha_client, self.config, self.tool_registry)
        await connection.handle()

    async def start(self):
        provider = self.config.get("provider", "gemini")

        # Validate required keys for selected provider
        if provider == "gemini" and not self.config.get("gemini_api_key"):
            logger.error("gemini_api_key required for Gemini provider")
            sys.exit(1)
        elif provider == "openai" and not self.config.get("openai_api_key"):
            logger.error("openai_api_key required for OpenAI provider")
            sys.exit(1)
        elif provider == "hybrid":
            if not self.config.get("soniox_api_key"):
                logger.error("soniox_api_key required for Hybrid provider")
                sys.exit(1)

        if not HA_TOKEN:
            logger.warning("HA_TOKEN not set, Home Assistant API calls may fail")

        server = await asyncio.start_server(self.handle_client, '0.0.0.0', self.port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Bridge listening on {addr[0]}:{addr[1]}, provider: {provider}")

        async with server:
            await server.serve_forever()


async def main():
    logger.info("=" * 60)
    logger.info("Voice Assistant Bridge for Home Assistant")
    logger.info("=" * 60)

    provider = config.get("provider", "gemini")
    logger.info(f"Provider: {provider}")
    logger.info(f"Server Port: {SERVER_PORT}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Home Assistant URL: {HA_URL}")
    logger.info(f"HA Token available: {'Yes' if HA_TOKEN else 'No'}")

    # Load tools
    tool_registry = ToolRegistry()
    tool_registry.load_builtin_tools()
    tool_names = [t.name for t in tool_registry.all_tools()]
    logger.info(f"Loaded {len(tool_names)} tools: {tool_names}")

    # Initialize HA client
    ha_client = await HomeAssistantClient(HA_URL, HA_TOKEN).__aenter__()

    logger.info("=" * 60)

    server = BridgeServer(SERVER_PORT, config, ha_client, tool_registry)
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
