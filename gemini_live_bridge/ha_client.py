"""Home Assistant REST API client."""

import logging
from typing import Optional, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)


class HomeAssistantClient:
    """Client for Home Assistant REST API."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip('/')
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def call_service(self, domain: str, service: str, **kwargs) -> Dict[str, Any]:
        """Call a Home Assistant service."""
        url = f"{self.url}/api/services/{domain}/{service}"
        logger.info(f"Calling service: {domain}.{service} with data: {kwargs}")

        try:
            async with self.session.post(url, json=kwargs, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Service call successful: {result}")
                    return {"success": True, "result": result}
                else:
                    error_text = await response.text()
                    logger.error(f"Service call failed: {response.status} - {error_text}")
                    return {"success": False, "error": error_text}
        except Exception as e:
            logger.error(f"Service call exception: {e}")
            return {"success": False, "error": str(e)}

    async def get_states(self) -> list:
        """Get all entity states."""
        url = f"{self.url}/api/states"

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get states: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Get states exception: {e}")
            return []

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """Get the state of an entity."""
        url = f"{self.url}/api/states/{entity_id}"

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get state for {entity_id}: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Get state exception: {e}")
            return {}
