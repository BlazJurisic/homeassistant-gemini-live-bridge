"""Tool for querying Home Assistant device state."""

import logging
from typing import Dict, Any

from .base import BaseTool

logger = logging.getLogger(__name__)


class QueryDeviceStateTool(BaseTool):

    @property
    def name(self) -> str:
        return "query_device_state"

    @property
    def description(self) -> str:
        return "Get the current state of a Home Assistant device"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The entity ID to query"
                }
            },
            "required": ["entity_id"]
        }

    async def execute(self, args: Dict[str, Any], ha_client) -> Dict[str, Any]:
        entity_id = args.get("entity_id")
        if not entity_id:
            return {"error": "Missing entity_id"}

        state = await ha_client.get_state(entity_id)
        return {"state": state.get("state"), "attributes": state.get("attributes", {})}


tool = QueryDeviceStateTool()
