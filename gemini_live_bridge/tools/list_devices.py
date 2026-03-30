"""Tool for listing available Home Assistant devices."""

import logging
from typing import Dict, Any

from .base import BaseTool

logger = logging.getLogger(__name__)

# Domains we care about for voice control
CONTROLLABLE_DOMAINS = {"light", "switch", "cover", "climate", "fan", "media_player", "input_boolean", "scene", "script"}


class ListDevicesTool(BaseTool):

    @property
    def name(self) -> str:
        return "list_devices"

    @property
    def description(self) -> str:
        return "List all available Home Assistant devices and their current state"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional: filter by domain (light, switch, cover, climate, fan, media_player). Leave empty for all."
                }
            },
            "required": []
        }

    async def execute(self, args: Dict[str, Any], ha_client) -> Dict[str, Any]:
        domain_filter = args.get("domain", "")
        states = await ha_client.get_states()

        if not states:
            return {"error": "Could not fetch states from Home Assistant"}

        devices = []
        for s in states:
            eid = s.get("entity_id", "")
            domain = eid.split(".")[0]

            if domain not in CONTROLLABLE_DOMAINS:
                continue
            if domain_filter and domain != domain_filter:
                continue

            devices.append({
                "entity_id": eid,
                "state": s.get("state", "unknown"),
                "name": s.get("attributes", {}).get("friendly_name", eid),
            })

        return {"devices": devices, "count": len(devices)}


tool = ListDevicesTool()
