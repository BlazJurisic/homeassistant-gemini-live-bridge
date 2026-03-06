"""Tool for controlling Home Assistant devices."""

import logging
from typing import Dict, Any

from .base import BaseTool

logger = logging.getLogger(__name__)


class ControlDeviceTool(BaseTool):

    @property
    def name(self) -> str:
        return "control_device"

    @property
    def description(self) -> str:
        return "Control a Home Assistant device (lights, switches, etc.)"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "The entity ID (e.g., light.kitchen, switch.fan)"
                },
                "action": {
                    "type": "string",
                    "enum": ["turn_on", "turn_off", "toggle", "set_rgb_color", "set_brightness"],
                    "description": "Action to perform"
                },
                "rgb_color": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "RGB color values [r, g, b] (0-255) for set_rgb_color action"
                },
                "brightness": {
                    "type": "integer",
                    "description": "Brightness value (0-255) for set_brightness action"
                }
            },
            "required": ["entity_id", "action"]
        }

    async def execute(self, args: Dict[str, Any], ha_client) -> Dict[str, Any]:
        entity_id = args.get("entity_id")
        action = args.get("action")

        if not entity_id or not action:
            return {"error": "Missing entity_id or action"}

        domain = entity_id.split('.')[0]
        service_data = {"entity_id": entity_id}

        if action == "set_rgb_color" and "rgb_color" in args:
            service_data["rgb_color"] = args["rgb_color"]
            service = "turn_on"
        elif action == "set_brightness" and "brightness" in args:
            service_data["brightness"] = args["brightness"]
            service = "turn_on"
        else:
            service = action

        return await ha_client.call_service(domain, service, **service_data)


tool = ControlDeviceTool()
