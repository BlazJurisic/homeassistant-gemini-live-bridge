"""Tool for ending the conversation session."""

from typing import Dict, Any

from .base import BaseTool


class EndConversationTool(BaseTool):

    @property
    def name(self) -> str:
        return "end_conversation"

    @property
    def description(self) -> str:
        return ("End the conversation session and return to wake word mode. "
                "Call this when user says goodbye, thank you and indicates "
                "they're done, or after prolonged silence.")

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def is_session_ender(self) -> bool:
        return True

    async def execute(self, args: Dict[str, Any], ha_client) -> Dict[str, Any]:
        return {"success": True, "message": "Conversation ended"}


tool = EndConversationTool()
