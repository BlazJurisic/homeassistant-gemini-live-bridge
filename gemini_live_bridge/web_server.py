"""Web dashboard server for the Voice Assistant Bridge.

Serves the React dashboard and REST API for configuration, prompt editing,
tool management, and HA entity browsing. Runs on port 8099 for HA ingress.
"""

import asyncio
import importlib.util
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp as aiohttp_lib
from aiohttp import web

logger = logging.getLogger(__name__)

DASHBOARD_CONFIG_PATH = "/data/dashboard.json"
CUSTOM_TOOLS_DIR = "/data/custom_tools"
WEBAPP_DIR = "/webapp"
OPTIONS_PATH = "/data/options.json"


def _ensure_dirs():
    os.makedirs(CUSTOM_TOOLS_DIR, exist_ok=True)
    Path(DASHBOARD_CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)


def _read_dashboard_config() -> Dict[str, Any]:
    if os.path.exists(DASHBOARD_CONFIG_PATH):
        with open(DASHBOARD_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _write_dashboard_config(data: Dict[str, Any]):
    with open(DASHBOARD_CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_options() -> Dict[str, Any]:
    if os.path.exists(OPTIONS_PATH):
        with open(OPTIONS_PATH) as f:
            return json.load(f)
    return {}


class WebServer:
    """aiohttp web server for dashboard API and static files."""

    def __init__(self, bridge_server, ha_client, tool_registry, config: Dict[str, Any]):
        self.bridge = bridge_server
        self.ha_client = ha_client
        self.tool_registry = tool_registry
        self.config = config
        self.start_time = time.time()
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/api/status", self.handle_status)
        self.app.router.add_get("/api/config", self.handle_get_config)
        self.app.router.add_post("/api/config", self.handle_set_config)
        self.app.router.add_get("/api/prompt", self.handle_get_prompt)
        self.app.router.add_post("/api/prompt", self.handle_set_prompt)
        self.app.router.add_get("/api/tools", self.handle_list_tools)
        self.app.router.add_get("/api/tools/{name}", self.handle_get_tool)
        self.app.router.add_post("/api/tools", self.handle_create_tool)
        self.app.router.add_put("/api/tools/{name}", self.handle_update_tool)
        self.app.router.add_delete("/api/tools/{name}", self.handle_delete_tool)
        self.app.router.add_get("/api/devices", self.handle_devices)
        # Static files (React app) — must be last
        if os.path.isdir(WEBAPP_DIR):
            self.app.router.add_get("/{path:.*}", self.handle_static)
        else:
            self.app.router.add_get("/", self.handle_no_webapp)

    # ── Status ──────────────────────────────────────────────────────

    async def handle_status(self, request: web.Request) -> web.Response:
        uptime = int(time.time() - self.start_time)
        return web.json_response({
            "provider": self.config.get("provider", "gemini"),
            "uptime_seconds": uptime,
            "ha_connected": self.ha_client is not None and self.ha_client.token != "",
            "tools_loaded": len(self.tool_registry.all_tools()),
        })

    # ── Config ──────────────────────────────────────────────────────

    async def handle_get_config(self, request: web.Request) -> web.Response:
        options = _read_options()
        # Never expose full API keys
        for key in ("gemini_api_key", "openai_api_key"):
            if options.get(key):
                val = options[key]
                options[key] = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
        # Mask soniox key inside hybrid_settings
        hs = options.get("hybrid_settings", {})
        if isinstance(hs, dict) and hs.get("soniox_api_key"):
            val = hs["soniox_api_key"]
            hs["soniox_api_key"] = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
        return web.json_response(options)

    async def handle_set_config(self, request: web.Request) -> web.Response:
        body = await request.json()

        # Read current options to merge (don't lose API keys when UI sends masked values)
        current = _read_options()
        for key in ("gemini_api_key", "openai_api_key"):
            if body.get(key, "").endswith("...") or body.get(key, "") == "***":
                body[key] = current.get(key, "")
        # Handle nested soniox key
        if "soniox_api_key" in body:
            val = body.pop("soniox_api_key")
            if not val or val.endswith("...") or val == "***":
                val = current.get("hybrid_settings", {}).get("soniox_api_key", "")
            body.setdefault("hybrid_settings", current.get("hybrid_settings", {}))
            body["hybrid_settings"]["soniox_api_key"] = val

        # Merge with current options so we don't lose unset fields
        merged = {**current, **body}

        # Write to Supervisor API so it takes effect
        err = await self._set_supervisor_options(merged)
        if err:
            return web.json_response({"error": err}, status=500)

        # Also update in-memory config so next session uses new values
        self.config.update(merged)

        return web.json_response({"ok": True, "restart_required": True})

    # ── Prompt ──────────────────────────────────────────────────────

    async def handle_get_prompt(self, request: web.Request) -> web.Response:
        dashboard = _read_dashboard_config()
        prompt = dashboard.get("system_prompt", "")
        return web.json_response({"prompt": prompt})

    async def handle_set_prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        prompt = body.get("prompt", "")
        dashboard = _read_dashboard_config()
        dashboard["system_prompt"] = prompt
        _write_dashboard_config(dashboard)
        return web.json_response({"ok": True})

    # ── Tools ───────────────────────────────────────────────────────

    async def handle_list_tools(self, request: web.Request) -> web.Response:
        tools = []
        for t in self.tool_registry.all_tools():
            tools.append({
                "name": t.name,
                "description": t.description,
                "builtin": not self._is_custom_tool(t.name),
            })
        # Also list custom tool files not yet loaded
        if os.path.isdir(CUSTOM_TOOLS_DIR):
            loaded_names = {t["name"] for t in tools}
            for fname in os.listdir(CUSTOM_TOOLS_DIR):
                if fname.endswith(".py"):
                    name = fname[:-3]
                    if name not in loaded_names:
                        tools.append({"name": name, "description": "(not loaded)", "builtin": False})
        return web.json_response(tools)

    async def handle_get_tool(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        tool = self.tool_registry.get(name)

        result = {"name": name, "builtin": not self._is_custom_tool(name)}

        if tool:
            result["description"] = tool.description
            result["parameters"] = tool.parameters_schema

        # Read source code for custom tools
        custom_path = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py")
        if os.path.exists(custom_path):
            with open(custom_path) as f:
                result["code"] = f.read()
        elif tool:
            result["code"] = None  # builtin, no editable source
        else:
            return web.json_response({"error": "Tool not found"}, status=404)

        return web.json_response(result)

    async def handle_create_tool(self, request: web.Request) -> web.Response:
        body = await request.json()
        name = body.get("name", "").strip()
        code = body.get("code", "")

        if not name:
            return web.json_response({"error": "name is required"}, status=400)
        if not name.replace("_", "").isalnum():
            return web.json_response({"error": "name must be alphanumeric/underscores"}, status=400)

        # Don't overwrite builtins
        existing = self.tool_registry.get(name)
        if existing and not self._is_custom_tool(name):
            return web.json_response({"error": "Cannot overwrite builtin tool"}, status=400)

        _ensure_dirs()
        path = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(code)

        # Try to load it
        err = self._load_custom_tool(name, path)
        if err:
            return web.json_response({"error": err}, status=400)

        return web.json_response({"ok": True, "name": name})

    async def handle_update_tool(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not self._is_custom_tool(name):
            return web.json_response({"error": "Cannot edit builtin tool"}, status=400)

        body = await request.json()
        code = body.get("code", "")
        path = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py")

        if not os.path.exists(path):
            return web.json_response({"error": "Tool not found"}, status=404)

        with open(path, "w") as f:
            f.write(code)

        err = self._load_custom_tool(name, path)
        if err:
            return web.json_response({"error": err}, status=400)

        return web.json_response({"ok": True})

    async def handle_delete_tool(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not self._is_custom_tool(name):
            return web.json_response({"error": "Cannot delete builtin tool"}, status=400)

        path = os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py")
        if os.path.exists(path):
            os.remove(path)

        # Remove from registry
        if name in self.tool_registry._tools:
            del self.tool_registry._tools[name]

        return web.json_response({"ok": True})

    # ── Devices ─────────────────────────────────────────────────────

    async def handle_devices(self, request: web.Request) -> web.Response:
        states = await self.ha_client.get_states()
        devices = []
        for s in states:
            eid = s.get("entity_id", "")
            devices.append({
                "entity_id": eid,
                "domain": eid.split(".")[0],
                "state": s.get("state", "unknown"),
                "friendly_name": s.get("attributes", {}).get("friendly_name", eid),
                "attributes": s.get("attributes", {}),
            })
        devices.sort(key=lambda d: d["entity_id"])
        return web.json_response(devices)

    # ── Static files ────────────────────────────────────────────────

    async def handle_static(self, request: web.Request) -> web.Response:
        path = request.match_info.get("path", "")
        if not path:
            path = "index.html"

        file_path = os.path.join(WEBAPP_DIR, path)
        if os.path.isfile(file_path):
            return web.FileResponse(file_path)

        # SPA fallback — serve index.html for all routes
        index_path = os.path.join(WEBAPP_DIR, "index.html")
        if os.path.isfile(index_path):
            return web.FileResponse(index_path)

        return web.json_response({"error": "not found"}, status=404)

    async def handle_no_webapp(self, request: web.Request) -> web.Response:
        return web.Response(
            text="<h2>Voice Assistant Bridge</h2><p>Web dashboard not built. Run <code>cd webapp && npm run build</code>.</p>",
            content_type="text/html",
        )

    # ── Supervisor API ─────────────────────────────────────────────

    async def _set_supervisor_options(self, options: Dict[str, Any]) -> Optional[str]:
        """Write options to Supervisor API. Returns error string or None."""
        token = self._get_supervisor_token()
        if not token:
            # Fallback: write directly to options.json (dev mode)
            try:
                with open(OPTIONS_PATH, "w") as f:
                    json.dump(options, f, indent=2, ensure_ascii=False)
                return None
            except Exception as e:
                return str(e)

        slug = "350c5b11_gemini_live_bridge"
        url = f"http://supervisor/addons/{slug}/options"
        try:
            async with aiohttp_lib.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={"options": options},
                    timeout=aiohttp_lib.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    if data.get("result") == "ok":
                        logger.info("Supervisor options updated")
                        return None
                    else:
                        err = data.get("message", str(data))
                        logger.error(f"Supervisor options error: {err}")
                        return err
        except Exception as e:
            logger.error(f"Failed to set supervisor options: {e}")
            return str(e)

    def _get_supervisor_token(self) -> Optional[str]:
        token = os.environ.get("SUPERVISOR_TOKEN", "")
        if token:
            return token
        for path in [
            "/run/s6/container_environment/SUPERVISOR_TOKEN",
            "/var/run/s6/container_environment/SUPERVISOR_TOKEN",
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    token = f.read().strip()
                if token:
                    return token
        return None

    # ── Helpers ──────────────────────────────────────────────────────

    def _is_custom_tool(self, name: str) -> bool:
        return os.path.exists(os.path.join(CUSTOM_TOOLS_DIR, f"{name}.py"))

    def _load_custom_tool(self, name: str, path: str) -> Optional[str]:
        """Load a custom tool from file. Returns error string or None on success."""
        try:
            spec = importlib.util.spec_from_file_location(f"custom_tool_{name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "tool"):
                self.tool_registry.register(mod.tool)
            elif hasattr(mod, "CustomTool"):
                instance = mod.CustomTool()
                self.tool_registry.register(instance)
            else:
                return "Tool file must define 'tool' instance or 'CustomTool' class"
            logger.info(f"Loaded custom tool: {name}")
            return None
        except Exception as e:
            logger.error(f"Failed to load custom tool {name}: {e}")
            return str(e)

    def load_all_custom_tools(self):
        """Load all custom tools from /data/custom_tools/."""
        _ensure_dirs()
        if not os.path.isdir(CUSTOM_TOOLS_DIR):
            return
        for fname in sorted(os.listdir(CUSTOM_TOOLS_DIR)):
            if fname.endswith(".py"):
                name = fname[:-3]
                path = os.path.join(CUSTOM_TOOLS_DIR, fname)
                err = self._load_custom_tool(name, path)
                if err:
                    logger.warning(f"Skipping custom tool {name}: {err}")


async def start_web_server(bridge_server, ha_client, tool_registry, config: Dict[str, Any], port: int = 8099):
    """Start the web server. Call from the main asyncio loop."""
    server = WebServer(bridge_server, ha_client, tool_registry, config)
    server.load_all_custom_tools()
    runner = web.AppRunner(server.app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"Web dashboard listening on 0.0.0.0:{port}")
    return server
