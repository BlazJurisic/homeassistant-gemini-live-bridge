#!/usr/bin/env python3
"""
Gemini Live Bridge for Home Assistant
Full duplex voice assistant with wake word activation
"""

import os
import sys
import asyncio
import json
import logging
import struct
import socket
import array
from typing import Optional, Dict, Any

import numpy as np
from google import genai
from google.genai import types
import aiohttp

# Configuration from add-on options
CONFIG_PATH = "/data/options.json"

# Load configuration
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
else:
    config = {}

GEMINI_API_KEY = config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
SERVER_PORT = int(config.get("server_port", 9872))
LOG_LEVEL = config.get("log_level", "INFO").upper()
SESSION_TIMEOUT = int(config.get("session_timeout_seconds", 300))
CROATIAN_PERSONALITY = config.get("croatian_personality", True)

# Home Assistant connection
# In HA add-on context, SUPERVISOR_TOKEN may be env var or file
_supervisor_token = os.environ.get("SUPERVISOR_TOKEN", "")
if not _supervisor_token:
    # Try reading from s6 container environment
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

# Audio configuration
SEND_SAMPLE_RATE = 16000  # From ESP32 device
RECEIVE_SAMPLE_RATE = 48000  # To ESP32 device (Gemini sends 24kHz, we upsample to 48kHz)
CHANNELS = 1
CHUNK_SIZE = 1024
DEVICE_BITS_PER_SAMPLE = 32  # ESP32 I2S sends 32-bit samples


def convert_32bit_to_16bit(data: bytes) -> bytes:
    """Convert 32-bit stereo I2S to 16-bit mono PCM (extract XMOS AEC channel).
    Log both channels to find which has better AEC."""
    if len(data) % 4 != 0:
        return data
    all_samples = np.frombuffer(data, dtype='<i4')
    ch0 = all_samples[0::2]  # left = XMOS AEC processed (confirmed: R is silent)
    return (ch0 >> 16).astype('<i2').tobytes()


def process_gemini_audio(data: bytes) -> bytes:
    """Convert Gemini 24kHz 16-bit mono to device 48kHz 32-bit stereo.

    All numpy vectorized (C-speed):
    - 2x nearest-neighbor upsample (24kHz -> 48kHz)
    - mono to stereo (duplicate L=R)
    - 16-bit to 32-bit left-justified for I2S
    Each input sample becomes 4 output int32 values.
    """
    samples = np.frombuffer(data, dtype='<i2')
    if len(samples) < 1:
        return data
    # repeat 4x = 2x upsample * 2 channels, then left-justify to 32-bit
    return (np.repeat(samples.astype(np.int32), 4) << 16).tobytes()

# Gemini configuration
MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy websockets debug logging
logging.getLogger("websockets").setLevel(logging.WARNING)


class HomeAssistantClient:
    """Client for Home Assistant REST API"""

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
        """Call a Home Assistant service"""
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

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """Get the state of an entity"""
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


class GeminiSession:
    """Manages a Gemini Live API session"""

    def __init__(self, ha_client: HomeAssistantClient, session_id: str):
        self.ha_client = ha_client
        self.session_id = session_id
        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=GEMINI_API_KEY
        )

        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue(maxsize=50)
        self.session = None
        self.active = False
        self.playing = False  # True while sending audio to speaker (mute mic)

        # System instructions
        if CROATIAN_PERSONALITY:
            system_instruction = """Pozdrav! Ti si virtualni asistent u pametnoj kući.

VAŽNO:
- Govoriš isključivo hrvatski jezik
- Ženskog si roda
- Prijateljski i prirodan ton razgovora
- Kratki i jasni odgovori (ne duljiti)

MOGUĆNOSTI:
- Kontrola rasvjete (Shelly i Tuya integracije)
- Promjena boje svjetala
- Provjera statusa uređaja
- Odgovaranje na jednostavna pitanja

PRAVILA RAZGOVORA:
- Kada korisnik kaže "hvala", "to je sve", "doviđenja" ili prestane govoriti dulje vrijeme - pozovi funkciju end_conversation()
- Potvrdi svaku izvršenu akciju kratko i jasno
- Ako nešto ne možeš napraviti, iskreno reci

PRIMJERI:
Korisnik: "Upali svjetlo u kuhinji"
Ti: "Upalila sam svjetlo u kuhinji."

Korisnik: "Promijeni boju u plavu"
Ti: "Gotovo, svjetlo je sada plavo."

Korisnik: "Hvala, to je sve"
Ti: [pozovi end_conversation()] "Doviđenja!"
"""
        else:
            system_instruction = "You are a helpful home assistant. Control devices and answer questions."

        self.config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Zephyr"  # Newer voice
                    )
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=system_instruction)],
                role="user"
            ),
        )

        # Register function tools
        self.tools = self._create_function_tools()
        if self.tools:
            self.config.tools = self.tools

    def _create_function_tools(self):
        """Create function calling tools for Home Assistant control"""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="control_device",
                        description="Control a Home Assistant device (lights, switches, etc.)",
                        parameters={
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
                    ),
                    types.FunctionDeclaration(
                        name="query_device_state",
                        description="Get the current state of a Home Assistant device",
                        parameters={
                            "type": "object",
                            "properties": {
                                "entity_id": {
                                    "type": "string",
                                    "description": "The entity ID to query"
                                }
                            },
                            "required": ["entity_id"]
                        }
                    ),
                    types.FunctionDeclaration(
                        name="end_conversation",
                        description="End the conversation session and return to wake word mode. Call this when user says goodbye, thank you and indicates they're done, or after prolonged silence.",
                        parameters={
                            "type": "object",
                            "properties": {}
                        }
                    )
                ]
            )
        ]

    async def handle_function_call(self, function_call) -> Dict[str, Any]:
        """Handle function calls from Gemini"""
        function_name = function_call.name
        args = dict(function_call.args) if function_call.args else {}

        logger.info(f"Function call: {function_name}({args})")

        try:
            if function_name == "control_device":
                return await self._handle_control_device(args)
            elif function_name == "query_device_state":
                return await self._handle_query_device_state(args)
            elif function_name == "end_conversation":
                return await self._handle_end_conversation()
            else:
                return {"error": f"Unknown function: {function_name}"}
        except Exception as e:
            logger.error(f"Error handling function call: {e}", exc_info=True)
            return {"error": str(e)}

    async def _handle_control_device(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle control_device function call"""
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

        result = await self.ha_client.call_service(domain, service, **service_data)
        return result

    async def _handle_query_device_state(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query_device_state function call"""
        entity_id = args.get("entity_id")

        if not entity_id:
            return {"error": "Missing entity_id"}

        state = await self.ha_client.get_state(entity_id)
        return {"state": state.get("state"), "attributes": state.get("attributes", {})}

    async def _handle_end_conversation(self) -> Dict[str, Any]:
        """Handle end_conversation function call"""
        logger.info("End conversation requested by Gemini")
        self.active = False
        return {"success": True, "message": "Conversation ended"}

    async def send_audio_to_gemini(self):
        """Background task to send audio from device to Gemini"""
        chunks_to_gemini = 0
        try:
            logger.info("send_audio_to_gemini task started")
            while self.active:
                audio_data = await self.audio_out_queue.get()
                if audio_data is None:
                    break

                # Convert stereo 32-bit I2S → mono 16-bit PCM
                raw_audio = audio_data  # save for logging
                if DEVICE_BITS_PER_SAMPLE == 32:
                    audio_data = convert_32bit_to_16bit(audio_data)

                # Log both channels to compare AEC effectiveness
                chunks_to_gemini += 1
                if chunks_to_gemini % 50 == 0:
                    raw_samples = np.frombuffer(raw_audio, dtype='<i4') if len(raw_audio) % 4 == 0 else np.array([])
                    if len(raw_samples) >= 2:
                        p0 = int(np.abs(raw_samples[0::2] >> 16).max())
                        p1 = int(np.abs(raw_samples[1::2] >> 16).max())
                        playing = "SPEAKER_ON" if self.playing else "speaker_off"
                        logger.info(f"MIC #{chunks_to_gemini}: L={p0} R={p1} [{playing}]")

                await self.session.send_realtime_input(audio={"data": audio_data, "mime_type": "audio/pcm"})
            logger.info("Exited send loop, session no longer active")
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}", exc_info=True)
        finally:
            logger.info(f"send_audio_to_gemini ended, total chunks: {chunks_to_gemini}")

    async def receive_audio_from_gemini(self):
        """Background task to receive audio and function calls from Gemini"""
        chunks_from_gemini = 0
        try:
            logger.info("receive_audio_from_gemini task started")
            while self.active:
                logger.info("Waiting for Gemini turn...")
                turn = self.session.receive()

                async for response in turn:
                    # Handle audio data - queue immediately (like Google example)
                    if data := response.data:
                        chunks_from_gemini += 1
                        self.audio_in_queue.put_nowait(data)
                        continue

                    # Handle text (for debugging)
                    if text := response.text:
                        logger.info(f"Gemini text: {text}")

                    # Handle function calls
                    if response.tool_call:
                        logger.info(f"Got tool_call: {response.tool_call}")
                        for function_call in response.tool_call.function_calls:
                            result = await self.handle_function_call(function_call)

                            await self.session.send_tool_response(
                                function_responses=[
                                    types.FunctionResponse(
                                        id=function_call.id,
                                        name=function_call.name,
                                        response=result
                                    )
                                ]
                            )

                            if function_call.name == "end_conversation":
                                self.active = False
                                return

                # Turn complete - do NOT flush queue
                # XMOS echo cancellation should prevent self-interruption
                # but if it doesn't, flushing would discard valid audio
                logger.info(f"Turn complete, {chunks_from_gemini} Gemini chunks, queue: {self.audio_in_queue.qsize()}")

            logger.info("Exited receive loop, session no longer active")
        except Exception as e:
            logger.error(f"Error receiving from Gemini: {e}", exc_info=True)
        finally:
            logger.info(f"receive_audio_from_gemini ended, total chunks: {chunks_from_gemini}")

    async def start(self):
        """Start the Gemini session"""
        logger.info(f"Starting Gemini session {self.session_id}")

        self.active = True

        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=self.config) as session,
                asyncio.TaskGroup() as tg,
            ):
                logger.info("Connected to Gemini Live API")
                self.session = session

                tg.create_task(self.send_audio_to_gemini())
                tg.create_task(self.receive_audio_from_gemini())
                await asyncio.sleep(0)

                # Wait until session ends
                while self.active:
                    await asyncio.sleep(0.1)

                logger.info("Session ended, canceling tasks")
                raise asyncio.CancelledError("Session ended")

        except asyncio.CancelledError:
            logger.debug("Gemini session cancelled normally")
        except Exception as e:
            logger.error(f"Error in Gemini session: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Is API key set: {'Yes' if GEMINI_API_KEY else 'No'}")
        finally:
            self.active = False
            logger.info(f"Gemini session {self.session_id} terminated")


class DeviceConnection:
    """Handles connection from ESP32 device"""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 ha_client: HomeAssistantClient):
        self.reader = reader
        self.writer = writer
        self.ha_client = ha_client
        self.address = writer.get_extra_info('peername')
        self.session_id = f"device_{self.address[0]}_{self.address[1]}"

        self.gemini_session: Optional[GeminiSession] = None
        self.active = False

    async def handle(self):
        """Handle device connection"""
        logger.info(f"New device connection from {self.address}")

        try:
            self.active = True

            # Enable TCP_NODELAY for low-latency audio streaming
            sock = self.writer.transport.get_extra_info('socket')
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.info("TCP_NODELAY enabled")

            # Create Gemini session
            try:
                self.gemini_session = GeminiSession(self.ha_client, self.session_id)
                logger.info(f"Gemini session created for {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to create Gemini session: {e}", exc_info=True)
                raise

            # Mark session as active before starting tasks
            self.gemini_session.active = True

            # Start tasks
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.receive_from_device())
                    tg.create_task(self.send_to_device())
                    tg.create_task(self.gemini_session.start())

                    logger.info("All session tasks started")

                    # Give tasks time to start
                    await asyncio.sleep(0.1)

                    # Wait for session to end
                    while self.active and self.gemini_session.active:
                        await asyncio.sleep(0.1)

                    logger.info("Device session ending")
                    raise asyncio.CancelledError("Session ended")
            except* Exception as eg:
                logger.error(f"TaskGroup exception(s): {eg.exceptions}")
                for exc in eg.exceptions:
                    logger.error(f"  - {type(exc).__name__}: {exc}", exc_info=exc)
                raise

        except asyncio.CancelledError:
            logger.debug("Device connection cancelled")
        except Exception as e:
            logger.error(f"Error handling device: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
        finally:
            self.active = False
            await self.cleanup()

    async def receive_from_device(self):
        """Receive audio from device and send to Gemini"""
        chunks_from_device = 0
        try:
            logger.info("receive_from_device task started")
            while self.active:
                # Read length header (4 bytes)
                length_data = await self.reader.readexactly(4)
                length = struct.unpack('>I', length_data)[0]

                if length == 0:
                    logger.info("Received end signal from device")
                    self.active = False
                    break

                # Read audio data
                audio_data = await self.reader.readexactly(length)
                chunks_from_device += 1

                # Send to Gemini
                if self.gemini_session:
                    await self.gemini_session.audio_out_queue.put(audio_data)

        except asyncio.IncompleteReadError:
            logger.info("Device disconnected")
            self.active = False
        except Exception as e:
            logger.error(f"Error receiving from device: {e}", exc_info=True)
            self.active = False
        finally:
            logger.info(f"receive_from_device ended, total chunks: {chunks_from_device}")

    async def send_to_device(self):
        """Send raw 24kHz audio paced at real-time (like PyAudio stream.write).

        This is THE CLOCK - it paces audio delivery at exactly real-time speed.
        The queue absorbs Gemini's bursts, and we drain it steadily.
        """
        logger.info("send_to_device task started")
        chunks_sent = 0
        bytes_sent = 0
        try:
            while self.active:
                if not self.gemini_session:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    data = await asyncio.wait_for(
                        self.gemini_session.audio_in_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    self.gemini_session.playing = False
                    continue

                self.gemini_session.playing = True

                # THE CLOCK: pace at 60% real-time (like PyAudio stream.write)
                chunk_duration = len(data) / 48000.0 * 0.9

                header = struct.pack('>I', len(data))
                self.writer.write(header + data)
                await self.writer.drain()
                chunks_sent += 1
                bytes_sent += len(data)

                await asyncio.sleep(chunk_duration)

                # Mark not playing when queue is empty
                if self.gemini_session.audio_in_queue.empty():
                    self.gemini_session.playing = False

                if chunks_sent % 20 == 1:
                    qsize = self.gemini_session.audio_in_queue.qsize()
                    logger.info(f"Chunk #{chunks_sent}: {len(data)}B, queue: {qsize}, total: {bytes_sent}B")

        except Exception as e:
            logger.error(f"Error sending to device: {e}", exc_info=True)
            self.active = False
        finally:
            logger.info(f"send_to_device ended, {chunks_sent} chunks, {bytes_sent}B")

    async def cleanup(self):
        """Clean up connection"""
        logger.info(f"Cleaning up device connection from {self.address}")

        # Send end signal to device
        try:
            end_signal = struct.pack('>I', 0)
            self.writer.write(end_signal)
            await self.writer.drain()
        except:
            pass

        # Close connection
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except:
            pass


class BridgeServer:
    """Main bridge server"""

    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.ha_client = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle new client connection"""
        connection = DeviceConnection(reader, writer, self.ha_client)
        await connection.handle()

    async def start(self):
        """Start the bridge server"""
        logger.info("Starting Gemini Live Bridge...")

        # Validate configuration
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY not set!")
            sys.exit(1)

        if not HA_TOKEN:
            logger.warning("HA_TOKEN not set, using Supervisor token or connection may fail")

        # Initialize Home Assistant client
        self.ha_client = await HomeAssistantClient(HA_URL, HA_TOKEN).__aenter__()

        # Start TCP server
        self.server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.port
        )

        addr = self.server.sockets[0].getsockname()
        logger.info(f"Bridge server listening on {addr[0]}:{addr[1]}")

        async with self.server:
            await self.server.serve_forever()


async def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Gemini Live Bridge for Home Assistant")
    logger.info("=" * 60)
    logger.info(f"Server Port: {SERVER_PORT}")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Croatian Personality: {CROATIAN_PERSONALITY}")
    logger.info(f"Session Timeout: {SESSION_TIMEOUT}s")
    logger.info(f"Home Assistant URL: {HA_URL}")
    logger.info(f"HA Token available: {'Yes' if HA_TOKEN else 'No'}")
    logger.debug(f"SUPERVISOR_TOKEN env: {'set' if os.environ.get('SUPERVISOR_TOKEN') else 'from s6 file'}")
    logger.info("=" * 60)

    server = BridgeServer(SERVER_PORT)
    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
