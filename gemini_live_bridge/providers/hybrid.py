"""Hybrid provider: Soniox STT + OpenAI Realtime (text in, audio out).

Pipeline: Mic audio -> Soniox (streaming STT) -> OpenAI Realtime (text -> audio) -> Speaker
Same as OpenAI Realtime provider but input is text from Soniox instead of raw audio.
"""

import asyncio
import base64
import json
import logging
from typing import Dict, Any, List

import aiohttp

from .base import BaseProvider

logger = logging.getLogger(__name__)

SONIOX_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"


class HybridProvider(BaseProvider):
    """Hybrid provider: Soniox STT feeds text into OpenAI Realtime API.

    OpenAI Realtime handles LLM + TTS in one WebSocket, outputting streamed audio.
    This gives us Soniox's superior multilingual STT with OpenAI's low-latency
    voice synthesis — no separate TTS API call needed.
    """

    def __init__(self, config: Dict[str, Any], ha_client, tool_registry, session_id: str):
        super().__init__(config, ha_client, tool_registry, session_id)

        self.soniox_api_key = config.get("soniox_api_key")
        if not self.soniox_api_key:
            raise ValueError("soniox_api_key is required for Hybrid provider")

        self.openai_api_key = config.get("openai_api_key")
        if not self.openai_api_key:
            raise ValueError("openai_api_key is required for Hybrid provider")

        self.model = config.get("openai_model", DEFAULT_MODEL)
        self.voice = config.get("openai_voice", "alloy").lower()
        self._audio_accumulator = b""

        # Build system instruction
        croatian = config.get("croatian_personality", True)
        if croatian:
            self.system_instruction = """Pozdrav! Ja sam Jarvis, tvoj virtualni asistent u pametnoj kući.

VAŽNO:
- Govoriš isključivo hrvatski jezik
- Muškog si roda (ja sam Jarvis)
- Prijateljski i prirodan ton razgovora
- Kratki i jasni odgovori (ne duljiti)
- Odgovaram na SVA pitanja - od kontrole kuće do općih znanja, matematike, programiranja, savjeta, itd.

MOGUĆNOSTI:
- Kontrola pametne kuće (rasvjeta, uređaji)
- Odgovaranje na bilo koja pitanja (opće znanje, matematika, savjeti, programiranje, itd.)
- Razgovori o bilo kojoj temi

PRAVILA RAZGOVORA:
- Kada korisnik kaže "hvala", "to je sve", "doviđenja" ili prestane govoriti dulje vrijeme - pozovi funkciju end_conversation()
- Potvrdi svaku izvršenu akciju kratko i jasno
- Uvijek pokušaj pomoći, bez obzira na temu pitanja
"""
        else:
            self.system_instruction = "You are a helpful home assistant. Control devices and answer questions."

        # Use custom prompt from dashboard if available
        custom_prompt = self.get_custom_prompt()
        if custom_prompt:
            self.system_instruction = custom_prompt

        # STT state
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.soniox_ws = None
        self.soniox_session = None

        # OpenAI Realtime state
        self.openai_ws = None
        self.openai_session = None

    async def start(self) -> None:
        """Start the hybrid pipeline: Soniox STT + OpenAI Realtime."""
        logger.info(f"Starting Hybrid session {self.session_id}")
        self.active = True

        # Append available devices to system instruction
        device_list = await self.fetch_device_list()
        if device_list:
            self.system_instruction += device_list

        # Connect to OpenAI Realtime
        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.openai_session = aiohttp.ClientSession()
            self.openai_ws = await self.openai_session.ws_connect(url, headers=headers, heartbeat=30)
            logger.info(f"Connected to OpenAI Realtime API ({self.model})")

            # Configure session — text input only, audio output
            await self._send_session_update()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._stt_stream_loop())
                tg.create_task(self._transcript_to_openai_loop())
                tg.create_task(self._openai_receive_loop())
                await asyncio.sleep(0)

                while self.active:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Session ended")

        except asyncio.CancelledError:
            logger.debug("Hybrid session cancelled normally")
        except Exception as e:
            logger.error(f"Error in Hybrid session: {e}", exc_info=True)
        finally:
            self.active = False
            if self.soniox_ws and not self.soniox_ws.closed:
                await self.soniox_ws.close()
            if self.soniox_session:
                await self.soniox_session.close()
            if self.openai_ws and not self.openai_ws.closed:
                await self.openai_ws.close()
            if self.openai_session:
                await self.openai_session.close()
            logger.info(f"Hybrid session {self.session_id} terminated")

    async def _send_session_update(self):
        """Configure OpenAI Realtime for text input + audio output."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.system_instruction,
                "voice": self.voice,
                "output_audio_format": "pcm16",
                "turn_detection": None,  # We handle turn detection via Soniox
                "tools": self.tool_registry.to_openai_tools(),
                "tool_choice": "auto",
                "temperature": 0.6,
            }
        }
        await self.openai_ws.send_json(session_config)
        logger.info(f"OpenAI session configured: voice={self.voice}, text input mode")

    # ── Soniox STT ──────────────────────────────────────────────

    async def _stt_stream_loop(self):
        """Stream mic audio to Soniox STT, emit transcripts."""
        try:
            self.soniox_session = aiohttp.ClientSession()
            self.soniox_ws = await self.soniox_session.ws_connect(SONIOX_WS_URL)
            logger.info("Connected to Soniox STT")

            stt_config = {
                "api_key": self.soniox_api_key,
                "model": "stt-rt-preview",
                "audio_format": "s16le",
                "sample_rate": 16000,
                "num_channels": 1,
                "language_hints": ["hr", "en"],
            }
            await self.soniox_ws.send_json(stt_config)
            logger.info("Soniox config sent")

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._stt_send_audio())
                tg.create_task(self._stt_receive_transcripts())

                while self.active:
                    await asyncio.sleep(0.1)
                raise asyncio.CancelledError()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Soniox STT error: {e}", exc_info=True)

    async def _stt_send_audio(self):
        """Send device audio to Soniox."""
        chunks = 0
        keepalive_interval = 15
        last_keepalive = asyncio.get_event_loop().time()

        try:
            while self.active:
                try:
                    audio_data = await asyncio.wait_for(
                        self.audio_out_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    now = asyncio.get_event_loop().time()
                    if now - last_keepalive > keepalive_interval:
                        await self.soniox_ws.send_json({"type": "keepalive"})
                        last_keepalive = now
                    continue

                if audio_data is None:
                    break

                mono_audio = self.convert_device_audio(audio_data)
                await self.soniox_ws.send_bytes(mono_audio)
                chunks += 1

                if chunks % 100 == 0:
                    logger.info(f"Sent {chunks} audio chunks to Soniox")

        except Exception as e:
            logger.error(f"Error sending to Soniox: {e}", exc_info=True)
        finally:
            logger.info(f"STT send ended, {chunks} chunks")

    async def _stt_receive_transcripts(self):
        """Receive transcripts from Soniox and queue them."""
        final_text = ""
        last_final_time = 0.0
        SENTENCE_PAUSE_MS = 800

        try:
            async for msg in self.soniox_ws:
                if not self.active:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    if data.get("error_message"):
                        logger.error(f"Soniox error: {data['error_message']}")
                        continue

                    tokens = data.get("tokens", [])
                    for token in tokens:
                        text = token.get("text", "")
                        is_final = token.get("is_final", False)

                        if is_final:
                            final_text += text
                            last_final_time = asyncio.get_event_loop().time()

                    if final_text and last_final_time > 0:
                        elapsed = (asyncio.get_event_loop().time() - last_final_time) * 1000
                        if elapsed >= SENTENCE_PAUSE_MS:
                            clean = final_text.strip()
                            if clean:
                                logger.info(f"STT transcript: '{clean}'")
                                await self.transcript_queue.put(clean)
                            final_text = ""

                    if data.get("finished"):
                        clean = final_text.strip()
                        if clean:
                            logger.info(f"STT final: '{clean}'")
                            await self.transcript_queue.put(clean)
                        final_text = ""

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.info("Soniox WebSocket closed")
                    break

        except Exception as e:
            logger.error(f"Error receiving from Soniox: {e}", exc_info=True)

    # ── OpenAI Realtime (text in, audio out) ────────────────────

    async def _transcript_to_openai_loop(self):
        """Take Soniox transcripts and send as text to OpenAI Realtime."""
        try:
            logger.info("Transcript->OpenAI loop started")
            while self.active:
                try:
                    transcript = await asyncio.wait_for(
                        self.transcript_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                logger.info(f"Sending text to OpenAI Realtime: '{transcript}'")

                # Create a user message with text content
                await self.openai_ws.send_json({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": transcript,
                            }
                        ]
                    }
                })

                # Trigger response generation
                await self.openai_ws.send_json({
                    "type": "response.create",
                })

        except Exception as e:
            logger.error(f"Transcript->OpenAI error: {e}", exc_info=True)
        finally:
            logger.info("Transcript->OpenAI loop ended")

    async def _openai_receive_loop(self):
        """Receive audio and events from OpenAI Realtime WebSocket.

        Same logic as openai_realtime.py _receive_loop.
        """
        audio_chunks = 0
        try:
            logger.info("OpenAI receive loop started")
            async for msg in self.openai_ws:
                if not self.active:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    event_type = event.get("type", "")

                    if event_type == "response.audio.delta":
                        self.playing = True
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            self._audio_accumulator += base64.b64decode(audio_b64)
                            audio_chunks += 1
                            CHUNK = 19200
                            while len(self._audio_accumulator) >= CHUNK:
                                chunk = self._audio_accumulator[:CHUNK]
                                self._audio_accumulator = self._audio_accumulator[CHUNK:]
                                try:
                                    self.audio_in_queue.put_nowait(chunk)
                                except asyncio.QueueFull:
                                    try:
                                        self.audio_in_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        pass
                                    self.audio_in_queue.put_nowait(chunk)

                    elif event_type == "response.audio.done":
                        if self._audio_accumulator:
                            CHUNK = 19200
                            padded = self._audio_accumulator + b'\x00' * (CHUNK - len(self._audio_accumulator))
                            try:
                                self.audio_in_queue.put_nowait(padded)
                            except asyncio.QueueFull:
                                pass
                            self._audio_accumulator = b""
                        logger.info(f"Audio response complete, {audio_chunks} deltas")
                        audio_chunks = 0

                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "")
                        logger.info(f"Agent said: {transcript}")

                    elif event_type == "response.function_call_arguments.done":
                        await self._handle_function_call(event)

                    elif event_type == "error":
                        error = event.get("error", {})
                        logger.error(f"OpenAI error: {error}")

                    elif event_type == "session.created":
                        logger.info("OpenAI Realtime session created")

                    elif event_type == "session.updated":
                        logger.info("OpenAI Realtime session updated")

                    elif event_type == "response.done":
                        logger.debug(f"Response done, queue: {self.audio_in_queue.qsize()}")

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.info(f"OpenAI WebSocket closed/error: {msg.type}")
                    self.active = False
                    break

        except Exception as e:
            logger.error(f"Error in OpenAI receive loop: {e}", exc_info=True)
        finally:
            logger.info("OpenAI receive loop ended")

    async def _handle_function_call(self, event: dict):
        """Handle a function call from OpenAI."""
        call_id = event.get("call_id", "")
        name = event.get("name", "")
        arguments_str = event.get("arguments", "{}")

        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Function call: {name}({args}), call_id={call_id}")

        result = await self.execute_tool(name, args)

        await self.openai_ws.send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        })

        await self.openai_ws.send_json({
            "type": "response.create",
        })

        logger.info(f"Tool response sent for {name}, call_id={call_id}")
