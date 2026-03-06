"""OpenAI Realtime API provider for bidirectional audio."""

import asyncio
import base64
import json
import logging
from typing import Dict, Any

import aiohttp

from audio import convert_32bit_stereo_to_16bit_mono, resample_16k_to_24k
from .base import BaseProvider

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-4o-realtime-preview"


class OpenAIRealtimeProvider(BaseProvider):
    """Provider using OpenAI Realtime API for bidirectional audio.

    Audio format: 24kHz 16-bit PCM, base64 encoded in JSON WebSocket messages.
    Device sends 16kHz -> we resample to 24kHz for OpenAI.
    OpenAI sends 24kHz -> passthrough to ESP32 (same format).
    """

    def __init__(self, config: Dict[str, Any], ha_client, tool_registry, session_id: str):
        super().__init__(config, ha_client, tool_registry, session_id)

        self.api_key = config.get("openai_api_key")
        if not self.api_key:
            raise ValueError("openai_api_key is required for OpenAI provider")

        self.model = config.get("openai_model", DEFAULT_MODEL)
        self.ws = None
        self.ws_session = None

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

        voice_name = config.get("voice_name", "alloy")
        # Map Gemini voices to OpenAI equivalents
        openai_voices = {"alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"}
        if voice_name.lower() not in openai_voices:
            voice_name = "alloy"
        self.voice = voice_name.lower()

    def convert_device_audio(self, data: bytes) -> bytes:
        """Stereo 32-bit 16kHz -> mono 16-bit 24kHz for OpenAI."""
        mono_16k = convert_32bit_stereo_to_16bit_mono(data)
        return resample_16k_to_24k(mono_16k)

    async def start(self) -> None:
        """Connect to OpenAI Realtime WebSocket and run audio loops."""
        logger.info(f"Starting OpenAI Realtime session {self.session_id}")
        self.active = True

        url = f"{OPENAI_REALTIME_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            self.ws_session = aiohttp.ClientSession()
            self.ws = await self.ws_session.ws_connect(url, headers=headers, heartbeat=30)
            logger.info(f"Connected to OpenAI Realtime API ({self.model})")

            # Configure session
            await self._send_session_update()

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._send_audio_loop())
                tg.create_task(self._receive_loop())
                await asyncio.sleep(0)

                while self.active:
                    await asyncio.sleep(0.1)

                raise asyncio.CancelledError("Session ended")

        except asyncio.CancelledError:
            logger.debug("OpenAI session cancelled normally")
        except Exception as e:
            logger.error(f"Error in OpenAI session: {e}", exc_info=True)
        finally:
            self.active = False
            if self.ws and not self.ws.closed:
                await self.ws.close()
            if self.ws_session:
                await self.ws_session.close()
            logger.info(f"OpenAI session {self.session_id} terminated")

    async def _send_session_update(self):
        """Send session configuration to OpenAI."""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.system_instruction,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500,
                    "prefix_padding_ms": 300,
                },
                "tools": self.tool_registry.to_openai_tools(),
                "tool_choice": "auto",
                "temperature": 0.6,
            }
        }
        await self.ws.send_json(session_config)
        logger.info(f"Session configured: voice={self.voice}, tools={len(self.tool_registry.all_tools())}")

    async def _send_audio_loop(self):
        """Read device audio, resample, base64 encode, send to OpenAI."""
        chunks = 0
        try:
            logger.info("OpenAI send loop started")
            while self.active:
                audio_data = await self.audio_out_queue.get()
                if audio_data is None:
                    break

                # Convert: stereo 32-bit 16kHz -> mono 16-bit 24kHz
                pcm_24k = self.convert_device_audio(audio_data)

                # Base64 encode and send
                audio_b64 = base64.b64encode(pcm_24k).decode('ascii')
                await self.ws.send_json({
                    "type": "input_audio_buffer.append",
                    "audio": audio_b64,
                })
                chunks += 1

                if chunks % 50 == 0:
                    logger.info(f"Sent {chunks} audio chunks to OpenAI")

        except Exception as e:
            logger.error(f"Error sending audio to OpenAI: {e}", exc_info=True)
        finally:
            logger.info(f"OpenAI send loop ended, {chunks} chunks")

    async def _receive_loop(self):
        """Receive events from OpenAI WebSocket."""
        audio_chunks = 0
        try:
            logger.info("OpenAI receive loop started")
            async for msg in self.ws:
                if not self.active:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(msg.data)
                    event_type = event.get("type", "")

                    if event_type == "response.audio.delta":
                        # Decode base64 audio, queue for device
                        audio_b64 = event.get("delta", "")
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            self.audio_in_queue.put_nowait(audio_data)
                            audio_chunks += 1

                    elif event_type == "response.audio.done":
                        logger.info(f"Audio response complete, {audio_chunks} chunks")
                        audio_chunks = 0

                    elif event_type == "response.audio_transcript.delta":
                        pass  # Partial transcript, ignore

                    elif event_type == "response.audio_transcript.done":
                        transcript = event.get("transcript", "")
                        logger.info(f"Agent said: {transcript}")

                    elif event_type == "response.function_call_arguments.done":
                        await self._handle_function_call(event)

                    elif event_type == "input_audio_buffer.speech_started":
                        logger.info("User started speaking")
                        # Clear queued audio (user interrupted)
                        while not self.audio_in_queue.empty():
                            try:
                                self.audio_in_queue.get_nowait()
                            except:
                                break

                    elif event_type == "input_audio_buffer.speech_stopped":
                        logger.info("User stopped speaking")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = event.get("transcript", "")
                        logger.info(f"User said: {transcript}")

                    elif event_type == "error":
                        error = event.get("error", {})
                        logger.error(f"OpenAI error: {error}")

                    elif event_type == "session.created":
                        logger.info("OpenAI session created")

                    elif event_type == "session.updated":
                        logger.info("OpenAI session updated")

                    elif event_type == "response.done":
                        # Check for interruption
                        queue_size = self.audio_in_queue.qsize()
                        if queue_size > 10:
                            logger.info(f"Clearing {queue_size} chunks (possible interruption)")
                            while not self.audio_in_queue.empty():
                                try:
                                    self.audio_in_queue.get_nowait()
                                except:
                                    break

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.info(f"WebSocket closed/error: {msg.type}")
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

        # Execute tool
        result = await self.execute_tool(name, args)

        # Send tool output back to OpenAI
        await self.ws.send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            }
        })

        # Trigger response generation after tool output
        await self.ws.send_json({
            "type": "response.create",
        })

        logger.info(f"Tool response sent for {name}, call_id={call_id}")
